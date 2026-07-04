import base64
import io
from datetime import date, datetime
from decimal import Decimal

import langchain_ollama
import pypdfium2 as pdfium
import pytest

from invoicer.adapters.gemini_detector import InvoiceCheck
from invoicer.adapters.ollama_detector import OllamaInvoiceDetector, build_detection_message
from invoicer.adapters.ollama_extractor import OllamaVisionExtractor, build_extraction_message
from invoicer.adapters.ollama_reasoner import OllamaExceptionReasoner
from invoicer.extraction import InvoiceExtraction
from invoicer.models import (
    Classification,
    CountryBucket,
    Invoice,
    InvoiceDocument,
    LineItem,
    Party,
    TaxTreatment,
)
from invoicer.observability import LlmMetrics, LlmMetricsCallback
from invoicer.ports import ExceptionReasoner, InvoiceDetector, InvoiceExtractor
from invoicer.reasoning import ClassificationJudgment


def _pdf_bytes(pages: int = 2) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _doc(filename: str, content: bytes) -> InvoiceDocument:
    return InvoiceDocument(
        sender="a@b.pl", received_at=datetime(2026, 7, 4), filename=filename, content=content
    )


class _FakeStructured:
    def __init__(self, result):
        self.result = result
        self.received = None

    def invoke(self, messages):
        self.received = messages
        return self.result


class _FakeLLM:
    """Fake ChatOllama: zapamietuje schema i method z with_structured_output."""

    def __init__(self, result):
        self.structured = _FakeStructured(result)
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method=None):
        self.schema = schema
        self.method = method
        return self.structured


# --- budowanie wiadomosci (Ollama przyjmuje tylko obrazy — PDF rasteryzowany) ---


def test_pdf_message_has_prompt_and_png_page_blocks():
    msg = build_extraction_message(_doc("faktura.pdf", _pdf_bytes(pages=2)))
    blocks = msg.content
    assert blocks[0]["type"] == "text"
    images = [b for b in blocks[1:] if b["type"] == "image"]
    assert len(images) == 2  # strona = obraz
    for block in images:
        assert block["mime_type"] == "image/png"
        assert base64.b64decode(block["base64"]).startswith(b"\x89PNG")


def test_png_scan_passes_through_unchanged():
    msg = build_extraction_message(_doc("skan.png", b"\x89PNG surowe"))
    assert msg.content[1]["type"] == "image"
    assert msg.content[1]["mime_type"] == "image/png"
    assert base64.b64decode(msg.content[1]["base64"]) == b"\x89PNG surowe"


def test_jpeg_scan_mime():
    msg = build_extraction_message(_doc("skan.jpeg", b"\xff\xd8\xff"))
    assert msg.content[1]["mime_type"] == "image/jpeg"


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Nieobslugiwany"):
        build_extraction_message(_doc("dokument.tiff", b"x"))


def test_detection_message_mirrors_extraction_format():
    blocks = build_detection_message(_doc("faktura.pdf", _pdf_bytes(pages=1))).content
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image"
    assert blocks[1]["mime_type"] == "image/png"


# --- adaptery: protokoly, structured output przez json_schema, modele domyslne ---


def _extraction() -> InvoiceExtraction:
    from invoicer.extraction import LineItemExtraction, PartyExtraction

    return InvoiceExtraction(
        seller=PartyExtraction(name="ACME", nip="5260001246", country="PL"),
        buyer=PartyExtraction(name="Klient", country="PL"),
        number="FV/1",
        issue_date="2026-07-04",
        currency="PLN",
        lines=[
            LineItemExtraction(
                description="Usluga",
                quantity="1",
                unit_net="1000.00",
                vat_rate="0.23",
                net="1000.00",
                vat="230.00",
                gross="1230.00",
            )
        ],
        total_net="1000.00",
        total_vat="230.00",
        total_gross="1230.00",
        confidence=0.9,
    )


def test_extractor_satisfies_protocol_and_uses_json_schema():
    llm = _FakeLLM(_extraction())
    extractor = OllamaVisionExtractor(llm=llm)
    assert isinstance(extractor, InvoiceExtractor)
    inv = extractor.extract(_doc("faktura.pdf", _pdf_bytes(pages=1)))
    assert isinstance(inv, Invoice)
    assert inv.total_gross == Decimal("1230.00")
    assert llm.schema is InvoiceExtraction
    # ChatOllama: constrained decoding przez format (json_schema), nie tool-calling
    assert llm.method == "json_schema"


def test_detector_satisfies_protocol_and_uses_json_schema():
    llm = _FakeLLM(InvoiceCheck(is_invoice=True, reason="faktura"))
    detector = OllamaInvoiceDetector(llm=llm)
    assert isinstance(detector, InvoiceDetector)
    assert detector.is_invoice(_doc("f.pdf", _pdf_bytes(pages=1))) is True
    assert llm.schema is InvoiceCheck
    assert llm.method == "json_schema"


def _foreign_invoice() -> Invoice:
    line = LineItem(
        description="SaaS",
        quantity=Decimal("1"),
        unit_net=Decimal("100.00"),
        vat_rate=Decimal("0.00"),
        net=Decimal("100.00"),
        vat=Decimal("0.00"),
        gross=Decimal("100.00"),
    )
    return Invoice(
        seller=Party(name="Foreign Ltd", country="GB", vat_id="GB1"),
        buyer=Party(name="Klient", nip="5260001246", country="PL"),
        number="INV/1",
        issue_date=date(2026, 7, 4),
        currency="GBP",
        lines=[line],
        total_net=Decimal("100.00"),
        total_vat=Decimal("0.00"),
        total_gross=Decimal("100.00"),
    )


def test_reasoner_satisfies_protocol_and_uses_json_schema():
    judgment = ClassificationJudgment(
        treatment=TaxTreatment.IMPORT_USLUG,
        confidence=0.8,
        rationale_pl="SaaS -> import uslug",
        human_must_confirm=[],
        currency_note="",
    )
    llm = _FakeLLM(judgment)
    reasoner = OllamaExceptionReasoner(llm=llm)
    assert isinstance(reasoner, ExceptionReasoner)
    base = Classification(
        treatment=TaxTreatment.IMPORT_USLUG, country_bucket=CountryBucket.POZA_UE, confidence=0.6
    )
    out = reasoner.reason(_foreign_invoice(), base)
    assert out.treatment == TaxTreatment.IMPORT_USLUG
    assert out.country_bucket == CountryBucket.POZA_UE
    assert llm.schema is ClassificationJudgment
    assert llm.method == "json_schema"


def test_default_models_are_qwen3_vl():
    assert OllamaVisionExtractor()._model == "qwen3-vl:8b-instruct"
    assert OllamaInvoiceDetector()._model == "qwen3-vl:8b-instruct"
    assert OllamaExceptionReasoner()._model == "qwen3-vl:8b-instruct"


# --- wiring: leniwy ChatOllama dostaje model i callbacks ---


class _RecordingChat:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _RecordingChat.last_kwargs = kwargs

    def with_structured_output(self, schema, *, method=None):
        return self


@pytest.mark.parametrize(
    "factory", [OllamaVisionExtractor, OllamaInvoiceDetector, OllamaExceptionReasoner]
)
def test_adapter_passes_model_and_callbacks_to_chatollama(monkeypatch, factory):
    monkeypatch.setattr(langchain_ollama, "ChatOllama", _RecordingChat)
    _RecordingChat.last_kwargs = {}
    cb = LlmMetricsCallback(LlmMetrics(), model="qwen3-vl:8b-instruct")
    factory(callbacks=[cb])._client()
    assert _RecordingChat.last_kwargs["model"] == "qwen3-vl:8b-instruct"
    assert _RecordingChat.last_kwargs["callbacks"] == [cb]
    # Qwen3 mysli domyslnie, a thinking + constrained JSON (format) = puste content
    # uciete limitem tokenow (done_reason=length) — reasoning MUSI byc wylaczony.
    assert _RecordingChat.last_kwargs["reasoning"] is False
    # Domyslny kontekst Ollamy jest maly; wielostronicowy PDF (obrazy) by sie nie zmiescil.
    assert _RecordingChat.last_kwargs["num_ctx"] == 16384
    assert _RecordingChat.last_kwargs["num_predict"] == 4096
    # Modelfile qwen3-vl ma temperature=1 — ekstrakcja danych musi byc deterministyczna,
    # inaczej pola typu quantity losowo dostaja jednostki ('80 hrs', 'time:80 hours').
    assert _RecordingChat.last_kwargs["temperature"] == 0
