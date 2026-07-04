"""InvoiceExtractor na lokalnym Qwen3-VL przez Ollame.

Darmowy bez limitow i prywatny (faktury nie opuszczaja maszyny) — do dev/testow
lokalnych; produkcyjny Fly (512 MB) lokalnego modelu nie udzwignie. Ollama nie
przyjmuje PDF — strony rasteryzujemy do PNG (pdf_render). Structured output przez
method="json_schema" (natywne constrained decoding Ollamy; pewniejsze niz
tool-calling na malych modelach).
"""

from __future__ import annotations

import base64
from typing import Any

from langchain_core.messages import HumanMessage

from invoicer.adapters.gemini_extractor import EXTRACTION_PROMPT
from invoicer.adapters.pdf_render import pdf_to_png_pages
from invoicer.extraction import InvoiceExtraction, extraction_to_invoice
from invoicer.models import Invoice, InvoiceDocument

# Koniecznie wariant -instruct: goly tag qwen3-vl:8b to model MYSLACY, ktory przy
# obrazach + constrained JSON przepala caly budzet tokenow na niewidoczny thinking
# (done_reason=length, puste content) — nawet z reasoning=False.
DEFAULT_MODEL = "qwen3-vl:8b-instruct"


def document_image_blocks(document: InvoiceDocument) -> list[dict]:
    """Dokument -> bloki obrazow: PDF strona po stronie (PNG), skany wprost."""
    lower = document.filename.lower()
    if lower.endswith(".pdf"):
        pages = pdf_to_png_pages(document.content)
        images = [("image/png", page) for page in pages]
    elif lower.endswith(".png"):
        images = [("image/png", document.content)]
    elif lower.endswith((".jpg", ".jpeg")):
        images = [("image/jpeg", document.content)]
    else:
        raise ValueError(
            f"Nieobslugiwany typ pliku do ekstrakcji: {document.filename!r} "
            "(obslugiwane: pdf, png, jpg)"
        )
    return [
        {"type": "image", "base64": base64.b64encode(data).decode("utf-8"), "mime_type": mime}
        for mime, data in images
    ]


def build_extraction_message(document: InvoiceDocument) -> HumanMessage:
    """Prompt + obrazy stron dokumentu (ten sam prompt co adapter Gemini)."""
    return HumanMessage(
        content=[{"type": "text", "text": EXTRACTION_PROMPT}, *document_image_blocks(document)]
    )


class OllamaVisionExtractor:
    """InvoiceExtractor oparty o lokalny model vision (Ollama) + structured output.

    LLM wstrzykiwalny (CI: fake); domyslnie leniwie ChatOllama(model).
    """

    def __init__(
        self, *, model: str = DEFAULT_MODEL, llm: Any = None, callbacks: list | None = None
    ) -> None:
        self._model = model
        self._llm = llm
        self._callbacks = callbacks

    def _client(self):
        if self._llm is None:
            from langchain_ollama import ChatOllama

            # reasoning=False: Qwen3 mysli domyslnie, a thinking + constrained JSON
            # (format) konczy sie pustym content ucietym limitem (done_reason=length).
            # num_ctx/num_predict: domyslne Ollamy sa za male na obrazy stron + dlugi JSON.
            self._llm = ChatOllama(
                model=self._model,
                callbacks=self._callbacks,
                reasoning=False,
                num_ctx=16384,
                num_predict=4096,
                temperature=0,
            )
        return self._llm

    def extract(self, document: InvoiceDocument) -> Invoice:
        message = build_extraction_message(document)
        structured = self._client().with_structured_output(InvoiceExtraction, method="json_schema")
        extraction = structured.invoke([message])
        return extraction_to_invoice(extraction)
