import langchain_google_genai
import pytest

from invoicer.adapters.gemini_detector import GeminiInvoiceDetector
from invoicer.adapters.gemini_extractor import GeminiVisionExtractor
from invoicer.adapters.gemini_reasoner import GeminiExceptionReasoner
from invoicer.observability import LlmMetrics, LlmMetricsCallback

# Split modeli: detektor (proste tak/nie) na flash-lite oszczedza ciasna dzienna pule
# darmowego tieru; ekstrakcja vision i osad VAT na flash (jakosc — lite mylil wnt).
_FACTORIES = [
    (GeminiVisionExtractor, "gemini-2.5-flash"),
    (GeminiInvoiceDetector, "gemini-2.5-flash-lite"),
    (GeminiExceptionReasoner, "gemini-2.5-flash"),
]


class _RecordingChat:
    """Podstawiany w miejsce ChatGoogleGenerativeAI — zapamietuje kwargs konstruktora."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _RecordingChat.last_kwargs = kwargs

    def with_structured_output(self, schema):
        return self


def _cb(model: str) -> LlmMetricsCallback:
    return LlmMetricsCallback(LlmMetrics(), model=model)


@pytest.mark.parametrize(("factory", "expected_model"), _FACTORIES)
def test_adapter_passes_callbacks_to_chatgooglegenai(monkeypatch, factory, expected_model):
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", _RecordingChat)
    _RecordingChat.last_kwargs = {}  # brak forwardu -> glosny KeyError, nie ciche stale
    cb = _cb(expected_model)
    factory(callbacks=[cb])._client()
    assert _RecordingChat.last_kwargs["model"] == expected_model
    assert _RecordingChat.last_kwargs["callbacks"] == [cb]


@pytest.mark.parametrize(("factory", "expected_model"), _FACTORIES)
def test_adapter_default_callbacks_is_none(monkeypatch, factory, expected_model):
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", _RecordingChat)
    _RecordingChat.last_kwargs = {}  # brak forwardu -> glosny KeyError, nie ciche stale
    factory()._client()
    assert _RecordingChat.last_kwargs["callbacks"] is None
