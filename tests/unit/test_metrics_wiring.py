import langchain_google_genai
import pytest

from invoicer.adapters.gemini_detector import GeminiInvoiceDetector
from invoicer.adapters.gemini_extractor import GeminiVisionExtractor
from invoicer.adapters.gemini_reasoner import GeminiExceptionReasoner
from invoicer.observability import LlmMetrics, LlmMetricsCallback

_FACTORIES = [GeminiVisionExtractor, GeminiInvoiceDetector, GeminiExceptionReasoner]


class _RecordingChat:
    """Podstawiany w miejsce ChatGoogleGenerativeAI — zapamietuje kwargs konstruktora."""

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        _RecordingChat.last_kwargs = kwargs

    def with_structured_output(self, schema):
        return self


def _cb() -> LlmMetricsCallback:
    return LlmMetricsCallback(LlmMetrics(), model="gemini-2.5-flash")


@pytest.mark.parametrize("factory", _FACTORIES)
def test_adapter_passes_callbacks_to_chatgooglegenai(monkeypatch, factory):
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", _RecordingChat)
    _RecordingChat.last_kwargs = {}  # brak forwardu -> glosny KeyError, nie ciche stale
    cb = _cb()
    factory(callbacks=[cb])._client()
    assert _RecordingChat.last_kwargs["model"] == "gemini-2.5-flash"
    assert _RecordingChat.last_kwargs["callbacks"] == [cb]


@pytest.mark.parametrize("factory", _FACTORIES)
def test_adapter_default_callbacks_is_none(monkeypatch, factory):
    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", _RecordingChat)
    _RecordingChat.last_kwargs = {}  # brak forwardu -> glosny KeyError, nie ciche stale
    factory()._client()
    assert _RecordingChat.last_kwargs["callbacks"] is None
