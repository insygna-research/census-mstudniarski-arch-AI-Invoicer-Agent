import pytest

from invoicer.adapters.provider import (
    build_detector,
    build_extractor,
    build_reasoner,
    provider_name,
)


def test_default_provider_is_gemini(monkeypatch):
    monkeypatch.delenv("INVOICER_LLM_PROVIDER", raising=False)
    assert provider_name() == "gemini"
    from invoicer.adapters.gemini_detector import GeminiInvoiceDetector
    from invoicer.adapters.gemini_extractor import GeminiVisionExtractor
    from invoicer.adapters.gemini_reasoner import GeminiExceptionReasoner

    assert isinstance(build_extractor(), GeminiVisionExtractor)
    assert isinstance(build_detector(), GeminiInvoiceDetector)
    assert isinstance(build_reasoner(), GeminiExceptionReasoner)


def test_ollama_provider_builds_ollama_adapters(monkeypatch):
    monkeypatch.setenv("INVOICER_LLM_PROVIDER", "ollama")
    from invoicer.adapters.ollama_detector import OllamaInvoiceDetector
    from invoicer.adapters.ollama_extractor import OllamaVisionExtractor
    from invoicer.adapters.ollama_reasoner import OllamaExceptionReasoner

    assert isinstance(build_extractor(), OllamaVisionExtractor)
    assert isinstance(build_detector(), OllamaInvoiceDetector)
    assert isinstance(build_reasoner(), OllamaExceptionReasoner)


def test_provider_name_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("INVOICER_LLM_PROVIDER", " Ollama ")
    assert provider_name() == "ollama"


def test_unknown_provider_raises_loudly(monkeypatch):
    monkeypatch.setenv("INVOICER_LLM_PROVIDER", "bard")
    with pytest.raises(ValueError, match="INVOICER_LLM_PROVIDER"):
        build_extractor()


def test_callbacks_are_forwarded(monkeypatch):
    monkeypatch.delenv("INVOICER_LLM_PROVIDER", raising=False)
    from invoicer.observability import LlmMetrics, LlmMetricsCallback

    cb = LlmMetricsCallback(LlmMetrics(), model="gemini-2.5-flash")
    assert build_extractor(callbacks=[cb])._callbacks == [cb]
