"""Wybor providera LLM dla lokalnych sciezek (runner, skrypty).

INVOICER_LLM_PROVIDER: "gemini" (domyslnie, hostowany darmowy tier z dziennymi
kwotami) albo "ollama" (lokalny Qwen3-VL — bez limitow, faktury nie opuszczaja
maszyny). Produkcyjny app.py celowo NIE korzysta z przelacznika: Fly jezdzi na
Gemini (lokalny model nie zmiesci sie na 512 MB VM).
"""

from __future__ import annotations

import os

from invoicer.ports import ExceptionReasoner, InvoiceDetector, InvoiceExtractor

_ENV = "INVOICER_LLM_PROVIDER"


def provider_name() -> str:
    return os.getenv(_ENV, "gemini").strip().lower()


def _unknown(name: str) -> ValueError:
    return ValueError(f"{_ENV}={name!r} — obslugiwane: 'gemini' (domyslnie), 'ollama'.")


def build_extractor(*, callbacks: list | None = None) -> InvoiceExtractor:
    name = provider_name()
    if name == "gemini":
        from invoicer.adapters.gemini_extractor import GeminiVisionExtractor

        return GeminiVisionExtractor(callbacks=callbacks)
    if name == "ollama":
        from invoicer.adapters.ollama_extractor import OllamaVisionExtractor

        return OllamaVisionExtractor(callbacks=callbacks)
    raise _unknown(name)


def build_detector(*, callbacks: list | None = None) -> InvoiceDetector:
    name = provider_name()
    if name == "gemini":
        from invoicer.adapters.gemini_detector import GeminiInvoiceDetector

        return GeminiInvoiceDetector(callbacks=callbacks)
    if name == "ollama":
        from invoicer.adapters.ollama_detector import OllamaInvoiceDetector

        return OllamaInvoiceDetector(callbacks=callbacks)
    raise _unknown(name)


def build_reasoner(*, callbacks: list | None = None) -> ExceptionReasoner:
    name = provider_name()
    if name == "gemini":
        from invoicer.adapters.gemini_reasoner import GeminiExceptionReasoner

        return GeminiExceptionReasoner(callbacks=callbacks)
    if name == "ollama":
        from invoicer.adapters.ollama_reasoner import OllamaExceptionReasoner

        return OllamaExceptionReasoner(callbacks=callbacks)
    raise _unknown(name)
