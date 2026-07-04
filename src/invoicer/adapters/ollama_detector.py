"""InvoiceDetector na lokalnym Qwen3-VL przez Ollame (por. ollama_extractor)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from invoicer.adapters.gemini_detector import DETECTION_PROMPT, InvoiceCheck
from invoicer.adapters.ollama_extractor import DEFAULT_MODEL, document_image_blocks
from invoicer.models import InvoiceDocument


def build_detection_message(document: InvoiceDocument) -> HumanMessage:
    """Prompt detekcji + obrazy stron (ten sam prompt co adapter Gemini)."""
    return HumanMessage(
        content=[{"type": "text", "text": DETECTION_PROMPT}, *document_image_blocks(document)]
    )


class OllamaInvoiceDetector:
    """InvoiceDetector oparty o lokalny model vision (Ollama) + structured output.

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

            # Konfiguracja jak w ollama_extractor (reasoning off + wieksze limity) —
            # uzasadnienie przy tamtym _client().
            self._llm = ChatOllama(
                model=self._model,
                callbacks=self._callbacks,
                reasoning=False,
                num_ctx=16384,
                num_predict=4096,
                temperature=0,
            )
        return self._llm

    def is_invoice(self, document: InvoiceDocument) -> bool:
        message = build_detection_message(document)
        structured = self._client().with_structured_output(InvoiceCheck, method="json_schema")
        check = structured.invoke([message])
        return check.is_invoice
