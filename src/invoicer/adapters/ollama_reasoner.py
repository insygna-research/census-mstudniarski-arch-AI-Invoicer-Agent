"""ExceptionReasoner na lokalnym modelu przez Ollame.

Tekstowy osad VAT — wiadomosc identyczna jak w adapterze Gemini
(build_reason_message jest provider-neutralne). Structured output przez
method="json_schema" (constrained decoding).
"""

from __future__ import annotations

from typing import Any

from invoicer.adapters.gemini_reasoner import build_reason_message
from invoicer.adapters.ollama_extractor import DEFAULT_MODEL
from invoicer.models import Classification, Invoice
from invoicer.rag.models import RetrievedChunk
from invoicer.reasoning import ClassificationJudgment, judgment_to_classification


class OllamaExceptionReasoner:
    """ExceptionReasoner oparty o lokalny model (Ollama) + structured output.

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

    def reason(
        self, invoice: Invoice, base: Classification, context: list[RetrievedChunk] | None = None
    ) -> Classification:
        message = build_reason_message(invoice, context)
        structured = self._client().with_structured_output(
            ClassificationJudgment, method="json_schema"
        )
        judgment = structured.invoke([message])
        return judgment_to_classification(judgment, base.country_bucket)
