from datetime import datetime

import pytest
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from invoicer.adapters.gemini_detector import GeminiInvoiceDetector, InvoiceCheck
from invoicer.adapters.gemini_retry import call_with_rate_limit_retry
from invoicer.models import InvoiceDocument

_QUOTA_MSG = (
    "Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}"
)


def _quota_error() -> ChatGoogleGenerativeAIError:
    return ChatGoogleGenerativeAIError(_QUOTA_MSG)


def test_returns_result_when_no_error():
    sleeps: list[float] = []
    assert call_with_rate_limit_retry(lambda: "ok", sleep=sleeps.append) == "ok"
    assert sleeps == []


def test_retries_after_quota_error_then_succeeds():
    sleeps: list[float] = []
    attempts = iter([_quota_error(), "ok"])

    def flaky():
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_rate_limit_retry(flaky, wait_s=35.0, sleep=sleeps.append) == "ok"
    assert sleeps == [35.0]


def test_reraises_after_exhausted_retries():
    sleeps: list[float] = []

    def always_limited():
        raise _quota_error()

    with pytest.raises(ChatGoogleGenerativeAIError):
        call_with_rate_limit_retry(always_limited, retries=2, sleep=sleeps.append)
    assert len(sleeps) == 2


def test_non_quota_gemini_error_is_not_retried():
    sleeps: list[float] = []

    def broken():
        raise ChatGoogleGenerativeAIError(
            "Error calling model 'gemini-2.5-flash' (INVALID_ARGUMENT): 400 INVALID_ARGUMENT."
        )

    with pytest.raises(ChatGoogleGenerativeAIError):
        call_with_rate_limit_retry(broken, sleep=sleeps.append)
    assert sleeps == []


def test_other_exception_types_are_not_caught():
    sleeps: list[float] = []

    def crash():
        raise ValueError("nie-Gemini blad")

    with pytest.raises(ValueError):
        call_with_rate_limit_retry(crash, sleep=sleeps.append)
    assert sleeps == []


class _FlakyStructured:
    """Fake structured-output: pierwsze invoke -> 429, potem wynik."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise _quota_error()
        return self._result


class _FlakyLLM:
    def __init__(self, result):
        self.structured = _FlakyStructured(result)

    def with_structured_output(self, _schema):
        return self.structured


def test_detector_retries_quota_error(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr("invoicer.adapters.gemini_retry.time.sleep", recorded.append)
    llm = _FlakyLLM(InvoiceCheck(is_invoice=True, reason="faktura"))
    doc = InvoiceDocument(
        sender="a@b.pl", received_at=datetime(2026, 7, 3), filename="f.pdf", content=b"%PDF"
    )
    assert GeminiInvoiceDetector(llm=llm).is_invoice(doc) is True
    assert len(recorded) == 1  # jeden 429 -> jedno odczekanie, potem sukces
