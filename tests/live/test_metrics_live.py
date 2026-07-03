import os
from datetime import datetime
from pathlib import Path

import pytest

from invoicer.adapters.gemini_detector import GeminiInvoiceDetector
from invoicer.models import InvoiceDocument
from invoicer.observability import LlmMetrics, LlmMetricsCallback

_FIXTURE = Path(__file__).parent / "fixtures" / "sample_invoice.pdf"

pytestmark = pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY") or not _FIXTURE.exists(),
    reason="wymaga GOOGLE_API_KEY oraz tests/live/fixtures/sample_invoice.pdf (test live)",
)


def test_metrics_captured_on_real_detection():
    metrics = LlmMetrics()
    cb = LlmMetricsCallback(metrics, model="gemini-2.5-flash")
    detector = GeminiInvoiceDetector(callbacks=[cb])
    doc = InvoiceDocument(
        sender="a@b.pl",
        received_at=datetime(2026, 6, 23),
        filename="sample_invoice.pdf",
        content=_FIXTURE.read_bytes(),
    )
    detector.is_invoice(doc)
    assert metrics.calls, "callback nie zarejestrowal zadnego wywolania"
    call = metrics.calls[0]
    assert call.model == "gemini-2.5-flash"
    assert call.input_tokens > 0
    assert call.output_tokens > 0
    assert call.cost_usd > 0
    assert call.latency_ms > 0
