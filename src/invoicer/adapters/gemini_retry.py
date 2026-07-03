"""Retry z odczekaniem na 429/RESOURCE_EXHAUSTED z Gemini API.

Darmowy tier Gemini ma limity per model (RPM + pula dzienna), a langchain-google-genai
nie ponawia 429 — burst (batch evali, kilka faktur w jednym intake'u) wywracalby caly
przebieg. Odczekujemy okno limitu i ponawiamy; inne bledy leca dalej bez ponawiania.
Wyczerpana pula DZIENNA tez zglasza 429 — po wyczerpaniu retries wyjatek idzie dalej.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_WAIT_S = 35.0  # RetryInfo Google przy 429 sugeruje ~30 s; margines 5 s
_RETRIES = 4


def _is_rate_limit(error: Exception) -> bool:
    return "RESOURCE_EXHAUSTED" in str(error)


def call_with_rate_limit_retry[T](
    fn: Callable[[], T],
    *,
    retries: int = _RETRIES,
    wait_s: float = _WAIT_S,
    sleep: Callable[[float], object] | None = None,
) -> T:
    """Wykonuje fn(); na 429 z Gemini odczekuje wait_s i ponawia (max retries razy)."""
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    do_sleep = time.sleep if sleep is None else sleep
    for attempt in range(retries + 1):
        try:
            return fn()
        except ChatGoogleGenerativeAIError as error:
            if not _is_rate_limit(error) or attempt == retries:
                raise
            do_sleep(wait_s)
    raise AssertionError("unreachable")
