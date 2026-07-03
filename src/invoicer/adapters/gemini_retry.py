"""Retry z odczekaniem na przejsciowe bledy Gemini API (429, 500, 503).

Darmowy tier Gemini ma limity per model (RPM + pula dzienna), a langchain-google-genai
nie ponawia 429; do tego 503 "high demand" wycieka jako surowy
google.genai.errors.ServerError (bez opakowania w ChatGoogleGenerativeAIError).
Burst (batch evali, kilka faktur w jednym intake'u) wywracalby caly przebieg —
odczekujemy okno limitu i ponawiamy; inne bledy leca dalej bez ponawiania.
Wyczerpana pula DZIENNA tez zglasza 429 — po wyczerpaniu retries wyjatek idzie dalej.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_WAIT_S = 35.0  # RetryInfo Google przy 429 sugeruje ~30 s; margines 5 s
_RETRIES = 6  # "high demand" potrafi trwac pare minut — 6 x 35 s to ~3,5 min pokrycia
_RETRYABLE_CODES = (429, 500, 503)


def _is_retryable(error: Exception) -> bool:
    import httpx
    from google.genai import errors as genai_errors
    from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

    if isinstance(error, ChatGoogleGenerativeAIError):
        return "RESOURCE_EXHAUSTED" in str(error)
    if isinstance(error, genai_errors.APIError):
        return error.code in _RETRYABLE_CODES
    # zerwane polaczenie / timeout ponizej warstwy google.genai (obserwowane przy
    # "high demand": serwer zamyka polaczenie bez odpowiedzi)
    return isinstance(error, httpx.TransportError)


def call_with_rate_limit_retry[T](
    fn: Callable[[], T],
    *,
    retries: int = _RETRIES,
    wait_s: float = _WAIT_S,
    sleep: Callable[[float], object] | None = None,
) -> T:
    """Wykonuje fn(); na 429/500/503 z Gemini odczekuje wait_s i ponawia (max retries razy)."""
    do_sleep = time.sleep if sleep is None else sleep
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            if not _is_retryable(error) or attempt == retries:
                raise
            do_sleep(wait_s)
    raise AssertionError("unreachable")
