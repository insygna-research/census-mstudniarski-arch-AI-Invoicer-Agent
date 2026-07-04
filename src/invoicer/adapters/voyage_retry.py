"""Retry z odczekaniem na przejsciowe bledy Voyage.

Darmowy tier Voyage to 3 RPM / 10K TPM — batch evali albo zbieg zapytan intake'u
latwo go przekracza; do tego serwer potrafi zerwac polaczenie bez odpowiedzi
(APIConnectionError/RemoteDisconnected) albo zglosic 503. Zamiast wywracac caly
przebieg, odczekujemy pelne okno limitu i ponawiamy. Inne bledy (auth, walidacja)
leca dalej bez ponawiania.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_WAIT_S = 21.0  # 3 RPM = 1 zapytanie / 20 s; sekunda marginesu
_RETRIES = 5


def call_with_rate_limit_retry[T](
    fn: Callable[[], T],
    *,
    retries: int = _RETRIES,
    wait_s: float = _WAIT_S,
    sleep: Callable[[float], object] | None = None,
) -> T:
    """Wykonuje fn(); na przejsciowy blad Voyage odczekuje wait_s i ponawia (max retries)."""
    from voyageai import error as voyage_error

    retryable = (
        voyage_error.RateLimitError,
        voyage_error.APIConnectionError,
        voyage_error.ServiceUnavailableError,
    )
    do_sleep = time.sleep if sleep is None else sleep
    for attempt in range(retries + 1):
        try:
            return fn()
        except retryable:
            if attempt == retries:
                raise
            do_sleep(wait_s)
    raise AssertionError("unreachable")
