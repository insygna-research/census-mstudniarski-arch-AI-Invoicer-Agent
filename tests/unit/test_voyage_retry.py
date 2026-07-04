import pytest
from voyageai import error as voyage_error

from invoicer.adapters.voyage_embedder import VoyageEmbedder
from invoicer.adapters.voyage_reranker import VoyageReranker
from invoicer.adapters.voyage_retry import call_with_rate_limit_retry


def _rate_limit() -> voyage_error.RateLimitError:
    return voyage_error.RateLimitError("3 RPM na darmowym tierze")


def test_returns_result_when_no_error():
    sleeps: list[float] = []
    assert call_with_rate_limit_retry(lambda: 42, sleep=sleeps.append) == 42
    assert sleeps == []


def test_retries_after_rate_limit_then_succeeds():
    sleeps: list[float] = []
    attempts = iter([_rate_limit(), _rate_limit(), "ok"])

    def flaky():
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_rate_limit_retry(flaky, wait_s=21.0, sleep=sleeps.append) == "ok"
    assert sleeps == [21.0, 21.0]  # odczekane pelne okno przed kazda ponowna proba


def test_reraises_after_exhausted_retries():
    sleeps: list[float] = []

    def always_limited():
        raise _rate_limit()

    with pytest.raises(voyage_error.RateLimitError):
        call_with_rate_limit_retry(always_limited, retries=2, sleep=sleeps.append)
    assert len(sleeps) == 2  # retries=2 -> 2 odczekania, potem wyjatek leci dalej


def test_other_errors_are_not_retried():
    sleeps: list[float] = []

    def broken():
        raise voyage_error.AuthenticationError("zly klucz")

    with pytest.raises(voyage_error.AuthenticationError):
        call_with_rate_limit_retry(broken, sleep=sleeps.append)
    assert sleeps == []


def test_connection_error_is_retried():
    # Voyage potrafi zerwac polaczenie bez odpowiedzi (RemoteDisconnected) —
    # blad przejsciowy, ponawiamy jak rate limit.
    sleeps: list[float] = []
    attempts = iter([voyage_error.APIConnectionError("Error communicating with VoyageAI"), "ok"])

    def flaky():
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_rate_limit_retry(flaky, wait_s=21.0, sleep=sleeps.append) == "ok"
    assert sleeps == [21.0]


def test_service_unavailable_is_retried():
    sleeps: list[float] = []
    attempts = iter([voyage_error.ServiceUnavailableError("503"), "ok"])

    def flaky():
        item = next(attempts)
        if isinstance(item, Exception):
            raise item
        return item

    assert call_with_rate_limit_retry(flaky, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 1


class _FlakyVoyageClient:
    """Fake klienta voyageai: pierwsze wywolanie kazdej metody -> RateLimitError, potem wynik."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.rerank_calls = 0

    def embed(self, texts, model, input_type):
        self.embed_calls += 1
        if self.embed_calls == 1:
            raise _rate_limit()

        class _Result:
            embeddings = [[0.1, 0.2]] * len(texts)

        return _Result()

    def rerank(self, query, documents, model, top_k):
        self.rerank_calls += 1
        if self.rerank_calls == 1:
            raise _rate_limit()

        class _Item:
            index = 0
            relevance_score = 0.9

        class _Result:
            results = [_Item()]

        return _Result()


def test_embedder_retries_rate_limit(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr("invoicer.adapters.voyage_retry.time.sleep", recorded.append)
    embedder = VoyageEmbedder(client=_FlakyVoyageClient())
    assert embedder.embed_query("art. 28b") == [0.1, 0.2]
    assert len(recorded) == 1  # jeden rate limit -> jedno odczekanie


def test_reranker_retries_rate_limit(monkeypatch):
    recorded: list[float] = []
    monkeypatch.setattr("invoicer.adapters.voyage_retry.time.sleep", recorded.append)
    reranker = VoyageReranker(client=_FlakyVoyageClient())
    assert reranker.rerank("q", ["d1"], top_k=1) == [(0, 0.9)]
    assert len(recorded) == 1
