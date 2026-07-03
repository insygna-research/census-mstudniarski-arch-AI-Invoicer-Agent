"""Live eval harness dla legal-grounded RAG.

Wymaga VOYAGE_API_KEY + DATABASE_URL + GOOGLE_API_KEY.

Uruchom: PYTHONPATH=.:src VOYAGE_API_KEY=... DATABASE_URL=... GOOGLE_API_KEY=... \
    uv run python scripts/run_evals.py
Zaklada zindeksowany korpus (scripts/ingest_legal_corpus.py).
Koszt: ~60 wywolan Gemini (30 przypadkow x with/without RAG) + retrieval Voyage.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from invoicer.adapters.gemini_reasoner import GeminiExceptionReasoner
from invoicer.adapters.pgvector_store import PgVectorLegalStore
from invoicer.adapters.voyage_embedder import VoyageEmbedder
from invoicer.adapters.voyage_reranker import VoyageReranker
from invoicer.graph.nodes import EU_COUNTRIES
from invoicer.models import Classification, CountryBucket, Invoice, TaxTreatment
from invoicer.rag.eval import (
    append_checkpoint,
    build_invoice_from_case,
    faithfulness_rate,
    load_cases,
    load_checkpoint,
    mean,
    recall_at_k,
    reciprocal_rank,
    render_report,
)
from invoicer.rag.query import build_retrieval_query

_CASES = Path(__file__).resolve().parents[1] / "data" / "evals" / "legal_cases.jsonl"
_REPORT = Path(__file__).resolve().parents[1] / "docs" / "evals" / "legal-rag-report.md"
# Checkpoint: dzienna kwota darmowego tieru (20 zapytan/model) nie miesci 60 wywolan —
# przerwany bieg dokancza sie kolejnego dnia od miejsca przerwania. Kasowany po raporcie.
_CHECKPOINT = Path(__file__).resolve().parents[1] / "data" / "evals" / ".eval_checkpoint.jsonl"
_SETUP = "pgvector · voyage-3-large · rerank-2.5 · gemini-2.5-flash (reasoner)"


def _store() -> PgVectorLegalStore:
    return PgVectorLegalStore(VoyageEmbedder(), reranker=VoyageReranker())


def _base_classification(invoice: Invoice) -> Classification:
    """Deterministyczny prior — to samo co classify_node dla faktur zagranicznych."""
    bucket = (
        CountryBucket.UE
        if invoice.seller.country.upper() in EU_COUNTRIES
        else CountryBucket.POZA_UE
    )
    return Classification(
        treatment=TaxTreatment.IMPORT_USLUG, country_bucket=bucket, confidence=0.6
    )


def evaluate_retrieval(checkpoint: dict[str, dict], k: int = 5) -> dict:
    if "retrieval" in checkpoint:
        print("[retrieval] wynik z checkpointu")
        return checkpoint["retrieval"]
    store = _store()
    cases = load_cases(_CASES)
    recalls, rrs = [], []
    for case in cases:
        query = build_retrieval_query(build_invoice_from_case(case))
        refs = [c.article_ref for c in store.search(query, k=k)]
        expected = set(case["expected_article_refs"])
        recalls.append(recall_at_k(refs, expected, k=k))
        rrs.append(reciprocal_rank(refs, expected))
    out = {"k": k, "n": len(cases), "recall_at_k": mean(recalls), "mrr": mean(rrs)}
    append_checkpoint(_CHECKPOINT, "retrieval", out)
    return out


def evaluate_reasoning(checkpoint: dict[str, dict], *, with_rag: bool, k: int = 5) -> dict:
    """Accuracy traktowania vs golden; przy with_rag dodatkowo faithfulness cytatow.

    Kazdy przypadek trafia do checkpointu od razu po ocenie — przerwanie (dzienna
    kwota Gemini) nie gubi postepu, kolejny bieg pomija ocenione przypadki.
    """
    label = "with-rag" if with_rag else "no-rag"
    store = _store() if with_rag else None
    reasoner = GeminiExceptionReasoner()
    cases = load_cases(_CASES)
    hits, faiths = [], []
    for case in cases:
        key = f"{label}:{case['id']}"
        cached = checkpoint.get(key)
        if cached is not None:
            hits.append(cached["hit"])
            if with_rag:
                faiths.append(cached["faithfulness"])
            print(f"[{label}] {case['id']}: wynik z checkpointu")
            continue
        invoice = build_invoice_from_case(case)
        context = store.search(build_retrieval_query(invoice), k=k) if store is not None else None
        result = reasoner.reason(invoice, _base_classification(invoice), context=context)
        hit = 1.0 if result.treatment.value == case["expected_treatment"] else 0.0
        hits.append(hit)
        entry: dict = {"treatment": result.treatment.value, "hit": hit}
        if with_rag:
            faith = faithfulness_rate(result.citations, context or [])
            faiths.append(faith)
            entry["faithfulness"] = faith
        append_checkpoint(_CHECKPOINT, key, entry)
        print(
            f"[{label}] {case['id']}: {result.treatment.value} "
            f"(expected {case['expected_treatment']}) {'OK' if hit else 'MISS'}"
        )
    out = {"n": len(cases), "accuracy": mean(hits)}
    if with_rag:
        out["faithfulness"] = mean(faiths)
    return out


def main() -> None:
    checkpoint = load_checkpoint(_CHECKPOINT)
    if checkpoint:
        print(f"Wznawiam z checkpointu: {len(checkpoint)} wpisow ({_CHECKPOINT})")
    retrieval = evaluate_retrieval(checkpoint, k=5)
    print(retrieval)
    with_rag = evaluate_reasoning(checkpoint, with_rag=True)
    print(with_rag)
    without_rag = evaluate_reasoning(checkpoint, with_rag=False)
    print(without_rag)
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(
        render_report(
            generated=date.today().isoformat(),
            retrieval=retrieval,
            with_rag=with_rag,
            without_rag=without_rag,
            setup=_SETUP,
        ),
        encoding="utf-8",
    )
    _CHECKPOINT.unlink(missing_ok=True)  # pelny raport zapisany -> nastepny eval od zera
    print(f"Raport: {_REPORT}")


if __name__ == "__main__":
    main()
