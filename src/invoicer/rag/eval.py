from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from invoicer.graph.nodes import _span_supported
from invoicer.models import Citation, Invoice, LineItem, Party
from invoicer.rag.models import RetrievedChunk


def load_checkpoint(path: Path) -> dict[str, dict]:
    """Wczytuje checkpoint evalu (jsonl klucz->wynik); brak pliku -> pusty dict.

    Dzienna kwota darmowego tieru Gemini (20 zapytan/model) wymusza dokanczanie
    evalu przez wiele dni — checkpoint per przypadek robi postep trwalym.
    Zduplikowany klucz: ostatni wpis wygrywa.
    """
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            out[entry["key"]] = entry["value"]
    return out


def append_checkpoint(path: Path, key: str, value: dict) -> None:
    """Dopisuje jeden wynik do checkpointu (append-only jsonl, flush na wpis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def recall_at_k(retrieved_refs: list[str], expected_refs: set[str], k: int) -> float:
    """Ulamek oczekiwanych article_ref obecnych w top-k zwroconych. Pusty expected -> 1.0."""
    if not expected_refs:
        return 1.0
    top = set(retrieved_refs[:k])
    return len(top & expected_refs) / len(expected_refs)


def reciprocal_rank(retrieved_refs: list[str], expected_refs: set[str]) -> float:
    """1/pozycja pierwszego trafionego oczekiwanego ref (1-indexed); 0.0 gdy brak."""
    for position, ref in enumerate(retrieved_refs, start=1):
        if ref in expected_refs:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_cases(path: Path) -> list[dict]:
    """Wczytuje golden dataset (JSONL) z przypadkami ewaluacyjnymi."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_invoice_from_case(case: dict) -> Invoice:
    """Buduje minimalna Invoice z przypadku eval (do query/klasyfikacji). Kwoty umowne."""
    net = Decimal("1000.00")
    descriptions = case["line_descriptions"]
    per_line = (net / len(descriptions)).quantize(Decimal("0.01"))
    lines = [
        LineItem(
            description=desc,
            quantity=Decimal("1"),
            unit_net=per_line,
            vat_rate=Decimal("0.00") if case.get("no_vat") else Decimal("0.23"),
            net=per_line,
            vat=Decimal("0.00") if case.get("no_vat") else (per_line * Decimal("0.23")),
            gross=per_line if case.get("no_vat") else (per_line * Decimal("1.23")),
        )
        for desc in descriptions
    ]
    total_net = sum((ln.net for ln in lines), Decimal("0"))
    total_vat = sum((ln.vat for ln in lines), Decimal("0"))
    return Invoice(
        seller=Party(name="Eval Seller", country=case["seller_country"]),
        buyer=Party(name="Eval Buyer", nip="5260001246", country="PL"),
        number=f"EVAL/{case['id']}",
        issue_date=date(2026, 1, 1),
        currency=case["currency"],
        lines=lines,
        total_net=total_net,
        total_vat=total_vat,
        total_gross=total_net + total_vat,
    )


def faithfulness_rate(citations: list[Citation], context: list[RetrievedChunk]) -> float:
    """Ulamek cytatow, ktorych quoted_span jest faktycznie zawarty w cytowanym zrodle."""
    if not citations:
        return 0.0
    by_ref = {(c.source_id, c.article_ref): c.text for c in context}
    supported = sum(
        1
        for cit in citations
        if _span_supported(cit.quoted_span, by_ref.get((cit.source_id, cit.article_ref), ""))
    )
    return supported / len(citations)


def render_report(
    *,
    generated: str,
    retrieval: dict,
    with_rag: dict,
    without_rag: dict,
    setup: str,
) -> str:
    """Renderuje raport eval (markdown). Czysta funkcja — testowalna bez kluczy API."""
    delta = with_rag["accuracy"] - without_rag["accuracy"]
    return (
        "# Legal-Grounded RAG — Eval Report\n\n"
        f"- Generated: {generated}\n"
        f"- Cases: **{retrieval['n']}** (golden dataset, `data/evals/legal_cases.jsonl`)\n"
        f"- Setup: {setup}\n\n"
        "## Retrieval\n\n"
        f"- Recall@{retrieval['k']}: **{retrieval['recall_at_k']:.2f}**\n"
        f"- MRR: **{retrieval['mrr']:.2f}**\n\n"
        "## Reasoning (grounded classification)\n\n"
        f"- Treatment accuracy **with RAG**: **{with_rag['accuracy']:.2f}**\n"
        f"- Treatment accuracy **without RAG**: **{without_rag['accuracy']:.2f}**\n"
        f"- Ablation delta (with − without): **{delta:+.2f}**\n"
        f"- Faithfulness (cytaty potwierdzone w zrodle): **{with_rag['faithfulness']:.2f}**\n\n"
        "_Wygenerowane przez `scripts/run_evals.py` "
        "(retrieval: Voyage + pgvector; reasoner wg setupu powyzej)._\n"
    )
