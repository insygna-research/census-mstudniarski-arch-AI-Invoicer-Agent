# Portfolio Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Doprowadzić repo Invoicer do stanu "portfolio-ready" na rolę GenAI Engineer: evals dowożące to, co obiecuje README (30 przypadków, faithfulness, ablacja z/bez RAG), metadane repo na GitHubie, LICENSE, mypy + coverage w CI, sekcja "How it was built" i wizuale.

**Architecture:** Zero zmian w logice produkcyjnej. Rozszerzamy golden dataset i harness ewaluacyjny (czysta funkcja `render_report` w `src/invoicer/rag/eval.py`, testowana unitowo; `scripts/run_evals.py` pozostaje cienkim driverem live). Reszta to konfiguracja (pyproject, CI), dokumentacja (README, LICENSE) i metadane repo (gh CLI).

**Tech Stack:** Python 3.12 · uv · pytest · ruff · mypy · pytest-cov · gh CLI · GitHub Actions.

**Konwencje repo (obowiązują we wszystkich taskach):**
- Komentarze/docstringi po polsku **bez polskich znaków** (ASCII), jak w istniejącym kodzie.
- Commity w stylu conventional commits (`feat:`, `fix:`, `docs:`, `chore:`), opis może być po polsku.
- Testy: `uv run pytest -q` (pythonpath=src ustawione w pyproject). Lint: `uv run ruff check .` i `uv run ruff format --check .`.
- Przed KAŻDYM commitem: `uv run ruff format . && uv run ruff check .` musi przechodzić.

**Zależności między taskami:** Task 3 → 4 → 5 → 6 (evals, po kolei). Taski 1, 2, 7, 8, 9 są niezależne. Task 10 wymaga plików graficznych od użytkownika. Task 6 wymaga kluczy API (VOYAGE, ANTHROPIC, DATABASE_URL) — jeśli niedostępne, wykonaj wariant B z tego taska.

---

### Task 1: LICENSE (MIT)

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Utwórz plik LICENSE**

```text
MIT License

Copyright (c) 2026 Michał Studniarski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Commit**

```bash
git add LICENSE
git commit -m "chore: add MIT license"
```

---

### Task 2: Metadane repo na GitHubie

**Files:** brak (operacje przez `gh` CLI)

- [ ] **Step 1: Ustaw opis i topics**

```bash
gh repo edit mstudniarski-arch/invoicer \
  --description "Agentic AI invoice assistant: Gmail intake → Claude extraction → legal-grounded RAG (Polish VAT, pgvector + Voyage) → human approval over WhatsApp → accounting. LangGraph, TDD, live 24/7 on Fly.io." \
  --add-topic langgraph \
  --add-topic rag \
  --add-topic claude \
  --add-topic genai \
  --add-topic python \
  --add-topic human-in-the-loop \
  --add-topic pgvector \
  --add-topic ai-agents \
  --add-topic llm \
  --add-topic langchain
```

- [ ] **Step 2: Zweryfikuj**

Run: `gh repo view mstudniarski-arch/invoicer --json description,repositoryTopics`
Expected: opis niepusty, 10 topics.

- [ ] **Step 3 (użytkownik, ręcznie):** Przypnij repo na profilu GitHub (github.com → profil → "Customize your pins"). Tego nie da się zrobić przez `gh` CLI.

---

### Task 3: Golden dataset 6 → 30 przypadków

**Files:**
- Modify: `data/evals/legal_cases.jsonl`
- Test: `tests/unit/test_legal_cases_data.py`

- [ ] **Step 1: Zaostrz test datasetu (failing test)**

W `tests/unit/test_legal_cases_data.py` zamień funkcję `test_dataset_loads_and_has_expected_fields` oraz `test_dataset_covers_key_treatments` na:

```python
def test_dataset_loads_and_has_expected_fields():
    cases = load_cases(_CASES)
    assert len(cases) >= 30
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))  # id unikalne
    for case in cases:
        assert case["expected_treatment"] in {t.value for t in TaxTreatment}
        assert isinstance(case["expected_article_refs"], list)
        assert case["expected_article_refs"]  # kazdy przypadek ma oczekiwana podstawe prawna
        assert case["seller_country"]
        assert case["seller_country"] != "PL"  # eval dotyczy wylacznie faktur zagranicznych


def test_dataset_covers_key_treatments():
    cases = load_cases(_CASES)
    by_treatment: dict[str, int] = {}
    for c in cases:
        by_treatment[c["expected_treatment"]] = by_treatment.get(c["expected_treatment"], 0) + 1
    # rdzen kontrastu podatkowego: uslugi vs towary UE vs towary spoza UE
    assert by_treatment.get("import_uslug", 0) >= 15
    assert by_treatment.get("wnt", 0) >= 5
    assert by_treatment.get("import_towarow", 0) >= 5
```

- [ ] **Step 2: Uruchom test — ma failować**

Run: `uv run pytest tests/unit/test_legal_cases_data.py -v`
Expected: FAIL na `assert len(cases) >= 30` (jest 6).

- [ ] **Step 3: Dopisz 24 przypadki do `data/evals/legal_cases.jsonl`**

Dopisz poniższe linie na końcu pliku (zachowaj istniejące 6). Kraje UE muszą pochodzić z `EU_COUNTRIES` w `src/invoicer/graph/nodes.py:95` (uwaga: GB i CH NIE są w UE).

```jsonl
{"id": "nl-adtech", "seller_country": "NL", "currency": "EUR", "no_vat": true, "line_descriptions": ["Kampania reklamowa online"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "es-design", "seller_country": "ES", "currency": "EUR", "no_vat": true, "line_descriptions": ["Projekt graficzny identyfikacji wizualnej"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "it-legal", "seller_country": "IT", "currency": "EUR", "no_vat": true, "line_descriptions": ["Obsluga prawna kontraktu handlowego"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "se-devops", "seller_country": "SE", "currency": "SEK", "no_vat": true, "line_descriptions": ["Utrzymanie infrastruktury DevOps"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "cz-translations", "seller_country": "CZ", "currency": "CZK", "no_vat": true, "line_descriptions": ["Tlumaczenie dokumentacji technicznej"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "at-training", "seller_country": "AT", "currency": "EUR", "no_vat": true, "line_descriptions": ["Szkolenie zespolu developerskiego"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "be-audit", "seller_country": "BE", "currency": "EUR", "no_vat": true, "line_descriptions": ["Audyt bezpieczenstwa aplikacji webowej"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1", "art. 17 ust. 1 pkt 4"]}
{"id": "dk-saas", "seller_country": "DK", "currency": "DKK", "no_vat": true, "line_descriptions": ["Subskrypcja narzedzia analitycznego SaaS"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1", "art. 17 ust. 1 pkt 4"]}
{"id": "sk-support", "seller_country": "SK", "currency": "EUR", "no_vat": true, "line_descriptions": ["Wsparcie techniczne aplikacji"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "nl-goods", "seller_country": "NL", "currency": "EUR", "no_vat": true, "line_descriptions": ["Dostawa serwerow rack - 4 szt"], "expected_treatment": "wnt", "expected_article_refs": ["art. 9 ust. 1"]}
{"id": "cz-goods", "seller_country": "CZ", "currency": "CZK", "no_vat": true, "line_descriptions": ["Dostawa obudow aluminiowych"], "expected_treatment": "wnt", "expected_article_refs": ["art. 9 ust. 1"]}
{"id": "it-goods", "seller_country": "IT", "currency": "EUR", "no_vat": true, "line_descriptions": ["Partia mebli biurowych"], "expected_treatment": "wnt", "expected_article_refs": ["art. 9 ust. 1"]}
{"id": "fr-goods", "seller_country": "FR", "currency": "EUR", "no_vat": true, "line_descriptions": ["Dostawa komponentow optycznych"], "expected_treatment": "wnt", "expected_article_refs": ["art. 9 ust. 1"]}
{"id": "ro-goods", "seller_country": "RO", "currency": "EUR", "no_vat": true, "line_descriptions": ["Dostawa wiazek kablowych"], "expected_treatment": "wnt", "expected_article_refs": ["art. 9 ust. 1"]}
{"id": "ch-software", "seller_country": "CH", "currency": "CHF", "no_vat": true, "line_descriptions": ["Licencja oprogramowania ERP"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1", "art. 17 ust. 1 pkt 4"]}
{"id": "in-qa", "seller_country": "IN", "currency": "USD", "no_vat": true, "line_descriptions": ["Outsourcing testow QA"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "ua-dev", "seller_country": "UA", "currency": "USD", "no_vat": true, "line_descriptions": ["Uslugi programistyczne - sprint 12"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "jp-api", "seller_country": "JP", "currency": "JPY", "no_vat": true, "line_descriptions": ["Dostep do API - plan enterprise"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "no-consulting", "seller_country": "NO", "currency": "NOK", "no_vat": true, "line_descriptions": ["Konsulting inzynieryjny"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "il-pentest", "seller_country": "IL", "currency": "USD", "no_vat": true, "line_descriptions": ["Test penetracyjny aplikacji"], "expected_treatment": "import_uslug", "expected_article_refs": ["art. 28b ust. 1"]}
{"id": "us-laptops", "seller_country": "US", "currency": "USD", "no_vat": true, "line_descriptions": ["Zakup laptopow - 10 szt (import)"], "expected_treatment": "import_towarow", "expected_article_refs": ["art. 2 pkt 7"]}
{"id": "jp-robotics", "seller_country": "JP", "currency": "JPY", "no_vat": true, "line_descriptions": ["Import podzespolow robotycznych"], "expected_treatment": "import_towarow", "expected_article_refs": ["art. 2 pkt 7"]}
{"id": "tr-textiles", "seller_country": "TR", "currency": "USD", "no_vat": true, "line_descriptions": ["Import tkanin - 50 rolek"], "expected_treatment": "import_towarow", "expected_article_refs": ["art. 2 pkt 7"]}
{"id": "kr-displays", "seller_country": "KR", "currency": "USD", "no_vat": true, "line_descriptions": ["Import paneli LCD - partia"], "expected_treatment": "import_towarow", "expected_article_refs": ["art. 2 pkt 7"]}
```

Rozkład po zmianie: `import_uslug` 19 (4 stare + 15 nowych), `wnt` 6 (1+5), `import_towarow` 5 (1+4) = 30 przypadków.

- [ ] **Step 4: Uruchom testy — mają przechodzić**

Run: `uv run pytest tests/unit/test_legal_cases_data.py -v`
Expected: PASS (3 testy).

- [ ] **Step 5: Pełny suite + commit**

Run: `uv run pytest -q` — Expected: 351+ passed, 10 skipped.

```bash
git add data/evals/legal_cases.jsonl tests/unit/test_legal_cases_data.py
git commit -m "feat(evals): rozszerz golden dataset 6->30 przypadkow (uslugi/WNT/import, UE i poza UE)"
```

---

### Task 4: `render_report` — czysty renderer raportu ewaluacji

**Files:**
- Modify: `src/invoicer/rag/eval.py`
- Test: `tests/unit/test_rag_eval.py`

- [ ] **Step 1: Failing test**

Dopisz na końcu `tests/unit/test_rag_eval.py`:

```python
def test_render_report_includes_all_metrics_and_ablation_delta():
    from invoicer.rag.eval import render_report

    report = render_report(
        generated="2026-07-02",
        retrieval={"k": 5, "n": 30, "recall_at_k": 0.97, "mrr": 0.81},
        with_rag={"n": 30, "accuracy": 0.93, "faithfulness": 0.9},
        without_rag={"n": 30, "accuracy": 0.7},
        setup="pgvector · voyage-3-large · rerank-2.5 · claude-sonnet-4-6",
    )
    assert "Recall@5: **0.97**" in report
    assert "MRR: **0.81**" in report
    assert "with RAG**: **0.93**" in report
    assert "without RAG**: **0.70**" in report
    assert "**+0.23**" in report  # delta ablacji
    assert "Faithfulness" in report and "**0.90**" in report
    assert "2026-07-02" in report
    assert "Cases: **30**" in report
```

- [ ] **Step 2: Run — ma failować**

Run: `uv run pytest tests/unit/test_rag_eval.py::test_render_report_includes_all_metrics_and_ablation_delta -v`
Expected: FAIL — `ImportError: cannot import name 'render_report'`.

- [ ] **Step 3: Implementacja**

Dopisz na końcu `src/invoicer/rag/eval.py`:

```python
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
        "_Wygenerowane przez `scripts/run_evals.py` (Voyage + pgvector + Claude)._\n"
    )
```

- [ ] **Step 4: Run — ma przechodzić**

Run: `uv run pytest tests/unit/test_rag_eval.py -v`
Expected: PASS (wszystkie).

- [ ] **Step 5: Commit**

```bash
git add src/invoicer/rag/eval.py tests/unit/test_rag_eval.py
git commit -m "feat(evals): render_report — raport z retrieval + accuracy + faithfulness + ablacja"
```

---

### Task 5: Eval reasoningu (accuracy, faithfulness, ablacja z/bez RAG) w `run_evals.py`

**Files:**
- Modify: `scripts/run_evals.py`

Skrypt jest live-driverem (wymaga kluczy) i nie ma testów unitowych — cała testowalna logika (metryki, renderer) siedzi w `invoicer.rag.eval` (Task 4). Tu tylko wiring.

- [ ] **Step 1: Zastąp zawartość `scripts/run_evals.py`**

```python
"""Live eval harness dla legal-grounded RAG. Wymaga VOYAGE_API_KEY + DATABASE_URL + ANTHROPIC_API_KEY.

Uruchom: PYTHONPATH=.:src VOYAGE_API_KEY=... DATABASE_URL=... ANTHROPIC_API_KEY=... \
    uv run python scripts/run_evals.py
Zaklada zindeksowany korpus (scripts/ingest_legal_corpus.py).
Koszt: ~60 wywolan Claude (30 przypadkow x with/without RAG) + retrieval Voyage.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from invoicer.adapters.claude_reasoner import ClaudeExceptionReasoner
from invoicer.adapters.pgvector_store import PgVectorLegalStore
from invoicer.adapters.voyage_embedder import VoyageEmbedder
from invoicer.adapters.voyage_reranker import VoyageReranker
from invoicer.graph.nodes import EU_COUNTRIES
from invoicer.models import Classification, CountryBucket, Invoice, TaxTreatment
from invoicer.rag.eval import (
    build_invoice_from_case,
    faithfulness_rate,
    load_cases,
    mean,
    recall_at_k,
    reciprocal_rank,
    render_report,
)
from invoicer.rag.query import build_retrieval_query

_CASES = Path(__file__).resolve().parents[1] / "data" / "evals" / "legal_cases.jsonl"
_REPORT = Path(__file__).resolve().parents[1] / "docs" / "evals" / "legal-rag-report.md"
_SETUP = "pgvector · voyage-3-large · rerank-2.5 · claude-sonnet-4-6"


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


def evaluate_retrieval(k: int = 5) -> dict:
    store = _store()
    cases = load_cases(_CASES)
    recalls, rrs = [], []
    for case in cases:
        query = build_retrieval_query(build_invoice_from_case(case))
        refs = [c.article_ref for c in store.search(query, k=k)]
        expected = set(case["expected_article_refs"])
        recalls.append(recall_at_k(refs, expected, k=k))
        rrs.append(reciprocal_rank(refs, expected))
    return {"k": k, "n": len(cases), "recall_at_k": mean(recalls), "mrr": mean(rrs)}


def evaluate_reasoning(*, with_rag: bool, k: int = 5) -> dict:
    """Accuracy traktowania vs golden; przy with_rag dodatkowo faithfulness cytatow."""
    store = _store() if with_rag else None
    reasoner = ClaudeExceptionReasoner()
    cases = load_cases(_CASES)
    hits, faiths = [], []
    for case in cases:
        invoice = build_invoice_from_case(case)
        context = (
            store.search(build_retrieval_query(invoice), k=k) if store is not None else None
        )
        result = reasoner.reason(invoice, _base_classification(invoice), context=context)
        hit = 1.0 if result.treatment.value == case["expected_treatment"] else 0.0
        hits.append(hit)
        if with_rag:
            faiths.append(faithfulness_rate(result.citations, context or []))
        label = "with-rag" if with_rag else "no-rag"
        print(f"[{label}] {case['id']}: {result.treatment.value} "
              f"(expected {case['expected_treatment']}) {'OK' if hit else 'MISS'}")
    out = {"n": len(cases), "accuracy": mean(hits)}
    if with_rag:
        out["faithfulness"] = mean(faiths)
    return out


def main() -> None:
    retrieval = evaluate_retrieval(k=5)
    print(retrieval)
    with_rag = evaluate_reasoning(with_rag=True)
    print(with_rag)
    without_rag = evaluate_reasoning(with_rag=False)
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
    print(f"Raport: {_REPORT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity check bez kluczy (import + lint)**

Run: `PYTHONPATH=src uv run python -c "import scripts.run_evals" 2>/dev/null || PYTHONPATH=.:src uv run python -c "import scripts.run_evals; print('ok')"`
Expected: `ok` (import nie wymaga kluczy — klienci tworzeni leniwie).
Run: `uv run ruff check . && uv run ruff format --check .` — Expected: bez błędów.
Run: `uv run pytest -q` — Expected: wszystko zielone.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_evals.py
git commit -m "feat(evals): accuracy + faithfulness + ablacja z/bez RAG w harnessie live"
```

---

### Task 6: Uruchom live evals i domknij obietnicę z README

**Files:**
- Modify: `docs/evals/legal-rag-report.md` (generowany)
- Modify: `README.md` (linia ~46 i sekcja Testing, jeśli zmienią się liczby)

**Wymaga:** `VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, `DATABASE_URL` (lokalny pgvector wystarczy). Koszt: ~60 wywołań claude-sonnet (tekstowe, krótkie) — pojedyncze centy.

- [ ] **Step 1 (wariant A — klucze dostępne): Postaw pgvector i zaingestuj korpus**

```bash
docker run -d --name invoicer-eval-pg -e POSTGRES_PASSWORD=x -p 5432:5432 pgvector/pgvector:pg16
export DATABASE_URL=postgresql://postgres:x@localhost:5432/postgres
set -a; source .env; set +a   # VOYAGE_API_KEY, ANTHROPIC_API_KEY z .env
PYTHONPATH=src uv run python scripts/ingest_legal_corpus.py
```

Expected: ingest kończy się bez błędu (idempotentny).

- [ ] **Step 2: Uruchom evals**

```bash
PYTHONPATH=.:src uv run python scripts/run_evals.py
```

Expected: wydruk per-case `OK`/`MISS`, na końcu `Raport: .../docs/evals/legal-rag-report.md`. Sensowne wyniki: recall@5 ≥ 0.9, accuracy with RAG > accuracy without RAG. Jeśli accuracy with RAG < without — zatrzymaj się i przeanalizuj MISS-y per case (możliwy błąd w golden labels, np. dyskusyjny przypadek towar/usługa) zanim cokolwiek zacommitujesz; popraw dataset, nie metryki.

- [ ] **Step 3: Zaktualizuj README**

W `README.md` w sekcji "Legal-grounded corrective RAG" (linia ~46) zdanie kończące się na `are measured in [docs/evals/legal-rag-report.md]...` pozostaje prawdziwe — podmień tylko, jeśli chcesz, wtrącenie z liczbami. Dodatkowo w sekcji `## Testing` dopisz punkt (po pierwszym buletcie):

```markdown
- **Evals** — retrieval (recall@5 / MRR), grounded-classification accuracy **with vs without RAG** (ablation), and citation **faithfulness** over a 30-case golden dataset: [`docs/evals/legal-rag-report.md`](docs/evals/legal-rag-report.md).
```

- [ ] **Step 4: Commit + sprzątanie**

```bash
git add docs/evals/legal-rag-report.md README.md
git commit -m "chore(evals): pelny raport — retrieval + accuracy z/bez RAG + faithfulness (N=30)"
docker rm -f invoicer-eval-pg   # jesli stawiales kontener w Step 1
```

- [ ] **Step 1-alt (wariant B — brak kluczy):** Nie generuj raportu. Zamiast tego usuń rozjazd w README: w linii ~46 zamień `Retrieval quality, faithfulness, and with/without-RAG treatment accuracy are measured in` na `Retrieval quality is measured in` oraz w raporcie zostaw dopisek "do uzupelnienia". Commit: `docs(readme): zlagodz deklaracje eval do faktycznie zmierzonych metryk`. Wariant A wykonaj, gdy klucze będą pod ręką — README wróci wtedy do pełnej wersji (Step 3).

---

### Task 7: mypy w projekcie i w CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/invoicer/graph/nodes.py`, `src/invoicer/runner.py`, `src/invoicer/webhook.py`, `src/invoicer/ui/streamlit_app.py` (+ ewentualnie inne wskazane przez mypy — stan na 2026-07-02: 50 błędów w 5 plikach)
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Dodaj mypy do dev deps i konfigurację**

```bash
uv add --dev "mypy>=1.14"
```

W `pyproject.toml` dopisz na końcu:

```toml
[tool.mypy]
python_version = "3.12"
files = ["src"]
check_untyped_defs = true
warn_unused_ignores = true
ignore_missing_imports = true
```

(`ignore_missing_imports` globalnie — google-api-python-client, apscheduler i voyageai nie mają stubów; per-module overrides to zbędna biurokracja na tę skalę.)

- [ ] **Step 2: Baseline**

Run: `uv run mypy`
Expected: ~50 błędów w 5 plikach. Zanotuj pełną listę.

- [ ] **Step 3: Napraw błędy — plik po pliku, po każdym pliku pełny pytest**

Zasady: **zachowaj zachowanie runtime** (testy strzegą), preferuj zawężenie typu nad `# type: ignore`; `# type: ignore[kod]` tylko tam, gdzie zawężenie wymagałoby przebudowy. Wzorce dla faktycznych kategorii błędów:

**(a) `union-attr` w `graph/nodes.py` (np. linie 309–333, `Classification | None`)** — na początku funkcji węzła zwiąż lokalną zmienną i zawęź assertem (router gwarantuje obecność — to inwariant grafu, nie logika):

```python
classification = state.get("classification")
assert classification is not None  # router_after_classify gwarantuje classification
# dalej uzywaj `classification` zamiast state["classification"]
```

**(b) `assignment` w `runner.py:182-183` (stub podmieniany za realny adapter)** — anotuj zmienną typem portu (Protocol), nie konkretnego adaptera:

```python
extractor: InvoiceExtractor = ClaudeVisionExtractor(...)
if offline:
    extractor = StubExtractor(...)   # oba spelniaja Protocol
```

(import portu z `invoicer.ports`; analogicznie `reasoner: ExceptionReasoner`).

**(c) `arg-type` w `webhook.py:75,107` (`str | None` do funkcji oczekującej `str`)** — jawny guard przed wywołaniem, zwracający dokładnie to, co dziś zwraca ścieżka odmowy (sprawdź istniejące zachowanie w `tests/unit/test_webhook.py` zanim wybierzesz kod odpowiedzi):

```python
if signature is None or public_url is None:
    return _reject_response()  # uzyj istniejacej sciezki odmowy z tego handlera
```

**(d) `union-attr` w `ui/streamlit_app.py:218` (`UploadedFile | None`)** — kod już działa tylko po uploadzie; owiń użycie w `if uploaded is not None:`.

Po każdym naprawionym pliku: `uv run pytest -q` — Expected: 0 failów.

- [ ] **Step 4: mypy zielone**

Run: `uv run mypy`
Expected: `Success: no issues found in 54 source files`.

- [ ] **Step 5: Krok w CI**

W `.github/workflows/ci.yml`, po kroku "Format check (ruff)" dodaj:

```yaml
      - name: Type check (mypy)
        run: uv run mypy
```

- [ ] **Step 6: Pełna weryfikacja + commit**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy`
Expected: wszystko zielone.

```bash
git add pyproject.toml uv.lock src .github/workflows/ci.yml
git commit -m "chore(types): mypy w CI + naprawa 50 bledow typowania (Optional-narrowing, typy portow)"
```

---

### Task 8: Coverage (pytest-cov) w CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md` (sekcja Testing — jedna linijka z coverage)

- [ ] **Step 1: Dodaj pytest-cov**

```bash
uv add --dev "pytest-cov>=6.0"
```

- [ ] **Step 2: Zmierz aktualne pokrycie**

Run: `uv run pytest -q --cov=invoicer --cov-report=term | tail -20`
Zanotuj `TOTAL ...%`. Ustal próg: `floor(TOTAL) - 2` (bufor na drobne zmiany), np. przy 91% → próg 89.

- [ ] **Step 3: Krok w CI**

W `.github/workflows/ci.yml` zamień krok "Tests" na (wstaw zmierzony próg zamiast `<PRÓG>` — to jedyne miejsce w tym planie uzupełniane pomiarem ze Step 2):

```yaml
      - name: Tests (with coverage)
        run: uv run pytest -q --cov=invoicer --cov-report=term --cov-fail-under=<PRÓG>
```

- [ ] **Step 4: README — jedna linijka**

W sekcji `## Testing`, w bloku poleceń, zamień `uv run pytest -q          # 351 passed, 10 skipped (live)` na:

```bash
uv run pytest -q --cov=invoicer   # 351+ passed, 10 skipped (live); coverage gated in CI
```

- [ ] **Step 5: Weryfikacja + commit**

Run: `uv run pytest -q --cov=invoicer --cov-fail-under=<PRÓG>` — Expected: PASS, próg spełniony.

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml README.md
git commit -m "chore(ci): pomiar coverage z progiem fail-under"
```

---

### Task 9: Sekcja "How it was built" w README (AI-assisted, jawnie)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Wstaw sekcję**

W `README.md`, bezpośrednio **po** sekcji `## Tech stack` (przed `## Run it`), wstaw:

```markdown
## How it was built

Spec-driven, AI-assisted TDD. Every feature started as a written design spec, was decomposed
into a step-by-step plan, and implemented red→green with Claude Code as the pair programmer.
The full trail — 10 specs and 17 plans — is committed in
[`docs/superpowers/`](docs/superpowers/), so you can audit the path from "what should exist"
to the code that exists. The architecture decisions (ports-and-adapters, the hard human gate,
grounded-RAG-with-abstention) and every green test are the point; the keystrokes were shared.
```

Sprawdź liczby przed commitem: `ls docs/superpowers/specs | wc -l` i `ls docs/superpowers/plans | wc -l` — po dodaniu planu z tego zadania liczba planów wzrośnie; wpisz faktyczne wartości.

- [ ] **Step 2: Commit**

```bash
git add README.md docs/superpowers/plans/2026-07-02-portfolio-polish.md
git commit -m "docs(readme): sekcja 'How it was built' — jawny spec-driven AI-assisted TDD"
```

---

### Task 10: Wizuale w README (wymaga plików od użytkownika)

**Files:**
- Create: `docs/assets/whatsapp-approval.png`, `docs/assets/streamlit-demo.png`, `docs/assets/langsmith-trace.png` (dostarcza użytkownik)
- Modify: `README.md`

- [ ] **Step 1 (użytkownik): Zrób i wrzuć zrzuty do `docs/assets/`**

1. **WhatsApp** (`whatsapp-approval.png`) — screenshot z telefonu wiadomości approval z linkami TAK/NIE. **Zamaż numery telefonów, NIP-y i kwoty prawdziwych faktur** (albo użyj przebiegu na fakturze testowej).
2. **Streamlit** (`streamlit-demo.png`) — `uv run --group demo streamlit run src/invoicer/ui/streamlit_app.py`, przeprowadź fakturę do bramki human-review, screenshot całego widoku pipeline.
3. **LangSmith** (`langsmith-trace.png`) — trace jednego przebiegu z widocznymi spanami extraction → retrieval → reasoning (smith.langchain.com). Zamaż nazwy/ID organizacji, jeśli widoczne.

Szerokość ≥1200px, PNG. Wykonawca: **nie kontynuuj bez plików** — sprawdź `ls docs/assets/` i jeśli brakuje, zostaw ten task otwarty i zgłoś to w podsumowaniu.

- [ ] **Step 2: Wstaw obrazy do README**

W `README.md`, w sekcji `## What it does`, bezpośrednio po bloku mermaid (przed listą numerowaną), wstaw:

```markdown
<p align="center">
  <img src="docs/assets/streamlit-demo.png" alt="Streamlit pipeline view at the human-review gate" width="70%">
</p>
<p align="center">
  <img src="docs/assets/whatsapp-approval.png" alt="WhatsApp approval with signed tap-to-approve links" width="35%">
  <img src="docs/assets/langsmith-trace.png" alt="LangSmith trace: extraction → retrieval → reasoning" width="55%">
</p>
```

- [ ] **Step 3: Weryfikacja renderu + commit**

Sprawdź, że ścieżki istnieją: `ls docs/assets/*.png` → 3 pliki.

```bash
git add docs/assets README.md
git commit -m "docs(readme): zrzuty — Streamlit gate, WhatsApp approval, LangSmith trace"
```

---

### Task 11: Push na oba remote'y

- [ ] **Step 1: Finalna weryfikacja**

Run: `uv run pytest -q && uv run ruff check . && uv run mypy`
Expected: wszystko zielone.

- [ ] **Step 2: Push**

Repo ma dwa remote'y: `origin` (GitLab) i `github` (GitHub — tam patrzą rekruterzy i tam działa CI + auto-deploy Fly).

```bash
git push origin main
git push github main
```

**Uwaga:** push na `github/main` uruchamia `.github/workflows/fly-deploy.yml` (deploy produkcyjny). Zmiany w tym planie nie dotykają logiki runtime (poza type-narrowing z Taska 7, pokrytym testami), więc deploy jest bezpieczny — ale obserwuj `gh run watch` po pushu i `curl -s https://<app>.fly.dev/health`.

---

## Poza zakresem (świadomie)

- Tłumaczenie polskich docstringów/komentarzy na angielski — duży churn, dyskusyjna wartość na rynek PL; osobna decyzja.
- Badge coverage (Codecov) — próg `--cov-fail-under` w CI daje ten sam sygnał bez zewnętrznego serwisu; YAGNI.
- Pozycje z Roadmapy README (compliance gate, rekonsyliacja płatności) — to feature'y, nie polish.
