# Legal-Grounded RAG — Eval Report

- Generated: 2026-07-02
- Cases: **30** (golden dataset, `data/evals/legal_cases.jsonl`)
- Retrieval store: pgvector · voyage-3-large · rerank-2.5

## Retrieval

- Recall@5: **0.93**
- MRR: **0.61**

## Reasoning (grounded classification) — pending

Treatment accuracy **with vs without RAG** (ablation) and citation **faithfulness** are
implemented in `scripts/run_evals.py` (see `evaluate_reasoning`) and unit-tested via
`render_report`, but require LLM API access to run against the 30 cases. Re-run
`PYTHONPATH=.:src uv run python scripts/run_evals.py` with `GOOGLE_API_KEY` set
(Gemini free tier suffices; ~60 reasoner calls) to populate accuracy / faithfulness /
ablation-delta here.

_Retrieval wygenerowane przez paced runner (Voyage + pgvector); N=30 golden dataset._
