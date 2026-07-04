# Legal-Grounded RAG — Eval Report

- Generated: 2026-07-04
- Cases: **30** (golden dataset, `data/evals/legal_cases.jsonl`)
- Setup: pgvector · voyage-3-large · rerank-2.5 · qwen3-vl:8b-instruct (ollama reasoner)

## Retrieval

- Recall@5: **0.93**
- MRR: **0.61**

## Reasoning (grounded classification)

- Treatment accuracy **with RAG**: **0.80**
- Treatment accuracy **without RAG**: **0.80**
- Ablation delta (with − without): **+0.00**
- Faithfulness (cytaty potwierdzone w zrodle): **0.29**

_Wygenerowane przez `scripts/run_evals.py` (retrieval: Voyage + pgvector; reasoner wg setupu powyzej)._
