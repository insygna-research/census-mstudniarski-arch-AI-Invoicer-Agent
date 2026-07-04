# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projekt

Agentowy asystent ksiegowy (portfolio AI/GenAI): Gmail -> detekcja zalacznikow -> ekstrakcja vision (Gemini) -> walidacja/klasyfikacja PL VAT -> corrective RAG na ustawie o VAT (pgvector + Voyage) -> akceptacja czlowieka przez WhatsApp (LangGraph interrupt) -> ksiegowanie do Fakturowni + hash-chained ledger. Python 3.12, uv, LangGraph/LangChain, FastAPI, deploy na Fly.io. Pelny opis przeplywu: `docs/flow-walkthrough.md`.

## Komendy

```bash
uv sync                                              # srodowisko (wszystko odpalaj przez `uv run`)
uv run pytest -q --cov=src --cov-fail-under=84       # pelny suite (bramka CI: coverage 84%)
uv run pytest tests/unit/test_gemini_extractor.py::test_nazwa -q   # pojedynczy test
uv run ruff check .                                  # lint (line-length 100)
uv run mypy                                          # typy (files=src)

# Testy live (realne API): skipif bez GOOGLE_API_KEY + tests/live/fixtures/sample_invoice.pdf
uv run pytest tests/live -v

# Demo UI (offline na stubach, gdy brak GOOGLE_API_KEY):
uv run --group demo streamlit run src/invoicer/ui/streamlit_app.py

# Skrypty (pytest ma pythonpath=src, goly python NIE — stad PYTHONPATH):
PYTHONPATH=.:src uv run python scripts/run_evals.py

# Evale RAG wymagaja lokalnego pgvector (dane logowania MUSZA zgadzac sie z DATABASE_URL w .env):
docker run -d --name invoicer-pgvector -e POSTGRES_PASSWORD=x -p 5432:5432 pgvector/pgvector:pg16
PYTHONPATH=src uv run python scripts/ingest_legal_corpus.py   # idempotentny ingest korpusu

# Deploy: lokalnie (workflow deploy.yml wymaga sekretu FLY_API_TOKEN — obecnie nieustawiony):
fly deploy
```

## Architektura

**Ports-and-adapters.** `src/invoicer/ports.py` definiuje Protocols (EmailSource, InvoiceDetector, InvoiceExtractor, ExceptionReasoner, AccountingSink, ApprovalChannel, Embedder, LegalKnowledgeStore, Reranker). `src/invoicer/adapters/` ma realny adapter + stub pod kazdy port. Testy jednostkowe NIGDY nie dotykaja sieci: LLM wstrzykiwany przez kwarg `llm=` (fake w testach), realni klienci budowani leniwie w `_client()`.

**Graf LangGraph** (`graph/build.py` + `graph/nodes.py`): `extract -> validate -> classify -> [faktury zagraniczne: retrieve_legal_context -> reason_exception -> verify_grounding] -> human_review -> book`. Human-in-the-loop = LangGraph `interrupt` + checkpointer SQLite; wznowienie przez webhook Twilio (`/whatsapp/inbound` w `app.py`) lub podpisane approval-linki. Faktury PL omijaja RAG; zagraniczne dostaja kontekst prawny, a `verify_grounding` sprawdza, czy cytaty zawieraja sie w zrodle.

**Kompozycja grafu w trzech miejscach:** `app.py` (produkcyjny FastAPI: `preflight_env` fail-fast na brakujace sekrety, APScheduler robi intake co `INTAKE_INTERVAL_MINUTES`, `/status` zwraca metryki LLM i pipeline), `runner.py` (`build_demo_graph`: realne adaptery gdy `GOOGLE_API_KEY`, inaczej stuby offline — z tego korzysta Streamlit), `scripts/` (jednorazowe przebiegi i debug).

**LLM — split modeli i retry.** Trzy adaptery `gemini_*` z publicznym `DEFAULT_MODEL`: extractor i reasoner na `gemini-2.5-flash`, detektor na `gemini-2.5-flash-lite`. Powod: darmowy tier Gemini ma dzienna pule **20 zapytan na model** (reset ~09:00 PL); pule sa per model, a jakosc osadu VAT wymaga flasha (lite mylil wnt z import_towarow). Kazde wywolanie modelu idzie przez `gemini_retry.call_with_rate_limit_retry` (429/500/503 + bledy transportowe httpx — langchain-google-genai sam ich NIE ponawia); analogicznie Voyage (3 RPM) przez `voyage_retry`. Nowe wywolania LLM/embeddingow tez prowadz przez te helpery.

**Structured output.** Schematy Pydantic: `extraction.py` (InvoiceExtraction), `reasoning.py` (ClassificationJudgment), InvoiceCheck w `gemini_detector.py` — przez `with_structured_output`. `extraction_to_invoice` parsuje scisle (Decimal, daty ISO) i glosno odrzuca zly output modelu.

**Metryki.** `LlmMetricsCallback` per adapter — model callbacku MUSI odpowiadac modelowi adaptera (cennik w `observability._PRICING`; nieznany model -> koszt 0.0, a `test_metrics_live` asertuje koszt > 0).

**Evale RAG** (`rag/eval.py` + `scripts/run_evals.py`): golden dataset 30 przypadkow (`data/evals/legal_cases.jsonl`), metryki recall@5/MRR + accuracy with/without RAG + faithfulness cytatow, raport do `docs/evals/legal-rag-report.md`. Eval ma checkpoint/resume (`data/evals/.eval_checkpoint.jsonl`, gitignored) — 60 wywolan nie miesci sie w dziennej kwocie, wiec pelny raport dociaga sie przez kilka dni; po zapisaniu raportu checkpoint znika.

## Konwencje

- **TDD bez wyjatkow**: najpierw failujacy test, potem minimalna implementacja. Testy celowo asertuja stringi modeli, format blokow wiadomosci i forwarding kwargs — zmiana providera/modelu ZACZYNA sie od testow.
- Komentarze, docstringi i komunikaty commitow po polsku, bez znakow diakrytycznych.
- Nowy wymagany sekret = trzy miejsca naraz: `app._REQUIRED_CORE_ENV`, `tests/unit/test_app.py` (_FULL_ENV + test preflight), `.env.example`.
- `docs/superpowers/` to datowane specy i plany — zapis historyczny procesu, nie edytowac wstecz przy zmianach kodu.
- CI (`.github/workflows/ci.yml`) biegnie bez sekretow — testy live zawsze tam skipuja; nie licz na nie przy bramce coverage.

## Pulapki

- **Remote'y gita**: push na `github` (kanoniczny, CI, tam patrza rekruterzy). `origin` to porzucony mirror GitLab (rozjechany o setki commitow) — NIE pushowac bez decyzji wlasciciela.
- **Darmowe tiery**: Gemini 20 zapytan/dzien/model, Voyage 3 RPM / 10K TPM — dluzsze przebiegi (evale, burst intake'u) przezywaja tylko dzieki helperom retry i checkpointowi evali.
- Po `fly deploy` lease maszyny potrafi przez ~3 min blokowac auto-start (proxy loguje "lease currently held") — mija samo, nie restartowac na sile.
- Darmowy tier Gemini trenuje na danych — wysylac tylko faktury testowe; realne dane = platny tier.
