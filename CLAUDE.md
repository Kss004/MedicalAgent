# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Healthcare Information Assistant — a RAG (Retrieval-Augmented Generation) POC that delivers verified, educational health information. The project has two main components:
1. **Core chat assistant** (`medical_assistant.py` + `server.py`) — domain-whitelisted Tavily search + LLM-based source verification + strict guardrails
2. **PCOS POC** (`pocs/womens_health_pcos/`) — offline knowledge base with its own ingestion pipeline, SQLAlchemy DB, assessment scoring, and a background scheduler

## Commands

### Backend
```bash
# Install Python deps (use venv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run backend server (FastAPI + Uvicorn, port 8000, auto-reload)
python server.py

# Run CLI-only mode (no web server)
python medical_assistant.py

# Run PCOS full ingestion pipeline (discovery → scrape → chunk → refresh)
python pocs/womens_health_pcos/pipeline/source_ingestion_pipeline.py

# Migrate existing PCOS data
python migrate_pcos_data.py
```

### Tests
```bash
# Run from project root (tests/ uses sys.path.insert to find project root)
python tests/test_pcos_pipeline.py       # DB init + ingestion + evaluate_pcos
python tests/test_ingestion_schema.py    # Source/Page/Content schema + versioning
python tests/test_scheduler.py           # Scheduler job tests
python tests/test_memory.py              # Memory tests
```

### Frontend
```bash
cd frontend
npm install
npm run dev      # Vite dev server (calls localhost:8000 directly, no proxy)
npm run build    # Output to frontend/dist/ (served by FastAPI in production)
npm run lint
```

## Architecture

### Backend (Python)

- **`server.py`** — FastAPI app. Mounts the PCOS questionnaire router, starts the APScheduler on startup, and serves built frontend as static files. Endpoints: `POST /api/chat`, `POST /api/upload_prescription`, `GET /api/health`, `GET /api/scheduler/jobs`.

- **`medical_assistant.py`** — Core RAG pipeline with three functions:
  - `ask_health_assistant(query, history)` — main chat pipeline: Tavily (domain-whitelisted to 11 trusted medical sites) → YouTube search → Shorts/Reels Tavily → per-source LLM verification (score 1–100) → response generation with inline `[Source N]` citations. Returns `{"response": str, "sources": [...]}`
  - `analyze_prescription(base64_images)` — OCR pipeline using GPT-4o vision to extract text from prescription images
  - `evaluate_pcos(data_dict)` — runs the PCOS scoring model against the local SQLite KB; called by the questionnaire API

  Key singletons: `tavily_search`, `shorts_tavily`, `youtube_search`, `llm` (all module-level).

- **`pocs/womens_health_pcos/`** — Self-contained PCOS knowledge system:
  - `database/` — SQLAlchemy models (`Source`, `SourcePage`, `PageContent`, `ChunkMetadata`, `ResearchSource`, `SymptomPattern`, `LabMarker`, `TreatmentProtocol`, `DemographicPattern`, `CommunityReport`). Session defaults to SQLite locally; set `DATABASE_URL` for PostgreSQL.
  - `ingestion/` — Scrapers for PubMed (`pubmed_ingest`), clinical guidelines (`guidelines_ingest`), Reddit community posts (`community_ingest`), population/WHO data (`population_ingest`), articles (`article_scraper`), YouTube (`youtube_scraper`), and Notte (`notte_scraper`)
  - `pipeline/source_ingestion_pipeline.py` — Orchestrates 4-phase pipeline: source discovery → content extraction → text chunking → 30-day refresh/revalidation
  - `assessment/` — `questionnaire.py` (Pydantic input model), `scoring_model.py`, `pattern_engine.py`, `content_mapper.py`
  - `retrieval/` — `confidence_scoring.py`, `constraints_validator.py`, `content_classifier.py`, `domain_filter.py`, `tavily_search.py`
  - `scheduler/` — APScheduler (`BackgroundScheduler`) with SQLAlchemy job store. Runs `refresh_stale_pages` daily. Requires `DATABASE_URL` env var for the persistent job store (defaults to local PostgreSQL URL if not set).
  - `api/questionnaire_api.py` — APIRouter at `/api/questionnaire/evaluate` (POST); calls `evaluate_pcos` from `medical_assistant.py`
  - `storage/local_kb.py` — SQLite-based local knowledge base at `pocs/womens_health_pcos/data/knowledge_base.db`

### Frontend (React + Vite + Tailwind v4)

- **`App.jsx`** — Chat state management, axios calls to `localhost:8000`
- **`components/Message.jsx`** — Renders chat bubbles with source cards and reliability score bars. Exports `cn()` utility (clsx + twMerge).
- **`components/ChatInput.jsx`** — Textarea with Enter-to-submit; imports `cn` from `Message`.

### API Contract

```
POST /api/chat
Body:     { "query": string, "history": [{"role": string, "content": string}]? }
Response: { "response": string, "sources": [{ "url": string, "score": int, "type": string, "title": string, "source_label": string, "confidence_level": string }] }

POST /api/upload_prescription
Body:     multipart/form-data with one or more image files (JPEG/PNG/WebP, max 10MB each)
Response: { "extracted_text": string }

POST /api/questionnaire/evaluate
Body:     UserQuestionnaire fields (age, cycle_regularity, energy_levels, symptoms[], etc.)
Response: PCOS evaluation result dict
```

## Environment Variables

Required in `.env` at project root:
- `OPENAI_API_KEY` — OpenAI API key (used for `gpt-4o` vision + chat via `langchain-openai`)
- `TAVILY_API_KEY` — Tavily Search API key
- `DATABASE_URL` — (optional) PostgreSQL connection string; falls back to SQLite for the PCOS KB. Also required for the APScheduler persistent job store.

## Key Constraints

- LLM responses must be grounded ONLY in retrieved context — never use parametric knowledge for medical claims
- All search results pass through LLM-based verification before inclusion
- The system prompt forbids diagnoses, treatment recommendations, and predictions
- Source URLs are passed through programmatically (never LLM-generated) to prevent URL hallucination
- PCOS community/Reddit sources get confidence score 30 (LOW); clinical guidelines get 90 (HIGH)
