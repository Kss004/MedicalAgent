# PCOS Knowledge Pipeline — Detailed Documentation

A domain-specific women's health RAG system that ingests verified medical content, scores it by confidence tier, and serves it through a multi-turn conversational API with strict safety guardrails.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Flow](#2-data-flow)
3. [Database Schema](#3-database-schema)
4. [Retrieval Pipeline](#4-retrieval-pipeline)
5. [Assessment Engine](#5-assessment-engine)
6. [Ingestion Pipeline](#6-ingestion-pipeline)
7. [Extraction System](#7-extraction-system)
8. [Storage Layer](#8-storage-layer)
9. [Scheduler](#9-scheduler)
10. [Configuration](#10-configuration)
11. [API Integration](#11-api-integration)
12. [Safety Model](#12-safety-model)
13. [Running the Pipeline](#13-running-the-pipeline)

---

## 1. System Architecture

The pipeline has three distinct runtime modes that share the same underlying data stores:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PCOS PIPELINE OVERVIEW                       │
│                                                                   │
│  OFFLINE INGESTION             REAL-TIME RETRIEVAL               │
│  ─────────────────             ───────────────────               │
│  source_discovery              search_trusted_sources()          │
│       ↓                               ↓                          │
│  web_scraper / notte_scraper   Layer 1: Local KB (SQLite/PG)     │
│       ↓                               ↓ (miss)                   │
│  text_chunker                  Layer 2: Tavily → Filter pipeline │
│       ↓                               ↓                          │
│  ChunkMetadata (DB)            Layer 3: Community reports        │
│                                        ↓                          │
│  ASSESSMENT                    Diversity mix → LLM response      │
│  ─────────────                                                    │
│  UserQuestionnaire             SCHEDULER                         │
│       ↓                        ─────────                         │
│  scoring_model                 APScheduler (daily)               │
│       ↓                               ↓                          │
│  pattern_engine                refresh_stale_pages()             │
│       ↓                        (re-scrapes pages >7 days old)    │
│  content_mapper                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Map

```
pocs/womens_health_pcos/
├── api/
│   └── questionnaire_api.py      # FastAPI router: POST /api/questionnaire/evaluate
├── assessment/
│   ├── questionnaire.py          # Pydantic input model (UserQuestionnaire)
│   ├── scoring_model.py          # Rule-based PCOS risk score
│   ├── pattern_engine.py         # Symptom cluster detection
│   └── content_mapper.py         # Orchestrates assessment → educational content
├── configs/
│   ├── sources.yaml              # Enabled source systems + priorities
│   ├── confidence_rules.yaml     # Source type → confidence multiplier
│   └── pattern_rules.yaml        # Symptom keyword → pattern name mappings
├── data/
│   ├── knowledge_base.db         # Local SQLite KB (local_kb.py)
│   └── raw_sources/pcos/         # JSON dumps from web scraper
├── database/
│   ├── base.py                   # SQLAlchemy declarative Base
│   ├── models.py                 # ORM models (all tables)
│   ├── registry_db.py            # CRUD helpers for 3-layer Source→Page→Content
│   └── session.py                # Engine + SessionLocal + init_db()
├── extraction/
│   ├── article_extractor.py      # LLM-assisted article field extraction
│   ├── web_scraper.py            # cloudscraper + BeautifulSoup scraper
│   └── youtube_extractor.py      # youtube-transcript-api + LLM metadata
├── ingestion/
│   ├── article_scraper.py        # Article-specific ingestion
│   ├── community_ingest.py       # Reddit/Quora community posts
│   ├── guidelines_ingest.py      # Clinical guideline ingestion
│   ├── notte_scraper.py          # Notte.cc AI extraction + BS4 fallback
│   ├── population_ingest.py      # WHO/ICMR/CDC population data
│   ├── pubmed_ingest.py          # PubMed E-Utilities API
│   ├── seed_kb.py                # One-time KB seed from JSON files
│   ├── source_discovery.py       # Discover new URLs from root sources
│   └── youtube_scraper.py        # YouTube content ingestion
├── pipeline/
│   ├── knowledge_ingestion.py    # 7-stage retrieval pipeline orchestrator
│   ├── refresh_job.py            # 30-day content revalidation
│   └── source_ingestion_pipeline.py  # 4-phase offline ingestion entry point
├── processing/
│   └── text_chunker.py           # Sliding window text chunker
├── retrieval/
│   ├── confidence_scoring.py     # Domain → HIGH/MEDIUM/LOW scoring
│   ├── constraints_validator.py  # Quality/length validation gate
│   ├── content_classifier.py     # ARTICLE vs VIDEO classification
│   ├── domain_filter.py          # Strict domain allowlist filter
│   └── tavily_search.py          # MedicalSearchEngine wrapper
├── scheduler/
│   ├── jobs.py                   # Scheduled job definitions
│   └── scheduler_config.py       # APScheduler setup + start/stop
├── schema/
│   ├── models.py                 # Pydantic models for records (ArticleRecord, etc.)
│   └── postgres_schema.sql       # Raw SQL for PostgreSQL schema
└── storage/
    └── local_kb.py               # SQLite-backed local knowledge base
```

---

## 2. Data Flow

### 2a. Offline Ingestion (run once / scheduled)

```
source_discovery.run_discovery()
    │
    ├── Reads sources.yaml (pubmed, cdc, etc.)
    ├── Crawls root source URLs
    └── Creates Source + SourcePage rows in DB
            │
web_scraper.run_extraction()
    │
    ├── Queries SourcePages where last_hash IS NULL (never scraped)
    ├── cloudscraper → BeautifulSoup → raw_text (max 20,000 chars)
    ├── Saves JSON dump to data/raw_sources/pcos/{type}/{id}_{domain}.json
    └── Updates SourcePage: http_status, last_hash
            │
text_chunker.run_chunking()
    │
    ├── Queries SourcePages with last_hash != NULL and no existing ChunkMetadata
    ├── Fetches latest PageContent text
    ├── Chunks text: 1000-char window, 100-char overlap
    └── Stores ChunkMetadata rows (embedding_status='pending')
            │
refresh_job.run_refresh_job()
    │
    ├── Queries SourcePages not checked in >30 days
    └── Re-scrapes and updates content/hash/version
```

### 2b. Real-Time Chat Query

```
User sends POST /api/chat { query, history }
    │
ask_health_assistant(query, history)
    │
    ├── [1] contextualize_query() — rewrites follow-up using last 4 history messages
    │
    ├── [2] search_trusted_sources(query)
    │       │
    │       ├── Layer 1: PostgreSQL full-text keyword search on ChunkMetadata
    │       │           JOIN SourcePage → Source (filter by trust_level != LOW)
    │       │           → Up to 5 cached trusted results
    │       │
    │       ├── Layer 2: (if Layer 1 miss) Tavily broad search (max 20)
    │       │           → domain_filter (allowlist) → classify → score → validate
    │       │           → auto-cache validated results to local_kb.store()
    │       │
    │       ├── Layer 3: retrieve_community_reports()
    │       │           → PostgreSQL: chunk_text LIKE '%query_keyword%'
    │       │              source_type='community', confidence_score=30
    │       │
    │       └── _select_diverse_sources() → 3-4 medical + 1-2 community
    │           → _build_response() → {context, source_urls}
    │
    ├── [3] LLM call (gpt-4o-mini via FallbackLLM)
    │       system_prompt + {context} + history[-6] + query
    │
    └── Returns {response, sources[{url, score, type, title, source_label, confidence_level}]}
```

### 2c. PCOS Assessment Query

```
User sends POST /api/questionnaire/evaluate {age, symptoms, cycle_regularity, ...}
    │
evaluate_pcos(user_data_dict)
    │
evaluate_pcos_patterns(user_data_dict)   [content_mapper.py]
    │
    ├── UserQuestionnaire(**user_data)    # Pydantic validation + BMI calc
    │
    ├── calculate_risk_score(user_data)   # Rule-based score 0–10+
    │       cycles: +2 irregular/absent
    │       symptoms: +1 each (hirsutism, acne, hair_loss, weight_gain)
    │       lifestyle: +1 low energy; +1 BMI≥25
    │       history: +2 family PCOS; +1 family T2D
    │       age: +1 if 15–30
    │       → {total_score, category, contributing_factors, recommendation}
    │
    ├── detect_patterns(user_data)        # Symptom cluster rules
    │       hormonal_imbalance, insulin_resistance_pattern,
    │       reproductive_irregularity, metabolic_risk
    │
    ├── generate_educational_queries(patterns)
    │       → 1 query per pattern (up to 3 patterns)
    │
    ├── For each query: search_trusted_sources(query)
    │       → verified educational content snippets
    │
    └── Returns {assessment, patterns_detected, educational_payload, display_text}
```

---

## 3. Database Schema

The pipeline uses a **3-layer source tracking schema** (Source → SourcePage → PageContent) plus specialized content tables and a lean local SQLite KB.

### 3a. 3-Layer Source Registry (SQLAlchemy ORM)

```
Source (sources)
│   source_id, source_name, base_url (UNIQUE), source_type
│   trust_level: HIGH | MEDIUM | LOW
│   crawl_frequency: daily | weekly | monthly
│   is_active: 1 | 0
│
└── SourcePage (source_pages)
    │   page_id, source_id (→ Source), page_url (UNIQUE)
    │   parent_page_id (self-ref for nested crawl)
    │   crawl_depth, discovery_method
    │   page_status: active | deprecated | failed
    │   last_hash (SHA-256 of last scraped content)
    │   last_crawled, last_checked, http_status
    │   is_active: 1 | 0
    │
    └── PageContent (page_content)
        │   content_id, page_id (→ SourcePage)
        │   content_text, content_json (JSON as text)
        │   content_hash (SHA-256)
        │   version_number  ← increments only when content changes
        │   created_at
        │
        └── ChunkMetadata (chunk_metadata)
                chunk_id, source_id (→ SourcePage.page_id)
                chunk_index, chunk_text
                token_count, char_count
                embedding_status: pending | complete
```

**Content versioning:** `save_page_content()` hashes incoming text and only writes a new `PageContent` row if the hash differs from `SourcePage.last_hash`. This prevents duplicate snapshots on re-scrape.

### 3b. Specialized Content Tables (ORM)

| Table | Confidence | Key Extra Columns |
|---|---|---|
| `research_sources` | 70 (default) | year, study_population, sample_size, age_group, symptoms_identified, lab_markers, treatment_protocol, outcomes, key_conclusions |
| `symptom_patterns` | varies | pattern_cluster (e.g. `insulin_resistance`), associated_symptoms |
| `lab_markers` | varies | marker_name, normal_range, pcos_pattern |
| `treatment_protocols` | 90 | diagnostic_criteria, hormone_thresholds, recommended_tests, treatment_pathways, lifestyle_recommendations |
| `demographic_patterns` | 90 | age_distribution, urban_vs_rural_patterns, comorbidities, prevalence_rates |
| `community_reports` | 30 (LOW) | symptom, treatment, medication, lifestyle_change, reported_outcome, sentiment |

### 3c. Local SQLite Knowledge Base (`local_kb.py`)

Separate from the SQLAlchemy-managed DB. Backed by `data/knowledge_base.db`.

```
sources (SQLite)
    id, title, url (UNIQUE), source_domain, content_type
    content, confidence_level, confidence_score
    query_topics, source_label, scraped_at
```

- **Write:** `store(record)` → INSERT OR REPLACE by URL
- **Read:** `search(query, limit=10)` → `LIKE` keyword match on title/content/query_topics, ordered by `confidence_score DESC`
- Acts as a fast Layer 1 cache; populated automatically when Tavily results are validated

### 3d. Database Connection

`database/session.py` resolves the DB URL in this priority order:

1. `DATABASE_URL` env var → PostgreSQL (schema `pcos`, `search_path=pcos`)
2. Fallback → SQLite at `data/pcos_knowledge.db`

The APScheduler job store also uses `DATABASE_URL` to persist jobs (table: `apscheduler_jobs`, schema: `pcos`). If only SQLite is available, the scheduler reverts to in-memory jobs (lost on restart).

---

## 4. Retrieval Pipeline

All stages are deterministic (no LLM) except content extraction. Each stage is independently importable from `retrieval/`.

### Stage 1 — Tavily Search (`retrieval/tavily_search.py`)

```python
class MedicalSearchEngine:
    search(query: str) -> list[dict]
    # Returns: [{title, url, content, snippet}, ...]
    # max_results=20, no domain restriction at this stage
```

This is the only stage that hits an external API. Domain filtering happens next, not here.

### Stage 2 — Domain Filter (`retrieval/domain_filter.py`)

**Allowlist:**

| Domain | Category |
|---|---|
| `who.int` | International health authority |
| `cdc.gov` | US public health |
| `nih.gov`, `ncbi.nlm.nih.gov`, `pubmed.ncbi.nlm.nih.gov` | NIH research |
| `nhs.uk` | UK national health |
| `mayoclinic.org`, `clevelandclinic.org` | Academic medical centres |
| `endocrine.org` | Endocrinology society |
| `eshre.eu` | European reproductive medicine |
| `monash.edu` | PCOS research (Monash) |
| `icmr.gov.in` | India health authority |
| `youtube.com`, `youtu.be` | Video (filtered further by confidence) |

```python
filter_by_domain(results) -> (accepted: list, rejected: list)
# Deduplicates by URL; adds source_domain field to accepted results
```

Subdomain matching is supported — `discover.nih.gov` passes the `nih.gov` check.

### Stage 3 — Content Classifier (`retrieval/content_classifier.py`)

```python
classify(url: str) -> str   # "VIDEO" | "ARTICLE"
classify_results(results)   # mutates each dict to add content_type field
```

Only `youtube.com` and `youtu.be` are classified as VIDEO. Everything else is ARTICLE.

### Stage 4 — Confidence Scoring (`retrieval/confidence_scoring.py`)

```python
score_source(url, content_type, channel_name="") -> {confidence_level, confidence_score}
score_results(results) -> list[dict]  # adds confidence fields to each result
```

Scoring matrix:

| Domain Group | Content Type | Score | Level |
|---|---|---|---|
| WHO, CDC, NIH, NHS, Mayo, Cleveland, Endocrine, ESHRE, ICMR | ARTICLE | 95 | HIGH |
| PubMed, NCBI, Monash | ARTICLE | 75 | MEDIUM |
| Mayo Clinic YT, Cleveland Clinic YT, Khan Academy, TED-Ed, etc. | VIDEO | 82 | MEDIUM |
| Other YouTube | VIDEO | 65 | MEDIUM |
| Anything else | * | 20 | LOW |

### Stage 5 — Constraints Validator (`retrieval/constraints_validator.py`)

```python
validate(record) -> {valid: bool, reason: str}
validate_batch(records) -> (accepted: list, rejected: list)
```

Rules (all must pass):

1. URL starts with `http`
2. Confidence level is HIGH or MEDIUM (LOW is rejected)
3. Content field is not empty
4. ARTICLE: content ≥ 80 characters
5. VIDEO: has a title OR description ≥ 30 characters
6. URL has not already been seen in this batch (deduplication)

### Stage 6 — Source Diversity Mixer (`medical_assistant.py`)

After the filter pipeline, `_select_diverse_sources()` assembles the final set:

- **Target:** 3–4 medical sources + 1–2 community sources, max 5 total
- **Deduplication:** by domain (one result per root domain)
- **Video inclusion:** ensures at least one VIDEO source if available

### Stage 7 — Response Builder (`medical_assistant.py`)

`_build_response(sources, query, source_layer)` generates an LLM-ready context string:

```
[Source 1] (ARTICLE) Mayo Clinic — HIGH confidence (95/100)
URL: https://mayoclinic.org/...
...content snippet...

[Source 2] (VIDEO) Cleveland Clinic — MEDIUM confidence (82/100)
URL: https://youtube.com/...
...content snippet...
```

---

## 5. Assessment Engine

The assessment system converts a user questionnaire into a structured risk profile with attached educational content. It never accesses live data — all content comes from `search_trusted_sources()`.

### 5a. Input Model (`assessment/questionnaire.py`)

```python
class UserQuestionnaire(BaseModel):
    age: int                          # 10–99
    height_cm: float | None
    weight_kg: float | None
    cycle_regularity: str             # "regular" | "irregular" | "absent"
    cycle_length_days: int | None
    symptoms: list[str]               # ["acne", "hirsutism", "hair_loss", "weight_gain", "fatigue"]
    energy_levels: str                # "high" | "normal" | "low" | "exhausted"
    family_history_pcos: bool
    family_history_diabetes: bool
    primary_health_concern: str | None

    @property
    def bmi(self) -> float | None:    # computed from height/weight
```

### 5b. Risk Scoring (`assessment/scoring_model.py`)

```python
calculate_risk_score(user_data: dict) -> dict
```

Scoring rules:

| Condition | Points |
|---|---|
| Cycle: irregular or absent | +2 |
| Symptom: hirsutism | +1 |
| Symptom: acne | +1 |
| Symptom: hair_loss | +1 |
| Symptom: weight_gain | +1 |
| Energy: low or exhausted | +1 |
| BMI ≥ 25 | +1 |
| Family history: PCOS | +2 |
| Family history: Type 2 Diabetes | +1 |
| Age 15–30 | +1 |

Risk categories:

| Score | Category | Recommendation |
|---|---|---|
| 0–2 | LOW | General wellness focus |
| 3–5 | MODERATE | Monitoring recommended |
| 6+ | HIGH | Consult healthcare provider |

Output includes: `total_score`, `category`, `contributing_factors` (list of triggered rules), `recommendation`, `safety_disclaimer`.

### 5c. Pattern Detection (`assessment/pattern_engine.py`)

```python
detect_patterns(user_data: dict) -> list[str]
```

Pattern rules (evaluated in order):

| Pattern | Condition |
|---|---|
| `hormonal_imbalance` | (acne OR hirsutism OR hair_loss) AND (irregular OR absent cycles) |
| `insulin_resistance_pattern` | (weight_gain OR low energy) AND (family T2D OR BMI≥25) |
| `reproductive_irregularity` | irregular/absent cycles AND NOT hormonal_imbalance |
| `metabolic_risk` | BMI > 30 |

```python
generate_educational_queries(patterns: list[str]) -> list[str]
```

Pattern → search query mapping:

| Pattern | Search Query |
|---|---|
| `hormonal_imbalance` | "PCOS hormonal imbalance symptoms and lifestyle management" |
| `insulin_resistance_pattern` | "PCOS insulin resistance weight gain and dietary treatment" |
| `reproductive_irregularity` | "PCOS irregular cycles anovulation causes" |
| `metabolic_risk` | "PCOS metabolic syndrome risk factors and exercise" |
| (none) | "PCOS general health guidelines and cycle tracking" |

### 5d. Content Mapper (`assessment/content_mapper.py`)

```python
evaluate_pcos_patterns(user_data_dict: dict) -> dict
```

Full output structure:

```json
{
  "assessment": {
    "total_score": 7,
    "category": "HIGH",
    "contributing_factors": ["irregular_cycles", "hirsutism", "family_history_pcos"],
    "recommendation": "...",
    "safety_disclaimer": "..."
  },
  "patterns_detected": ["hormonal_imbalance", "insulin_resistance_pattern"],
  "educational_payload": [
    {
      "theme": "hormonal_imbalance",
      "verified_sources": [...],
      "content": "..."
    }
  ],
  "display_text": "## PCOS Pattern Analysis\n..."
}
```

---

## 6. Ingestion Pipeline

The offline ingestion pipeline has four phases. Run via:

```bash
python pocs/womens_health_pcos/pipeline/source_ingestion_pipeline.py
```

### Phase 1 — Source Discovery (`ingestion/source_discovery.py`)

Reads the seed sources from `configs/sources.yaml` and from manually defined root URLs, then crawls them to discover article/video/guideline URLs. Creates `Source` + `SourcePage` rows in the DB.

### Phase 2 — Content Extraction (`extraction/web_scraper.py`)

For each `SourcePage` where `last_hash IS NULL`:

1. Tries `cloudscraper` (anti-bot) to fetch the page
2. Falls back to `requests` + `BeautifulSoup`
3. Extracts text from `<p>`, `<h1>`–`<h6>`, `<li>` tags; removes nav/footer/script noise
4. Truncates to 20,000 characters
5. Saves a JSON dump to `data/raw_sources/pcos/{type}/{page_id}_{domain}.json`
6. Updates `SourcePage.http_status`

Alternatively, `notte_scraper.py` uses the Notte.cc AI extraction API (requires `NOTTE_API_KEY`) with BeautifulSoup as fallback. Notte produces higher-quality structured extractions and also triggers `process_source_chunks()` immediately after scraping.

### Phase 3 — Chunk Preparation (`processing/text_chunker.py`)

```python
chunk_text(text, chunk_size=1000, overlap=100) -> list[str]
```

Character-based sliding window. Designed for future vector embedding (currently `embedding_status='pending'`).

```python
process_source_chunks(db, page: SourcePage) -> int
```

For a given page:
1. Fetches the latest `PageContent` record
2. Chunks the content text
3. Writes `ChunkMetadata` rows
4. Returns chunk count

```python
run_chunking()
```

Queries all active pages with content but no existing chunks, processes each one.

### Phase 4 — Refresh & Revalidation (`pipeline/refresh_job.py`)

```python
run_refresh_job()
```

Queries `SourcePage` records where `last_checked` is more than 30 days ago and re-scrapes them. This is the batch version; the scheduler runs `refresh_stale_pages()` (7-day threshold) daily.

### Specialized Ingestors

| File | Source | Method |
|---|---|---|
| `pubmed_ingest.py` | PubMed E-Utilities API | Searches PCOS MeSH terms, fetches abstracts via E-fetch |
| `guidelines_ingest.py` | ACOG, ESHRE, Endocrine Society | Direct URL scraping of clinical guideline pages |
| `community_ingest.py` | Reddit (r/PCOS) | PRAW Reddit API, top posts by flair |
| `population_ingest.py` | WHO, CDC, ICMR | Direct scraping of prevalence/population data pages |
| `article_scraper.py` | General medical articles | URL list scraping |
| `youtube_scraper.py` | YouTube | `yt-dlp` metadata + `youtube-transcript-api` for captions |
| `seed_kb.py` | Existing JSON dumps | One-time seed from `data/raw_sources/` into `local_kb` |

---

## 7. Extraction System

LLM-assisted field extraction transforms raw scraped text into structured records.

### Article Extraction (`extraction/article_extractor.py`)

```python
extract_article_fields(content, url, title="", llm=None) -> dict
```

Without LLM: returns a base record with `raw_article_text` populated and other fields empty.

With LLM: sends a structured extraction prompt requesting JSON output with:
- `title`, `source`, `publication_year`, `medical_topic`
- `key_symptoms` (list), `lab_markers` (list), `treatments` (list)
- `lifestyle_recommendations` (list), `key_conclusions`

Handles LLM output cleaning (strips markdown code fences), validates with `json.loads()`, falls back to base record on parse error.

### YouTube Extraction (`extraction/youtube_extractor.py`)

```python
extract_video_fields(content, url, title="", llm=None) -> dict
```

1. Extracts `video_id` via regex (handles `v=`, `/v/`, `youtu.be/`, `shorts/`)
2. Fetches transcript via `youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)` → space-joined string
3. With LLM: extracts `channel_name`, `medical_topic`, `video_summary`, `published_date`
4. Gracefully returns empty strings if transcript is unavailable or video is private

---

## 8. Storage Layer

### Local Knowledge Base (`storage/local_kb.py`)

The SQLite KB (`data/knowledge_base.db`) acts as Layer 1 cache for the retrieval pipeline. It is populated:
- Automatically when Tavily results pass validation (`store()` called in `search_trusted_sources`)
- Manually by running `seed_kb.py` over existing JSON dumps

Key methods:

```python
store(record: dict) -> bool          # True if newly inserted, False if duplicate URL
store_batch(records: list) -> int    # Returns count of newly stored
search(query: str, limit=10) -> list # LIKE match, ordered by confidence_score DESC
count() -> int
get_all_domains() -> list[str]
```

### PostgreSQL Knowledge Base (via SQLAlchemy)

Used for production and when `DATABASE_URL` is set. The `ChunkMetadata` table is the primary Layer 1 source for retrieval — `search_trusted_sources()` queries it with keyword matching and joins through to `SourcePage` and `Source` for trust-level filtering.

---

## 9. Scheduler

### Setup (`scheduler/scheduler_config.py`)

APScheduler `BackgroundScheduler` configured with:
- **Job store:** `SQLAlchemyJobStore` → PostgreSQL table `pcos.apscheduler_jobs` (persists jobs across restarts)
- **Executor:** `ThreadPoolExecutor(10)` workers
- **Timezone:** UTC
- **`coalesce: False`** — missed jobs do not run catch-up executions
- **`max_instances: 1`** — prevents overlapping runs

```python
start_scheduler()    # idempotent, checks scheduler.running
shutdown_scheduler() # idempotent
```

### Jobs (`scheduler/jobs.py`)

```python
refresh_stale_pages()
# Finds SourcePages with last_checked > 7 days ago
# Calls scrape_and_store(db, page) for each
# Logs success count
```

The `daily_refresh` job is registered on FastAPI startup in `server.py`:

```python
scheduler.add_job(refresh_stale_pages, 'interval', days=1, id='daily_refresh', replace_existing=True)
```

Inspect scheduled jobs at runtime:

```
GET /api/scheduler/jobs
→ { "jobs": [{"id", "next_run_time", "trigger"}], "running": bool }
```

---

## 10. Configuration

### `configs/sources.yaml`

Defines which source systems are active and their crawl priority:

```yaml
pubmed:
  enabled: true
  priority: 1
cdc:
  enabled: true
  priority: 2
```

### `configs/confidence_rules.yaml`

Source type → confidence multiplier (used by scoring logic):

```yaml
peer_reviewed: 0.9
clinical_guideline: 1.0
community_anecdote: 0.3
```

### `configs/pattern_rules.yaml`

Keyword → pattern name mappings for rule-based symptom detection:

```yaml
irregular_periods: ["irregular", "missed", "delayed"]
hirsutism: ["excessive hair", "facial hair"]
```

### Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM calls (gpt-4o-mini) + FallbackLLM |
| `TAVILY_API_KEY` | Yes | Layer 2 search |
| `DATABASE_URL` | No | PostgreSQL connection; falls back to SQLite |
| `NOTTE_API_KEY` | No | Notte.cc AI extraction in notte_scraper.py |

---

## 11. API Integration

The PCOS pipeline is integrated into the main FastAPI server (`server.py`) via:

### Questionnaire Endpoint

```
POST /api/questionnaire/evaluate
Content-Type: application/json

{
  "age": 28,
  "cycle_regularity": "irregular",
  "energy_levels": "low",
  "symptoms": ["acne", "weight_gain", "hirsutism"],
  "height_cm": 165,
  "weight_kg": 85,
  "family_history_pcos": false,
  "family_history_diabetes": true,
  "primary_health_concern": "Hair loss and irregular periods"
}
```

Response:

```json
{
  "assessment": {
    "total_score": 8,
    "category": "HIGH",
    "contributing_factors": ["irregular_cycles", "acne", "weight_gain", "hirsutism", "low_energy", "high_bmi", "family_diabetes"],
    "recommendation": "...",
    "safety_disclaimer": "..."
  },
  "patterns_detected": ["hormonal_imbalance", "insulin_resistance_pattern"],
  "educational_payload": [...],
  "display_text": "## PCOS Pattern Analysis\n..."
}
```

### Chat Endpoint (PCOS topics)

The main `POST /api/chat` endpoint automatically routes PCOS-related queries through `search_trusted_sources()`, which queries the PCOS knowledge base first (Layer 1) before falling back to Tavily.

---

## 12. Safety Model

The pipeline enforces safety at multiple layers:

| Layer | Mechanism | What it prevents |
|---|---|---|
| Domain allowlist | `domain_filter.py` | Non-medical or unreliable sources entering the KB |
| Confidence gate | `constraints_validator.py` | LOW-confidence sources used in responses |
| Content length gate | `constraints_validator.py` | Stubs or empty pages treated as valid sources |
| LLM system prompt | `medical_assistant.py` | Diagnoses, medication recommendations, certainty claims |
| Source attribution | `_build_response()` | Untraced claims — every fact requires `[Source N]` |
| URL provenance | Pipeline-only URLs | LLM-hallucinated URLs in responses |
| Community labelling | `confidence_scoring.py` (score=30) | Community anecdotes treated as clinical evidence |
| Response fallback | `search_trusted_sources()` | Silent failure — returns explicit "no verified sources found" |

**The LLM is never given the ability to generate source URLs.** All URLs come from the retrieval pipeline and are attached post-generation.

---

## 13. Running the Pipeline

### Initialize the database

```bash
python -c "from pocs.womens_health_pcos.database.session import init_db; init_db()"
```

### Run a quick functional test

```bash
# DB schema + ingestion + evaluate_pcos
python tests/test_pcos_pipeline.py

# Source/Page/Content versioning
python tests/test_ingestion_schema.py

# Scheduler jobs
python tests/test_scheduler.py
```

### Run the full offline ingestion

```bash
python pocs/womens_health_pcos/pipeline/source_ingestion_pipeline.py
```

This runs all 4 phases sequentially. Each phase catches exceptions independently, so a failure in source discovery will not block content extraction on already-known pages.

### Migrate legacy data

If you have an older flat `source_registry` table, migrate to the 3-layer schema:

```bash
python migrate_pcos_data.py
```

### Seed the local KB from existing JSON dumps

```bash
python pocs/womens_health_pcos/ingestion/seed_kb.py
```

### Run individual ingestors (for testing or targeted refresh)

```bash
python -c "from pocs.womens_health_pcos.ingestion import guidelines_ingest; guidelines_ingest.run_ingestion()"
python -c "from pocs.womens_health_pcos.ingestion import community_ingest; community_ingest.run_ingestion()"
python -c "from pocs.womens_health_pcos.ingestion import population_ingest; population_ingest.run_ingestion()"
# Note: pubmed_ingest hits E-Utilities API — avoid in rapid testing (rate limits)
```
