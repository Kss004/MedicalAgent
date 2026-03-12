-- Verified Medical Retrieval Pipeline — PostgreSQL Schema
-- Supports PCOS knowledge base with article + video content types

-- Core table for all retrieved sources
CREATE TABLE IF NOT EXISTS research_sources (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    source_url TEXT UNIQUE NOT NULL,
    source_domain TEXT NOT NULL,
    content_type TEXT NOT NULL CHECK (content_type IN ('ARTICLE', 'VIDEO')),
    confidence_level TEXT NOT NULL CHECK (confidence_level IN ('HIGH', 'MEDIUM', 'LOW_VIDEO_CONFIDENCE')),
    confidence_score INTEGER CHECK (confidence_score BETWEEN 0 AND 100),
    query_topic TEXT,
    raw_content TEXT,
    extracted_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Symptom patterns extracted from sources
CREATE TABLE IF NOT EXISTS symptom_patterns (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES research_sources(id) ON DELETE CASCADE,
    symptom_name TEXT NOT NULL,
    prevalence TEXT,
    severity TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lab markers referenced in sources
CREATE TABLE IF NOT EXISTS lab_markers (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES research_sources(id) ON DELETE CASCADE,
    marker_name TEXT NOT NULL,
    normal_range TEXT,
    pcos_range TEXT,
    significance TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Treatment protocols
CREATE TABLE IF NOT EXISTS treatment_protocols (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES research_sources(id) ON DELETE CASCADE,
    treatment_name TEXT NOT NULL,
    treatment_type TEXT, -- 'medication', 'lifestyle', 'surgical', 'supplement'
    evidence_level TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Demographic patterns
CREATE TABLE IF NOT EXISTS demographic_patterns (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES research_sources(id) ON DELETE CASCADE,
    population TEXT,
    age_group TEXT,
    prevalence_rate TEXT,
    region TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Community reports and patient experience data
CREATE TABLE IF NOT EXISTS community_reports (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES research_sources(id) ON DELETE CASCADE,
    report_type TEXT, -- 'experience', 'survey', 'case_study'
    summary TEXT,
    sentiment TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sources_topic ON research_sources(query_topic);
CREATE INDEX IF NOT EXISTS idx_sources_type ON research_sources(content_type);
CREATE INDEX IF NOT EXISTS idx_sources_confidence ON research_sources(confidence_level);
CREATE INDEX IF NOT EXISTS idx_symptoms_name ON symptom_patterns(symptom_name);
CREATE INDEX IF NOT EXISTS idx_lab_markers_name ON lab_markers(marker_name);

-- Source Registry for large-scale ingestion pipeline
CREATE TABLE IF NOT EXISTS source_registry (
    source_id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    canonical_url TEXT,
    title TEXT,
    domain TEXT NOT NULL,
    topic TEXT,
    source_type TEXT NOT NULL,
    confidence_level TEXT NOT NULL,
    discovery_method TEXT,
    is_active BOOLEAN DEFAULT true,
    http_status INTEGER,
    content_hash TEXT,
    last_scraped_at TIMESTAMP,
    last_checked_at TIMESTAMP,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chunk Metadata for vectorization
CREATE TABLE IF NOT EXISTS chunk_metadata (
    chunk_id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES source_registry(source_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_count INTEGER,
    char_count INTEGER,
    embedding_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_source_registry_url ON source_registry(url);
CREATE INDEX IF NOT EXISTS idx_source_registry_domain ON source_registry(domain);
CREATE INDEX IF NOT EXISTS idx_source_registry_type_conf ON source_registry(source_type, confidence_level);
CREATE INDEX IF NOT EXISTS idx_chunk_source_id ON chunk_metadata(source_id);
