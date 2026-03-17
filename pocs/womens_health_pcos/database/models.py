from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from pocs.womens_health_pcos.database.base import Base

try:
    from pgvector.sqlalchemy import Vector
    _HAS_PGVECTOR = True
except ImportError:
    _HAS_PGVECTOR = False

class ResearchSource(Base):
    """Stores PubMed and other research papers."""
    __tablename__ = "research_sources"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="ARTICLE")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, default=70) # Medium confidence
    source_link = Column(String, unique=True, nullable=False)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    # Paper-specific metadata
    year = Column(Integer, nullable=True)
    study_population = Column(String, nullable=True)
    sample_size = Column(Integer, nullable=True)
    age_group = Column(String, nullable=True)
    symptoms_identified = Column(String, nullable=True)
    lab_markers = Column(String, nullable=True)
    treatment_protocol = Column(String, nullable=True)
    outcomes = Column(String, nullable=True)
    key_conclusions = Column(Text, nullable=True)


class SymptomPattern(Base):
    """Stores symptom relationships and cluster definitions."""
    __tablename__ = "symptom_patterns"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="PATTERN")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, nullable=False)
    source_link = Column(String, nullable=True)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    pattern_cluster = Column(String, nullable=False) # e.g. insulin_resistance
    associated_symptoms = Column(String, nullable=False)


class LabMarker(Base):
    """Stores information about relevant lab tests and thresholds."""
    __tablename__ = "lab_markers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="LAB_GUIDE")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, nullable=False)
    source_link = Column(String, nullable=True)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    marker_name = Column(String, nullable=False)
    normal_range = Column(String, nullable=True)
    pcos_pattern = Column(String, nullable=True)


class TreatmentProtocol(Base):
    """Stores clinical guidelines for treatments and lifestyle changes."""
    __tablename__ = "treatment_protocols"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="GUIDELINE")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, default=90) # High confidence
    source_link = Column(String, unique=True, nullable=False)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    diagnostic_criteria = Column(Text, nullable=True)
    symptom_requirements = Column(Text, nullable=True)
    hormone_thresholds = Column(Text, nullable=True)
    recommended_tests = Column(Text, nullable=True)
    treatment_pathways = Column(Text, nullable=True)
    lifestyle_recommendations = Column(Text, nullable=True)


class DemographicPattern(Base):
    """Stores population health data (WHO, ICMR, etc.)."""
    __tablename__ = "demographic_patterns"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="POPULATION_DATA")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, default=90)
    source_link = Column(String, nullable=False)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    age_distribution = Column(String, nullable=True)
    urban_vs_rural_patterns = Column(String, nullable=True)
    lifestyle_factors = Column(String, nullable=True)
    comorbidities = Column(String, nullable=True)
    prevalence_rates = Column(String, nullable=True)


class CommunityReport(Base):
    """Stores anecdotal community experiences (Reddit, Quora, etc.)."""
    __tablename__ = "community_reports"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    source_domain = Column(String, nullable=False)
    content_type = Column(String, default="ANECDOTAL")
    text_content = Column(Text, nullable=False)
    confidence_score = Column(Integer, default=30) # LOW confidence
    source_link = Column(String, unique=True, nullable=False)
    date_added = Column(DateTime(timezone=True), server_default=func.now())
    
    symptom = Column(String, nullable=True)
    treatment = Column(String, nullable=True)
    medication = Column(String, nullable=True)
    lifestyle_change = Column(String, nullable=True)
    reported_outcome = Column(String, nullable=True)
    sentiment = Column(String, nullable=True)

class Source(Base):
    """Website/Domain level root source."""
    __tablename__ = "sources"

    source_id = Column(Integer, primary_key=True, index=True)
    source_name = Column(String, nullable=False)
    base_url = Column(String, unique=True, nullable=False, index=True)
    source_type = Column(String, nullable=False) # guideline, research_paper, article, community, video
    trust_level = Column(String, default="MEDIUM") # HIGH, MEDIUM, LOW
    crawl_frequency = Column(String, default="weekly") # daily, weekly, monthly
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class SourcePage(Base):
    """Actual discovered URLs belonging to a Source."""
    __tablename__ = "source_pages"

    page_id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, nullable=False, index=True)
    page_url = Column(String, unique=True, nullable=False, index=True)
    parent_page_id = Column(Integer, nullable=True)
    crawl_depth = Column(Integer, default=0)
    discovery_method = Column(String, nullable=True)
    page_status = Column(String, default="active") # active, deprecated, failed
    last_hash = Column(String, nullable=True)
    last_crawled = Column(DateTime(timezone=True), nullable=True)
    last_checked = Column(DateTime(timezone=True), nullable=True)
    http_status = Column(Integer, nullable=True)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class PageContent(Base):
    """Snapshots of extracted content for individual pages."""
    __tablename__ = "page_content"

    content_id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, nullable=False, index=True)
    content_text = Column(Text, nullable=False)
    content_json = Column(Text, nullable=True) # JSON stored as text
    content_hash = Column(String, nullable=False)
    version_number = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChunkMetadata(Base):
    """Metadata for textual chunks to be vectorized."""
    __tablename__ = "chunk_metadata"
    
    chunk_id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, nullable=False, index=True) # Assuming no hard FK restraint in simplest implementation, or DB enforces it
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=True)
    char_count = Column(Integer, nullable=True)
    embedding_status = Column(String, default='pending')
    embedding = Column(Vector(1536), nullable=True) if _HAS_PGVECTOR else None
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
