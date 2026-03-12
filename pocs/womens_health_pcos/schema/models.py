"""
Pydantic Models for the Verified Medical Retrieval Pipeline.

Defines structured data models for articles, videos, and retrieval results.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ArticleRecord(BaseModel):
    """Structured record for an article source."""
    title: str = ""
    source: str = ""
    source_link: str = ""
    source_domain: str = ""
    publication_year: str = ""
    medical_topic: str = ""
    content_type: str = "ARTICLE"
    confidence_level: str = ""
    confidence_score: int = 0
    query_topic: str = ""
    key_symptoms: list[str] = Field(default_factory=list)
    lab_markers: list[str] = Field(default_factory=list)
    treatments: list[str] = Field(default_factory=list)
    lifestyle_recommendations: list[str] = Field(default_factory=list)
    key_conclusions: str = ""
    raw_article_text: str = ""
    created_at: Optional[datetime] = None


class VideoRecord(BaseModel):
    """Structured record for a video source."""
    video_title: str = ""
    channel_name: str = ""
    video_url: str = ""
    video_id: str = ""
    video_description: str = ""
    published_date: str = ""
    medical_topic: str = ""
    video_summary: str = ""
    transcript: str = ""
    content_type: str = "VIDEO"
    confidence_level: str = ""
    confidence_score: int = 0
    query_topic: str = ""
    created_at: Optional[datetime] = None


class SourceReference(BaseModel):
    """Minimal source reference for API responses."""
    url: str
    score: int
    type: str  # "article" or "video"


class RetrievalResult(BaseModel):
    """Complete pipeline output."""
    articles: list[ArticleRecord] = Field(default_factory=list)
    videos: list[VideoRecord] = Field(default_factory=list)
    context: str = ""
    source_urls: list[SourceReference] = Field(default_factory=list)
    message: str = ""

class SourceRegistryRecord(BaseModel):
    """Structured record for the source registry table."""
    source_id: Optional[int] = None
    url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    domain: str
    topic: Optional[str] = None
    source_type: str
    confidence_level: str
    discovery_method: Optional[str] = None
    is_active: bool = True
    http_status: Optional[int] = None
    content_hash: Optional[str] = None
    last_scraped_at: Optional[datetime] = None
    last_checked_at: Optional[datetime] = None
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class ChunkMetadataRecord(BaseModel):
    """Structured record for the chunk metadata table."""
    chunk_id: Optional[int] = None
    source_id: int
    chunk_index: int
    chunk_text: str
    token_count: Optional[int] = None
    char_count: Optional[int] = None
    embedding_status: str = 'pending'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

