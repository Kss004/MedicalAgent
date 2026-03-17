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

class SourceRecord(BaseModel):
    """Pydantic model for clinical source root."""
    source_id: Optional[int] = None
    source_name: str
    base_url: str
    source_type: str
    trust_level: str = "MEDIUM"
    crawl_frequency: str = "weekly"
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class SourcePageRecord(BaseModel):
    """Pydantic model for individual source pages."""
    page_id: Optional[int] = None
    source_id: int
    page_url: str
    parent_page_id: Optional[int] = None
    crawl_depth: int = 0
    discovery_method: Optional[str] = None
    page_status: str = "active"
    last_hash: Optional[str] = None
    last_crawled: Optional[datetime] = None
    last_checked: Optional[datetime] = None
    http_status: Optional[int] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class PageContentRecord(BaseModel):
    """Pydantic model for page content snapshots."""
    content_id: Optional[int] = None
    page_id: int
    content_text: str
    content_json: Optional[str] = None
    content_hash: str
    version_number: int = 1
    created_at: Optional[datetime] = None

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

