import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, asc
from pocs.womens_health_pcos.database.models import SourceRegistry, ChunkMetadata
from pocs.womens_health_pcos.schema.models import SourceRegistryRecord, ChunkMetadataRecord

def get_source_by_url(db: Session, url: str) -> SourceRegistry:
    return db.query(SourceRegistry).filter(SourceRegistry.url == url).first()

def create_source(db: Session, source_record: SourceRegistryRecord) -> SourceRegistry:
    db_source = SourceRegistry(
        url=source_record.url,
        canonical_url=source_record.canonical_url,
        title=source_record.title,
        domain=source_record.domain,
        topic=source_record.topic,
        source_type=source_record.source_type,
        confidence_level=source_record.confidence_level,
        discovery_method=source_record.discovery_method,
        is_active=int(source_record.is_active),
        http_status=source_record.http_status,
        content_hash=source_record.content_hash,
        last_scraped_at=source_record.last_scraped_at,
        last_checked_at=source_record.last_checked_at,
        file_path=source_record.file_path
    )
    db.add(db_source)
    db.commit()
    db.refresh(db_source)
    return db_source

def update_source(db: Session, source_id: int, updates: dict) -> SourceRegistry:
    db_source = db.query(SourceRegistry).filter(SourceRegistry.source_id == source_id).first()
    if db_source:
        for key, value in updates.items():
            setattr(db_source, key, value)
        db.commit()
        db.refresh(db_source)
    return db_source

def get_sources_for_refresh(db: Session, days_older_than: int = 30):
    """Get all active sources that haven't been checked in `days_older_than` days."""
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days_older_than)
    return db.query(SourceRegistry).filter(
        SourceRegistry.is_active == 1,
        or_(
            SourceRegistry.last_checked_at == None,
            SourceRegistry.last_checked_at < cutoff_date
        )
    ).order_by(asc(SourceRegistry.last_checked_at)).all()

def add_chunks(db: Session, chunks_data: list[ChunkMetadataRecord]):
    """Bulk insert chunks into chunk metadata."""
    db_chunks = [
        ChunkMetadata(
            source_id=c.source_id,
            chunk_index=c.chunk_index,
            chunk_text=c.chunk_text,
            token_count=c.token_count,
            char_count=c.char_count,
            embedding_status=c.embedding_status
        ) for c in chunks_data
    ]
    db.bulk_save_objects(db_chunks)
    db.commit()

def clear_chunks_for_source(db: Session, source_id: int):
    """Remove existing chunks when refreshing a source."""
    db.query(ChunkMetadata).filter(ChunkMetadata.source_id == source_id).delete()
    db.commit()
