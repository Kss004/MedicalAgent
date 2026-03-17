import datetime
import hashlib
import json
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, asc, desc
from pocs.womens_health_pcos.database.models import Source, SourcePage, PageContent, ChunkMetadata
from pocs.womens_health_pcos.schema.models import SourceRecord, SourcePageRecord, PageContentRecord, ChunkMetadataRecord

# --- New 3-Layer Structure Functions ---

def hash_content(text: str) -> str:
    """Compute a deterministic SHA-256 hash of cleaned text."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

def get_source_by_base_url(db: Session, base_url: str) -> Source:
    return db.query(Source).filter(Source.base_url == base_url).first()

def create_source_root(db: Session, record: SourceRecord) -> Source:
    db_source = Source(
        source_name=record.source_name,
        base_url=record.base_url,
        source_type=record.source_type,
        trust_level=record.trust_level,
        crawl_frequency=record.crawl_frequency,
        is_active=int(record.is_active)
    )
    db.add(db_source)
    db.commit()
    db.refresh(db_source)
    return db_source

def get_page_by_url(db: Session, url: str) -> SourcePage:
    return db.query(SourcePage).filter(SourcePage.page_url == url).first()

def create_page(db: Session, record: SourcePageRecord) -> SourcePage:
    db_page = SourcePage(
        source_id=record.source_id,
        page_url=record.page_url,
        parent_page_id=record.parent_page_id,
        crawl_depth=record.crawl_depth,
        discovery_method=record.discovery_method,
        page_status=record.page_status,
        is_active=int(record.is_active)
    )
    db.add(db_page)
    db.commit()
    db.refresh(db_page)
    return db_page

def save_page_content(db: Session, page_id: int, text: str, json_data: dict = None) -> PageContent:
    """
    Saves page content snapshot if hash has changed.
    Updates SourcePage timestamps and last_hash.
    """
    new_hash = hash_content(text)
    db_page = db.query(SourcePage).filter(SourcePage.page_id == page_id).first()
    
    if not db_page:
        return None

    # Update check/crawl timestamps
    db_page.last_checked = datetime.datetime.utcnow()
    db_page.last_crawled = datetime.datetime.utcnow()
    
    # Check for change
    if db_page.last_hash == new_hash:
        db.commit()
        return None # No change, no new snapshot

    # Versioning: Get latest version number
    latest_content = db.query(PageContent).filter(PageContent.page_id == page_id).order_by(desc(PageContent.version_number)).first()
    new_version = (latest_content.version_number + 1) if latest_content else 1

    # Create new snapshot
    db_content = PageContent(
        page_id=page_id,
        content_text=text,
        content_json=json.dumps(json_data) if json_data else None,
        content_hash=new_hash,
        version_number=new_version
    )
    
    db_page.last_hash = new_hash
    db_page.http_status = 200 # Assuming success if we are saving content
    
    db.add(db_content)
    db.commit()
    db.refresh(db_content)
    return db_content

# --- Legacy Support Functions ---

# Legacy Support Stubs (To be deprecated)

def get_source_by_url(db: Session, url: str) -> SourcePage:
    return get_page_by_url(db, url)

def create_source(db: Session, source_record: any) -> SourcePage:
    """Redirect legacy create_source to new Page-based logic."""
    domain = source_record.domain
    source_root = get_source_by_base_url(db, f"https://{domain}")
    if not source_root:
        source_root = create_source_root(db, SourceRecord(
            source_name=domain,
            base_url=f"https://{domain}",
            source_type=source_record.source_type,
            trust_level="MEDIUM"
        ))
    
    return create_page(db, SourcePageRecord(
        source_id=source_root.source_id,
        page_url=source_record.url,
        discovery_method=source_record.discovery_method,
        is_active=source_record.is_active
    ))

def update_source(db: Session, source_id: int, updates: dict) -> any:
    """Legacy update stub."""
    db_page = db.query(SourcePage).filter(SourcePage.page_id == source_id).first()
    if db_page:
        for key, value in updates.items():
            if hasattr(db_page, key):
                setattr(db_page, key, value)
        db.commit()
        db.refresh(db_page)
    return db_page



def get_sources_for_refresh(db: Session, days_older_than: int = 30):
    """Get all active source pages that haven't been checked in `days_older_than` days."""
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=days_older_than)
    return db.query(SourcePage).filter(
        SourcePage.is_active == 1,
        or_(
            SourcePage.last_checked == None,
            SourcePage.last_checked < cutoff_date
        )
    ).order_by(asc(SourcePage.last_checked)).all()

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
