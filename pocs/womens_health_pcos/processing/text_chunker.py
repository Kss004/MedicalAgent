import sys
import os
import json

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import SourcePage, ChunkMetadata
from pocs.womens_health_pcos.database.registry_db import add_chunks, clear_chunks_for_source
from pocs.womens_health_pcos.schema.models import ChunkMetadataRecord
from pocs.womens_health_pcos.extraction.web_scraper import BASE_STORAGE_DIR

def chunk_text(text: str, chunk_size=1000, overlap=100) -> list[str]:
    """Simple character-based sliding window chunking. Future proof for Langchain/LlamaIndex."""
    chunks = []
    if not text:
        return chunks
    
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start += (chunk_size - overlap)
        
    return chunks

def process_source_chunks(db, page: SourcePage) -> int:
    """Find the latest content for a page, chunk its text, store metadata."""
    from pocs.womens_health_pcos.database.models import PageContent
    
    # Get latest content snapshot
    latest_content = db.query(PageContent).filter(PageContent.page_id == page.page_id).order_by(PageContent.version_number.desc()).first()
    
    if not latest_content:
        print(f"No content found for page {page.page_id}: {page.page_url}")
        return 0
        
    try:
        raw_text = latest_content.content_text
        if not raw_text:
            return 0
            
        # Clear existing chunks if this is a refresh
        clear_chunks_for_source(db, page.page_id)
        
        text_chunks = chunk_text(raw_text)
        chunk_records = []
        for i, chunk in enumerate(text_chunks):
            # Token count estimation: ~4 chars per token roughly
            token_count_est = len(chunk) // 4
            record = ChunkMetadataRecord(
                source_id=page.page_id, # Using page_id as the source_id for chunks now
                chunk_index=i,
                chunk_text=chunk,
                token_count=token_count_est,
                char_count=len(chunk),
                embedding_status='pending'
            )
            chunk_records.append(record)
            
        if chunk_records:
            add_chunks(db, chunk_records)
            
        return len(chunk_records)
        
    except Exception as e:
        print(f"Error chunking page {page.page_id}: {e}")
        return 0

def run_chunking():
    print("Starting Chunk Preparation...")
    init_db()
    db = SessionLocal()
    
    # We will chunk all active pages that have a hash (meaning they have been crawled)
    pages = db.query(SourcePage).filter(SourcePage.is_active == 1, SourcePage.last_hash != None).all()
    print(f"Found {len(pages)} pages to check for chunking.")
    
    total_chunks = 0
    processed_pages = 0
    
    for page in pages:
        # Check if chunks already exist
        existing_count = db.query(ChunkMetadata).filter(ChunkMetadata.source_id == page.page_id).count()
        if existing_count > 0:
            continue # already chunked
            
        print(f"Chunking page {page.page_id}: {page.page_url}")
        num_chunks = process_source_chunks(db, page)
        if num_chunks > 0:
            processed_pages += 1
            total_chunks += num_chunks
            
    print(f"Chunking complete. Created {total_chunks} chunks for {processed_pages} pages.")
    db.close()

if __name__ == "__main__":
    run_chunking()
