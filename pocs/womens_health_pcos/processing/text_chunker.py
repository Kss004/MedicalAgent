import sys
import os
import json

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import SourceRegistry
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

def process_source_chunks(db, source: SourceRegistry) -> int:
    """Read the JSON file for a source, chunk its raw text, store metadata."""
    if not source.file_path:
        return 0
        
    abs_path = os.path.join(BASE_STORAGE_DIR, source.file_path)
    if not os.path.exists(abs_path):
        print(f"File missing for {source.source_id}: {source.file_path}")
        return 0
        
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        raw_text = data.get("raw_text", "")
        if not raw_text:
            return 0
            
        # Clear existing chunks if this is a refresh
        clear_chunks_for_source(db, source.source_id)
        
        text_chunks = chunk_text(raw_text)
        chunk_records = []
        for i, chunk in enumerate(text_chunks):
            # Token count estimation: ~4 chars per token roughly
            token_count_est = len(chunk) // 4
            record = ChunkMetadataRecord(
                source_id=source.source_id,
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
        print(f"Error chunking source {source.source_id}: {e}")
        return 0

def run_chunking():
    print("Starting Chunk Preparation...")
    init_db()
    db = SessionLocal()
    
    # We will chunk all active sources that have a file_path
    # A real system might track `last_chunked_at` or similar to avoid re-chunking.
    # We will just process all valid sources here for demonstration, or we can check ChunkMetadata to skip.
    
    sources = db.query(SourceRegistry).filter(SourceRegistry.is_active == 1, SourceRegistry.file_path != None).all()
    print(f"Found {len(sources)} sources to check for chunking.")
    
    total_chunks = 0
    processed_sources = 0
    
    for source in sources:
        # Check if chunks already exist
        from pocs.womens_health_pcos.database.models import ChunkMetadata
        existing_count = db.query(ChunkMetadata).filter(ChunkMetadata.source_id == source.source_id).count()
        if existing_count > 0:
            continue # already chunked
            
        print(f"Chunking source {source.source_id}")
        num_chunks = process_source_chunks(db, source)
        if num_chunks > 0:
            processed_sources += 1
            total_chunks += num_chunks
            
    print(f"Chunking complete. Created {total_chunks} chunks for {processed_sources} sources.")
    db.close()

if __name__ == "__main__":
    run_chunking()
