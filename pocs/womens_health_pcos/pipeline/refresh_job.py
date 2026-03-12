import sys
import os
import json
from datetime import datetime

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.registry_db import get_sources_for_refresh, update_source
from pocs.womens_health_pcos.extraction.web_scraper import scrape_url, generate_hash, BASE_STORAGE_DIR
from pocs.womens_health_pcos.processing.text_chunker import process_source_chunks

REFRESH_DAYS = 30

def run_refresh_job():
    print("Starting 30-Day Refresh Job...")
    init_db()
    db = SessionLocal()
    
    # Get sources older than 30 days or never checked
    sources_to_refresh = get_sources_for_refresh(db, days_older_than=REFRESH_DAYS)
    
    if not sources_to_refresh:
        print("No sources need refreshing at this time.")
        db.close()
        return
        
    print(f"Found {len(sources_to_refresh)} sources to refresh.")
    
    success_count = 0
    drift_count = 0
    dead_count = 0
    
    for source in sources_to_refresh:
        print(f"Refreshing: {source.url}")
        
        content_dict = scrape_url(source.url, source.source_type)
        new_text = content_dict.get("raw_text", "")
        
        # Dead or 404
        if not new_text and not "simulated" in str(content_dict):
            print(f"  -> Deprecated/Broken link detected: {source.url}")
            update_source(db, source.source_id, {
                "is_active": 0,
                "http_status": 404,
                "last_checked_at": datetime.utcnow()
            })
            dead_count += 1
            continue
            
        new_hash = generate_hash(new_text)
        
        if new_hash != source.content_hash:
            print("  -> Content drift detected. Updating...")
            
            # Update json file
            if source.file_path:
                abs_path = os.path.join(BASE_STORAGE_DIR, source.file_path)
                try:
                    with open(abs_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    data['raw_text'] = new_text
                    data['extracted_sections'] = content_dict.get("extracted_sections", {})
                    data['scraped_at'] = datetime.utcnow().isoformat()
                    
                    with open(abs_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        
                    # Re-chunk
                    # Requires writing chunks directly without rewriting DB source yet
                    # We will update the DB object and commit after this.
                except Exception as e:
                    print(f"  -> Error rewriting file {abs_path}: {e}")
                    
            update_source(db, source.source_id, {
                "content_hash": new_hash,
                "last_scraped_at": datetime.utcnow(),
                "last_checked_at": datetime.utcnow()
            })
            
            # Rechunk
            # Fetch updated object
            from pocs.womens_health_pcos.database.models import SourceRegistry
            updated_source = db.query(SourceRegistry).filter(SourceRegistry.source_id == source.source_id).first()
            process_source_chunks(db, updated_source)
            drift_count += 1
        else:
            print("  -> Content unchanged. Marked verified.")
            update_source(db, source.source_id, {
                "last_checked_at": datetime.utcnow()
            })
            success_count += 1

    print(f"Refresh completed: {success_count} unchanged, {drift_count} updated, {dead_count} broken/dead.")
    db.close()

if __name__ == "__main__":
    run_refresh_job()
