import sys
import os
import json
from datetime import datetime

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.registry_db import get_sources_for_refresh, update_source, save_page_content
from pocs.womens_health_pcos.extraction.web_scraper import scrape_url, generate_hash
from pocs.womens_health_pcos.processing.text_chunker import process_source_chunks
from pocs.womens_health_pcos.database.models import SourcePage

REFRESH_DAYS = 30

def run_refresh_job():
    print("Starting 30-Day Refresh Job...")
    init_db()
    db = SessionLocal()
    
    # Get pages older than 30 days or never checked
    pages_to_refresh = get_sources_for_refresh(db, days_older_than=REFRESH_DAYS)
    
    if not pages_to_refresh:
        print("No pages need refreshing at this time.")
        db.close()
        return
        
    print(f"Found {len(pages_to_refresh)} pages to refresh.")
    
    success_count = 0
    drift_count = 0
    dead_count = 0
    
    for page in pages_to_refresh:
        print(f"Refreshing: {page.page_url}")
        
        # We need to find the source_type for this page's root source
        from pocs.womens_health_pcos.database.models import Source
        source_root = db.query(Source).filter(Source.source_id == page.source_id).first()
        source_type = source_root.source_type if source_root else "article"

        content_dict = scrape_url(page.page_url, source_type)
        new_text = content_dict.get("raw_text", "")
        
        # Dead or 404
        if not new_text and not "simulated" in str(content_dict):
            print(f"  -> Deprecated/Broken link detected: {page.page_url}")
            update_source(db, page.page_id, {
                "is_active": 0,
                "http_status": 404,
                "last_checked": datetime.utcnow()
            })
            dead_count += 1
            continue
            
        new_hash = generate_hash(new_text)
        
        if new_hash != page.last_hash:
            print("  -> Content drift detected. Saving new snapshot...")
            
            # Save new snapshot in PageContent
            # save_page_content handles hash comparison and versioning internally too,
            # but we already checked hash here for logging.
            save_page_content(db, page.page_id, new_text, content_dict.get("extracted_sections", {}))
            
            # Re-fetch page object to get updated state
            updated_page = db.query(SourcePage).filter(SourcePage.page_id == page.page_id).first()
            process_source_chunks(db, updated_page)
            drift_count += 1
        else:
            print("  -> Content unchanged. Marked verified.")
            update_source(db, page.page_id, {
                "last_checked": datetime.utcnow()
            })
            success_count += 1

    print(f"Refresh completed: {success_count} unchanged, {drift_count} updated, {dead_count} broken/dead.")
    db.close()

if __name__ == "__main__":
    run_refresh_job()
