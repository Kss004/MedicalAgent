import datetime
from pocs.womens_health_pcos.database.session import SessionLocal
from pocs.womens_health_pcos.database.registry_db import get_sources_for_refresh
from pocs.womens_health_pcos.ingestion.notte_scraper import scrape_and_store

def refresh_stale_pages():
    """
    Finds pages that haven't been checked in more than 7 days 
    and triggers a refresh to check for content updates.
    """
    print(f"[Job] Starting stale page refresh at {datetime.datetime.utcnow()}")
    db = SessionLocal()
    try:
        # Get pages older than 7 days
        stale_pages = get_sources_for_refresh(db, days_older_than=7)
        print(f"[Job] Found {len(stale_pages)} stale pages to refresh.")
        
        success = 0
        for page in stale_pages:
            try:
                if scrape_and_store(db, page):
                    success += 1
            except Exception as e:
                print(f"[Job] Failed to refresh page {page.page_id}: {e}")
        
        print(f"[Job] Refresh complete. Successfully updated {success}/{len(stale_pages)} pages.")
    except Exception as e:
        print(f"[Job] Refresh job CRITICAL failure: {e}")
    finally:
        db.close()
