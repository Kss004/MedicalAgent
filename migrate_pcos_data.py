import sys
import os

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocs.womens_health_pcos.database.session import SessionLocal
from pocs.womens_health_pcos.database.models import Source, SourcePage
from sqlalchemy import text

def migrate_legacy_data():
    db = SessionLocal()
    try:
        # Check if source_registry table exists
        res = db.execute(text("SELECT COUNT(*) FROM pcos.source_registry"))
        count = res.scalar()
        print(f"Found {count} legacy records in pcos.source_registry")
        
        if count == 0:
            print("No records found. Exiting.")
            return

        # Fetch records with explicit column mapping including file_path
        query = text("SELECT url, title, domain, source_type, confidence_level, is_active, file_path FROM pcos.source_registry")
        rows = db.execute(query).fetchall()
        
        migrated_roots = {} # base_url -> source_id
        migrated_count = 0
        content_count = 0

        # Import save_page_content and BASE_STORAGE_DIR
        from pocs.womens_health_pcos.database.registry_db import save_page_content
        from pocs.womens_health_pcos.extraction.web_scraper import BASE_STORAGE_DIR
        import json

        for row in rows:
            url, title, domain, stype, trust, is_active, file_path = row
            
            base_url = f"https://{domain}" if not domain.startswith('http') else domain
            if 'reddit.com' in domain:
                base_url = 'https://reddit.com'
            
            # 1. Ensure Source Root
            if base_url not in migrated_roots:
                existing_source = db.query(Source).filter(Source.base_url == base_url).first()
                if not existing_source:
                    new_source = Source(
                        source_name=domain,
                        base_url=base_url,
                        source_type=stype,
                        trust_level=trust,
                        is_active=int(is_active)
                    )
                    db.add(new_source)
                    db.flush()
                    migrated_roots[base_url] = new_source.source_id
                else:
                    migrated_roots[base_url] = existing_source.source_id
            
            # 2. Add Page
            existing_page = db.query(SourcePage).filter(SourcePage.page_url == url).first()
            if not existing_page:
                new_page = SourcePage(
                    source_id=migrated_roots[base_url],
                    page_url=url,
                    discovery_method="migration",
                    is_active=int(is_active)
                )
                db.add(new_page)
                db.flush()
                migrated_page = new_page
                migrated_count += 1
            else:
                migrated_page = existing_page
            
            # 3. Migrate Content if file_path exists
            if file_path:
                abs_path = os.path.join(BASE_STORAGE_DIR, file_path)
                if os.path.exists(abs_path):
                    try:
                        with open(abs_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        raw_text = data.get("raw_text", "")
                        if raw_text:
                            save_page_content(db, migrated_page.page_id, raw_text, data.get("extracted_sections", {}))
                            content_count += 1
                    except Exception as e:
                        print(f"  -> Failed to migrate content from {abs_path}: {e}")
        
        db.commit()
        print(f"Migration complete: Added {migrated_count} pages, migrated {content_count} content snapshots.")

        # 4. Trigger Chunking
        print("Triggering chunking for migrated data...")
        from pocs.womens_health_pcos.processing.text_chunker import run_chunking
        run_chunking()

    except Exception as e:
        import traceback
        print(f"Migration failed: {e}")
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    migrate_legacy_data()
