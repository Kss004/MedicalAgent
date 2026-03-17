import sys
import os
import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import Source, SourcePage, PageContent
from pocs.womens_health_pcos.database.registry_db import (
    get_source_by_base_url, create_source_root, 
    get_page_by_url, create_page, save_page_content
)
from pocs.womens_health_pcos.schema.models import SourceRecord, SourcePageRecord

def test_schema():
    print("Initializing Database...")
    init_db()
    db = SessionLocal()
    
    try:
        # 1. Test Source Root Creation
        base_url = "https://example-medical.org"
        print(f"\nStep 1: Creating Source Root for {base_url}")
        source = get_source_by_base_url(db, base_url)
        if not source:
            source = create_source_root(db, SourceRecord(
                source_name="Example Medical",
                base_url=base_url,
                source_type="article",
                trust_level="HIGH"
            ))
        print(f"Source Created: ID={source.source_id}")

        # 2. Test Page Creation (Nested)
        page_url = f"{base_url}/pcos-intro"
        print(f"\nStep 2: Creating Page: {page_url}")
        page = get_page_by_url(db, page_url)
        if not page:
            page = create_page(db, SourcePageRecord(
                source_id=source.source_id,
                page_url=page_url,
                discovery_method="manual",
                crawl_depth=0
            ))
        print(f"Page Created: ID={page.page_id}")

        # 3. Test Nested Page
        sub_page_url = f"{base_url}/pcos-intro/symptoms"
        print(f"\nStep 3: Creating Nested Page: {sub_page_url}")
        sub_page = get_page_by_url(db, sub_page_url)
        if not sub_page:
            sub_page = create_page(db, SourcePageRecord(
                source_id=source.source_id,
                page_url=sub_page_url,
                parent_page_id=page.page_id,
                discovery_method="crawler",
                crawl_depth=1
            ))
        print(f"Nested Page Created: ID={sub_page.page_id}, ParentID={sub_page.parent_page_id}, Depth={sub_page.crawl_depth}")

        # 4. Test Content Hashing and Versioning
        print("\nStep 4: Testing Content Versioning")
        content_v1_text = "PCOS is a hormonal disorder."
        print("Saving Version 1...")
        save_page_content(db, page.page_id, content_v1_text, {"meta": "v1"})
        
        db.refresh(page)
        hash_v1 = page.last_hash
        print(f"V1 Hash: {hash_v1}")

        print("\nSaving identical content (should not create new version)...")
        save_page_content(db, page.page_id, content_v1_text, {"meta": "v1-dup"})
        
        count = db.query(PageContent).filter(PageContent.page_id == page.page_id).count()
        print(f"Snapshot count (Expected 1): {count}")

        print("\nSaving changed content (should create Version 2)...")
        content_v2_text = "PCOS is a common hormonal disorder among women."
        save_page_content(db, page.page_id, content_v2_text, {"meta": "v2"})
        
        count = db.query(PageContent).filter(PageContent.page_id == page.page_id).count()
        db.refresh(page)
        print(f"Snapshot count (Expected 2): {count}")
        print(f"V2 Hash: {page.last_hash}")
        
        if count == 2 and page.last_hash != hash_v1:
            print("\nSUCCESS: Schema and Versioning verified!")
        else:
            print("\nFAILED: Versioning or hashing logic error.")

    finally:
        db.close()

if __name__ == "__main__":
    test_schema()
