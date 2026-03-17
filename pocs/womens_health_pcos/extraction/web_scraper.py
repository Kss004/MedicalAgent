import sys
import os
import json
import hashlib
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import Source, SourcePage
from pocs.womens_health_pcos.database.registry_db import save_page_content

BASE_STORAGE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
    "data", "raw_sources", "pcos"
)

def ensure_directories():
    folders = ["articles", "guidelines", "research_paper", "public_health", "videos", "community"]
    for f in folders:
        os.makedirs(os.path.join(BASE_STORAGE_DIR, f), exist_ok=True)

def generate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def scrape_url(url: str, source_type: str) -> dict:
    # Extremely basic scraping for demonstration.
    # In a real system you'd use trafilatura, Youtube API etc.
    if source_type == "video" and "youtube.com" in url:
        return {"raw_text": f"Simulated transcript extraction for {url}.", "extracted_sections": {}}

    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
        resp = scraper.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Strip script, style, nav, footer, header
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        # simplistic structural extraction
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        
        return {"raw_text": text[:20000], "extracted_sections": {"headings": headings}}
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return {"raw_text": "", "error": str(e)}

def run_extraction():
    print("Starting Content Extraction...")
    init_db()
    ensure_directories()
    db = SessionLocal()
    
    # Get unprocessed source pages
    pending_pages = db.query(SourcePage).filter(SourcePage.last_hash == None, SourcePage.is_active == 1).all()
    print(f"Found {len(pending_pages)} pending pages to scrape.")
    
    success_count = 0
    fail_count = 0
    
    for page in pending_pages:
        # Get Source Root for type info
        source = db.query(Source).filter(Source.source_id == page.source_id).first()
        stype = source.source_type if source else "article"
        
        print(f"Scraping [{stype}]: {page.page_url}")
        content_dict = scrape_url(page.page_url, stype)
        
        if not content_dict.get("raw_text") and not "simulated" in str(content_dict):
            fail_count += 1
            page.http_status = 404
            db.commit()
            continue
            
        raw_text = content_dict.get("raw_text", "")
        domain_label = page.page_url.replace("https://", "").replace("http://", "").split("/")[0]
        
        file_name = f"{page.page_id}_{domain_label}.json".replace(" ", "_")
        sub_dir = stype + "s" if stype in ["article", "guideline", "video"] else stype 
        if not os.path.exists(os.path.join(BASE_STORAGE_DIR, sub_dir)):
            sub_dir = "articles"
            
        absolute_path = os.path.join(BASE_STORAGE_DIR, sub_dir, file_name)
        
        json_payload = {
            "page_id": page.page_id,
            "url": page.page_url,
            "title": domain_label,
            "scraped_at": datetime.utcnow().isoformat(),
            "source_type": stype,
            "raw_text": raw_text,
            "extracted_sections": content_dict.get("extracted_sections", {}),
            "metadata": {"discovery_method": page.discovery_method}
        }
        
        try:
            with open(absolute_path, 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)
                
            # Update DB via hashing/snapshot logic
            save_page_content(db, page.page_id, raw_text, json_data=content_dict.get("extracted_sections", {}))
            success_count += 1
        except Exception as e:
            print(f"Error saving to disk for page {page.page_id}: {e}")
            fail_count += 1

    print(f"Extraction complete. Scraped {success_count} sources. Failed {fail_count}.")
    db.close()

if __name__ == "__main__":
    run_extraction()
