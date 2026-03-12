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
from pocs.womens_health_pcos.database.models import SourceRegistry

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
    
    # Get unprocessed sources
    pending_sources = db.query(SourceRegistry).filter(SourceRegistry.last_scraped_at == None, SourceRegistry.is_active == 1).all()
    print(f"Found {len(pending_sources)} pending sources to scrape.")
    
    success_count = 0
    fail_count = 0
    
    for source in pending_sources:
        print(f"Scraping [{source.source_type}]: {source.url}")
        content_dict = scrape_url(source.url, source.source_type)
        
        if not content_dict.get("raw_text") and not "simulated" in str(content_dict):
            fail_count += 1
            source.http_status = 404 # Placeholder for general fail
            db.commit()
            continue
            
        raw_text = content_dict.get("raw_text", "")
        content_hash = generate_hash(raw_text)
        
        file_name = f"{source.source_id}_{source.domain}.json".replace(" ", "_")
        # map source type to valid sub-directory fallback
        sub_dir = source.source_type + "s" if source.source_type in ["article", "guideline", "video"] else source.source_type 
        if not os.path.exists(os.path.join(BASE_STORAGE_DIR, sub_dir)):
            sub_dir = "articles" # Fallback mapping
            
        relative_path = os.path.join(sub_dir, file_name)
        absolute_path = os.path.join(BASE_STORAGE_DIR, relative_path)
        
        json_payload = {
            "source_id": source.source_id,
            "url": source.url,
            "title": source.title,
            "domain": source.domain,
            "scraped_at": datetime.utcnow().isoformat(),
            "source_type": source.source_type,
            "topic": source.topic,
            "confidence_level": source.confidence_level,
            "raw_text": raw_text,
            "extracted_sections": content_dict.get("extracted_sections", {}),
            "metadata": {"discovery_method": source.discovery_method}
        }
        
        try:
            with open(absolute_path, 'w', encoding='utf-8') as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)
                
            # Update DB Registry
            source.last_scraped_at = datetime.utcnow()
            source.content_hash = content_hash
            source.file_path = relative_path
            source.http_status = 200
            db.commit()
            success_count += 1
        except Exception as e:
            print(f"Error saving to disk for {source.source_id}: {e}")
            fail_count += 1

    print(f"Extraction complete. Scraped {success_count} sources. Failed {fail_count}.")
    db.close()

if __name__ == "__main__":
    run_extraction()
