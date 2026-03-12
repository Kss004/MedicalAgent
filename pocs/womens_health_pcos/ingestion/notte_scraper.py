"""
Standalone Scraper using Notte.cc for high-quality extraction.
Falls back to BeautifulSoup for reliability.
"""

import os
import sys
import json
import hashlib
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Add root project path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import SourceRegistry
from pocs.womens_health_pcos.database.registry_db import update_source
from pocs.womens_health_pcos.processing.text_chunker import process_source_chunks

BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw_sources")

def get_notte_content(url: str):
    """
    Experimental Notte.cc extraction.
    Note: Requires NOTTE_API_KEY in environment.
    """
    api_key = os.getenv("NOTTE_API_KEY")
    if not api_key:
        return None
    
    # Placeholder for actual Notte.cc SDK/API call
    # Assuming Notte provides a simple 'scrape' endpoint that returns clean markdown/text
    # and structured metadata.
    try:
        # This is a hypothetical implementation based on common AI scraping patterns
        # Replace with actual SDK once confirmed
        headers = {"Authorization": f"Bearer {api_key}"}
        # In a real scenario, we'd use the Notte SDK: 
        # from notte import Notte; client = Notte(api_key); data = client.scrape(url)
        response = requests.post("https://api.notte.cc/v1/scrape", json={"url": url}, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[Notte] API failed for {url}: {e}")
    return None

def bs4_fallback(url: str):
    """Clean text extraction using BeautifulSoup."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 MedicalAgent/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        return {
            "title": soup.title.string.strip() if soup.title else "Untitled Article",
            "raw_text": text,
            "extracted_sections": {"headings": [h.get_text().strip() for h in soup.find_all(['h1', 'h2', 'h3'])]}
        }
    except Exception as e:
        print(f"[Fallback] BeautifulSoup failed for {url}: {e}")
        return None

def scrape_and_store(db, source: SourceRegistry):
    """Scrape a single URL using Notte or Fallback, then store as JSON and update DB."""
    print(f"Scraping: {source.url}")
    
    # 1. Try Notte
    data = get_notte_content(source.url)
    method = "notte"
    
    # 2. Try Fallback
    if not data:
        print(f"  -> Falling back to BeautifulSoup...")
        data = bs4_fallback(source.url)
        method = "bs4_fallback"
        
    if not data or not data.get("raw_text"):
        print(f"  -> Error: No content extracted for {source.url}")
        return False
        
    # 3. Save to JSON
    domain_folder = os.path.join(BASE_STORAGE_DIR, "pcos", source.source_type)
    os.makedirs(domain_folder, exist_ok=True)
    
    filename = f"{source.source_id}_{source.domain}.json"
    file_path = os.path.join("pcos", source.source_type, filename)
    abs_path = os.path.join(BASE_STORAGE_DIR, file_path)
    
    full_data = {
        "source_id": source.source_id,
        "url": source.url,
        "title": data.get("title", source.title),
        "domain": source.domain,
        "scraped_at": datetime.utcnow().isoformat(),
        "source_type": source.source_type,
        "topic": source.topic,
        "confidence_level": source.confidence_level,
        "raw_text": data.get("raw_text"),
        "extracted_sections": data.get("extracted_sections", {}),
        "metadata": {
            "scraper_method": method,
            "discovery_method": source.discovery_method
        }
    }
    
    with open(abs_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False)
        
    # 4. Update DB Registry
    content_hash = hashlib.md5(data.get("raw_text").encode('utf-8')).hexdigest()
    update_source(db, source.source_id, {
        "title": full_data["title"],
        "file_path": file_path,
        "content_hash": content_hash,
        "last_scraped_at": datetime.utcnow(),
        "http_status": 200
    })
    
    # 5. Trigger Chunking - Refresh object to ensure file_path is loaded
    db.commit()
    db.refresh(source)
    process_source_chunks(db, source)
    
    return True

def run_notte_ingestion(limit=500):
    """Ingest up to `limit` sources from the registry that haven't been scraped yet."""
    init_db()
    db = SessionLocal()
    
    # Get active sources without a file_path (meaning they haven't been scraped)
    sources = db.query(SourceRegistry).filter(
        SourceRegistry.is_active == 1,
        SourceRegistry.file_path == None
    ).limit(limit).all()
    
    print(f"Found {len(sources)} sources in registry to scrape.")
    
    success_count = 0
    for source in sources:
        if scrape_and_store(db, source):
            success_count += 1
            
    print(f"Ingestion complete. Successfully scraped {success_count} sources.")
    db.close()

if __name__ == "__main__":
    # If called directly, scrape up to 500 sources
    run_notte_ingestion(500)
