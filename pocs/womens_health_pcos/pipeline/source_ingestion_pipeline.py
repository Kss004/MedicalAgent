import sys
import os

# Add root project path to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from pocs.womens_health_pcos.ingestion.source_discovery import run_discovery
from pocs.womens_health_pcos.extraction.web_scraper import run_extraction
from pocs.womens_health_pcos.processing.text_chunker import run_chunking
from pocs.womens_health_pcos.pipeline.refresh_job import run_refresh_job

def run_pipeline():
    print("=" * 60)
    print("WOMEN'S HEALTH PIPELINE: DATA INGESTION & KNOWLEDGE BUILDING")
    print("=" * 60)
    
    # 1. Discover New Sources
    print("\n>>> PHASE 1: SOURCE DISCOVERY")
    try:
        run_discovery()
    except Exception as e:
        print(f"Error in Source Discovery: {e}")

    # 2. Extract Raw Content
    print("\n>>> PHASE 2: CONTENT EXTRACTION (SCRAPING)")
    try:
        run_extraction()
    except Exception as e:
        print(f"Error in Content Extraction: {e}")

    # 3. Prepare Text Chunks for Vector DB
    print("\n>>> PHASE 3: CHUNK PREPARATION")
    try:
        run_chunking()
    except Exception as e:
        print(f"Error in Chunk Preparation: {e}")

    # 4. Refresh & Revalidation (runs asynchronously or daily in a real env)
    print("\n>>> PHASE 4: REFRESH & REVALIDATION (30-DAY CHECK)")
    try:
        run_refresh_job()
    except Exception as e:
        print(f"Error in Refresh Job: {e}")
        
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
