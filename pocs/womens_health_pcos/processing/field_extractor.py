"""
Structured field extraction from raw PageContent using LLM.
Populates domain tables (ResearchSource, LabMarker, etc.) with extracted fields.
Runs as a pipeline step after ingestion/chunking.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
load_dotenv()

from pocs.womens_health_pcos.database.session import SessionLocal, init_db
from pocs.womens_health_pcos.database.models import (
    PageContent, SourcePage, Source,
    ResearchSource, LabMarker, SymptomPattern,
)

EXTRACTION_PROMPT = """You are a medical content analyst. Extract structured fields from the following medical text about PCOS.

Return ONLY a JSON object with these fields (use null for missing fields):
{{
    "medical_topic": "string",
    "symptoms_identified": ["list of symptoms mentioned"],
    "lab_markers": ["list of lab tests/markers mentioned"],
    "treatments_mentioned": ["list of treatments/medications mentioned"],
    "lifestyle_recommendations": ["list of lifestyle suggestions"],
    "key_conclusions": "string summary of main findings",
    "study_population": "string or null",
    "sample_size": "integer or null"
}}

Text to analyze:
{content}
"""

MAX_CONTENT_FOR_LLM = 4000


def _get_llm():
    """Get an OpenAI client for extraction."""
    from openai import OpenAI
    return OpenAI()


def extract_fields_from_content(content_text: str) -> dict:
    """Use LLM to extract structured fields from raw content."""
    client = _get_llm()
    truncated = content_text[:MAX_CONTENT_FOR_LLM]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You extract structured medical data from text. Always respond with valid JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(content=truncated)},
            ],
            temperature=0,
            max_tokens=1000,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[FieldExtractor] LLM extraction failed: {e}")
        return {}


def run_field_extraction(limit: int = 50):
    """
    Process unextracted pages: extract structured fields and populate domain tables.
    Only processes research_paper and guideline source types.
    """
    init_db()
    db = SessionLocal()
    processed = 0

    try:
        # Find pages that have content but no corresponding ResearchSource entry
        pages_with_content = (
            db.query(SourcePage, PageContent, Source)
            .join(PageContent, PageContent.page_id == SourcePage.page_id)
            .join(Source, SourcePage.source_id == Source.source_id)
            .filter(
                SourcePage.is_active == 1,
                Source.source_type.in_(["research_paper", "guideline", "article"]),
            )
            .limit(limit)
            .all()
        )

        for page, content, source in pages_with_content:
            # Skip if already extracted
            existing = db.query(ResearchSource).filter(
                ResearchSource.source_link == page.page_url
            ).first()
            if existing:
                continue

            print(f"[FieldExtractor] Extracting: {page.page_url[:80]}...")
            fields = extract_fields_from_content(content.content_text)
            if not fields:
                continue

            # Store as ResearchSource
            record = ResearchSource(
                title=source.source_name or fields.get("medical_topic", "Unknown"),
                source_domain=source.base_url.replace("https://", "").replace("http://", "").split("/")[0],
                text_content=content.content_text[:5000],
                confidence_score=90 if source.trust_level == "HIGH" else 70,
                source_link=page.page_url,
                symptoms_identified=json.dumps(fields.get("symptoms_identified")) if fields.get("symptoms_identified") else None,
                lab_markers=json.dumps(fields.get("lab_markers")) if fields.get("lab_markers") else None,
                treatment_protocol=json.dumps(fields.get("treatments_mentioned")) if fields.get("treatments_mentioned") else None,
                key_conclusions=fields.get("key_conclusions"),
                study_population=fields.get("study_population"),
                sample_size=fields.get("sample_size"),
            )
            db.add(record)
            processed += 1

            # Also store lab markers if found
            for marker_name in (fields.get("lab_markers") or []):
                existing_marker = db.query(LabMarker).filter(
                    LabMarker.marker_name == marker_name
                ).first()
                if not existing_marker:
                    db.add(LabMarker(
                        title=f"Lab Marker: {marker_name}",
                        source_domain=source.base_url.replace("https://", "").replace("http://", "").split("/")[0],
                        text_content=f"Referenced in: {source.source_name}",
                        confidence_score=70,
                        source_link=page.page_url,
                        marker_name=marker_name,
                    ))

        db.commit()
        print(f"[FieldExtractor] Done. Extracted fields for {processed} pages.")

    except Exception as e:
        print(f"[FieldExtractor] Error: {e}")
        db.rollback()
    finally:
        db.close()

    return processed


if __name__ == "__main__":
    run_field_extraction()
