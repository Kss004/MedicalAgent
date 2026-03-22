"""Step 2: Generate instruction-tuning QA pairs from cleaned PCOS articles.

Uses GPT-4o-mini to produce grounded Q&A pairs with citations and safety
disclaimers. Adapts the number of pairs per article based on source type.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI

from config import (
    CLEANED_DIR,
    QA_PAIRS_DIR,
    QA_MODEL,
    QA_TEMPERATURE,
    QA_MAX_TOKENS,
    SOURCE_WEIGHTS,
)

client = OpenAI()   # uses OPENAI_API_KEY from env

# ── Prompt template ────────────────────────────────────────────────────────────
_QA_SYSTEM = """\
You are a medical education dataset creator. Given an article about PCOS, \
generate high-quality question-answer pairs for training a healthcare assistant.

Rules for EVERY answer:
1. Ground the answer ONLY in the provided article text — never invent facts.
2. Include an inline citation: [Source: {title}]
3. Never diagnose or prescribe medication.
4. End with: "Please consult a healthcare provider for personalized advice."
5. Write in clear, accessible language.

Rules for questions:
- Cover different aspects of the article (causes, symptoms, management, etc.)
- Vary question types: what, how, why, can, does, etc.
- Make questions realistic — things a patient or curious person would ask.
- Do NOT ask questions the article cannot answer.

Return a JSON array of objects with "question" and "answer" keys. Nothing else.\
"""

_QA_USER = """\
Article title: {title}
Source: {domain}
Confidence level: {confidence_level}
Source type: {source_type}

Generate exactly {num_pairs} question-answer pairs from this article:

---
{text}
---

Return ONLY a JSON array: [{{"question": "...", "answer": "..."}}]\
"""


def _get_num_pairs(source_type: str) -> int:
    """Return number of QA pairs to generate based on source type."""
    import random
    cfg = SOURCE_WEIGHTS.get(source_type, {"qa_per_file": (2, 4)})
    lo, hi = cfg["qa_per_file"]
    return random.randint(lo, hi)


def generate_pairs_for_article(article: dict) -> list[dict]:
    """Call GPT-4o-mini to generate QA pairs for one cleaned article."""
    source_type = article.get("source_type", "article")
    num_pairs = _get_num_pairs(source_type)

    # Truncate very long texts to stay within context limits
    text = article["cleaned_text"][:12000]

    user_msg = _QA_USER.format(
        title=article.get("title", "Untitled"),
        domain=article.get("domain", "unknown"),
        confidence_level=article.get("confidence_level", "MEDIUM"),
        source_type=source_type,
        num_pairs=num_pairs,
        text=text,
    )

    try:
        resp = client.chat.completions.create(
            model=QA_MODEL,
            temperature=QA_TEMPERATURE,
            max_tokens=QA_MAX_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _QA_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)

        # Handle various response shapes from the API:
        #   [...] — direct list of pairs
        #   {"pairs": [...]} or {"questions": [...]} — list nested under a key
        #   {"question": "...", "answer": "..."} — single pair as a dict
        if isinstance(parsed, list):
            pairs = parsed
        elif isinstance(parsed, dict):
            # Check for known wrapper keys first
            for key in ("pairs", "questions", "qa_pairs"):
                if key in parsed and isinstance(parsed[key], list):
                    pairs = parsed[key]
                    break
            else:
                # If the dict itself has "question"+"answer" keys, it's a single pair
                if "question" in parsed and "answer" in parsed:
                    pairs = [parsed]
                else:
                    # Try first list-valued entry
                    pairs = None
                    for v in parsed.values():
                        if isinstance(v, list):
                            pairs = v
                            break
                    if pairs is None:
                        print(f"  [WARN] Unexpected response shape for {article.get('source_id')}: {list(parsed.keys())}")
                        return []
        else:
            return []

        # Filter to only dicts with question+answer keys
        pairs = [p for p in pairs if isinstance(p, dict) and "question" in p and "answer" in p]

        # Attach metadata
        for p in pairs:
            p["source_id"] = article.get("source_id")
            p["source_url"] = article.get("url")
            p["source_title"] = article.get("title")
            p["source_type"] = source_type
            p["confidence_level"] = article.get("confidence_level", "MEDIUM")

        return pairs

    except Exception as e:
        print(f"  [ERROR] QA gen failed for {article.get('source_id')}: {e}")
        return []


def run() -> list[dict]:
    """Generate QA pairs for all cleaned articles."""
    QA_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    # Load cleaned articles
    cleaned_files = sorted(CLEANED_DIR.glob("*.json"))
    cleaned_files = [f for f in cleaned_files if f.name != "_manifest.json"]
    print(f"[Step 2] Generating QA pairs for {len(cleaned_files)} cleaned articles...")

    all_pairs = []
    for i, fpath in enumerate(cleaned_files):
        article = json.loads(fpath.read_text(encoding="utf-8"))
        pairs = generate_pairs_for_article(article)
        all_pairs.extend(pairs)

        # Save per-article pairs
        out_path = QA_PAIRS_DIR / f"{article.get('source_id', i)}_qa.json"
        out_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(cleaned_files)} — {len(all_pairs)} pairs so far")

        # Rate limiting: ~3 requests/sec for gpt-4o-mini
        time.sleep(0.35)

    # Save combined
    combined_path = QA_PAIRS_DIR / "_all_qa_pairs.json"
    combined_path.write_text(json.dumps(all_pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[Step 2] Done: {len(all_pairs)} QA pairs from {len(cleaned_files)} articles")

    # Stats
    by_type: dict[str, int] = {}
    for p in all_pairs:
        st = p.get("source_type", "unknown")
        by_type[st] = by_type.get(st, 0) + 1
    print(f"  By source type: {by_type}")

    return all_pairs


if __name__ == "__main__":
    run()
