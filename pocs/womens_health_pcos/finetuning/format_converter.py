"""Step 4a: Convert QA pairs to ShareGPT chat format as JSONL.

Produces {"conversations": [{"from": "system", ...}, {"from": "human", ...}, {"from": "gpt", ...}]}
format compatible with Unsloth/TRL training.
"""

import json
from pathlib import Path

from config import QA_PAIRS_DIR, TRAINING_DIR, SYSTEM_PROMPT, CONFIDENCE_WEIGHTS


def _to_sharegpt(pair: dict) -> dict:
    """Convert a single QA pair to ShareGPT format."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": pair["question"]},
            {"from": "gpt", "value": pair["answer"]},
        ],
        # Metadata for filtering/weighting (not used in training itself)
        "_meta": {
            "source_type": pair.get("source_type"),
            "confidence_level": pair.get("confidence_level"),
            "source_url": pair.get("source_url"),
        },
    }


def _apply_weights(examples: list[dict]) -> list[dict]:
    """Duplicate high-confidence examples based on weight config."""
    weighted = []
    for ex in examples:
        conf = ex["_meta"].get("confidence_level", "MEDIUM")
        weight = CONFIDENCE_WEIGHTS.get(conf, 1)
        for _ in range(weight):
            weighted.append(ex)
    return weighted


def run() -> list[dict]:
    """Convert all QA pairs + safety refusals to ShareGPT JSONL."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    # Load main QA pairs
    combined_path = QA_PAIRS_DIR / "_all_qa_pairs.json"
    if combined_path.exists():
        pairs = json.loads(combined_path.read_text(encoding="utf-8"))
        all_pairs.extend(pairs)
        print(f"[Step 4a] Loaded {len(pairs)} QA pairs")

    # Load safety refusals
    safety_path = QA_PAIRS_DIR / "_safety_refusals.json"
    if safety_path.exists():
        refusals = json.loads(safety_path.read_text(encoding="utf-8"))
        all_pairs.extend(refusals)
        print(f"[Step 4a] Loaded {len(refusals)} safety refusal pairs")

    # Convert to ShareGPT format
    examples = [_to_sharegpt(p) for p in all_pairs]

    # Apply confidence weighting
    examples = _apply_weights(examples)
    print(f"[Step 4a] {len(examples)} examples after confidence weighting")

    # Write intermediate (before dedup/filter)
    intermediate_path = TRAINING_DIR / "_sharegpt_all.jsonl"
    with open(intermediate_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[Step 4a] Wrote {len(examples)} examples to {intermediate_path.name}")
    return examples


if __name__ == "__main__":
    run()
