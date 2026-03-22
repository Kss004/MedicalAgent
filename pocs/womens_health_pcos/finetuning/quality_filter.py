"""Step 4b: Deduplicate, filter, and split training data.

- Deduplicates similar questions via TF-IDF cosine similarity
- Filters by answer length and required content (citation or disclaimer)
- Produces 85/15 train/eval split
"""

import json
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    TRAINING_DIR,
    TRAIN_FILE,
    EVAL_FILE,
    DEDUP_COSINE_THRESHOLD,
    MIN_ANSWER_CHARS,
    MAX_ANSWER_CHARS,
    TRAIN_SPLIT_RATIO,
)


def _load_sharegpt(path: Path) -> list[dict]:
    """Load JSONL file of ShareGPT examples."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _extract_question(ex: dict) -> str:
    """Extract the human question from a ShareGPT example."""
    for turn in ex["conversations"]:
        if turn["from"] == "human":
            return turn["value"]
    return ""


def _extract_answer(ex: dict) -> str:
    """Extract the assistant answer from a ShareGPT example."""
    for turn in ex["conversations"]:
        if turn["from"] == "gpt":
            return turn["value"]
    return ""


def deduplicate(examples: list[dict]) -> list[dict]:
    """Remove near-duplicate questions using TF-IDF cosine similarity."""
    if len(examples) <= 1:
        return examples

    questions = [_extract_question(ex) for ex in examples]

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(questions)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Greedy dedup: keep first occurrence, drop later duplicates
    keep = set(range(len(examples)))
    for i in range(len(examples)):
        if i not in keep:
            continue
        for j in range(i + 1, len(examples)):
            if j not in keep:
                continue
            if sim_matrix[i, j] > DEDUP_COSINE_THRESHOLD:
                keep.discard(j)

    deduped = [examples[i] for i in sorted(keep)]
    removed = len(examples) - len(deduped)
    if removed:
        print(f"  Dedup removed {removed} near-duplicate questions")
    return deduped


def filter_quality(examples: list[dict]) -> list[dict]:
    """Filter examples by answer quality criteria."""
    passed = []
    reasons: dict[str, int] = {}

    for ex in examples:
        answer = _extract_answer(ex)

        # Check answer length
        if len(answer) < MIN_ANSWER_CHARS:
            reasons["too_short"] = reasons.get("too_short", 0) + 1
            continue
        if len(answer) > MAX_ANSWER_CHARS:
            reasons["too_long"] = reasons.get("too_long", 0) + 1
            continue

        # Must contain citation OR healthcare disclaimer (safety refusals have disclaimer)
        has_citation = "[Source:" in answer or "[Source " in answer
        has_disclaimer = any(phrase in answer.lower() for phrase in [
            "healthcare provider",
            "healthcare professional",
            "consult your doctor",
            "medical professional",
            "consult a doctor",
        ])

        if not has_citation and not has_disclaimer:
            reasons["no_citation_or_disclaimer"] = reasons.get("no_citation_or_disclaimer", 0) + 1
            continue

        passed.append(ex)

    if reasons:
        print(f"  Quality filter removed: {reasons}")
    return passed


def split_train_eval(examples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into train and eval sets (stratified by source_type if possible)."""
    random.shuffle(examples)
    split_idx = int(len(examples) * TRAIN_SPLIT_RATIO)
    return examples[:split_idx], examples[split_idx:]


def _write_jsonl(examples: list[dict], path: Path):
    """Write examples to JSONL, stripping internal _meta field."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # Remove internal metadata before writing final training data
            clean = {"conversations": ex["conversations"]}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def run():
    """Run dedup, filter, and split pipeline."""
    input_path = TRAINING_DIR / "_sharegpt_all.jsonl"
    if not input_path.exists():
        print("[Step 4b] No input found. Run format_converter.py first.")
        return

    examples = _load_sharegpt(input_path)
    print(f"[Step 4b] Loaded {len(examples)} examples")

    # Deduplicate
    examples = deduplicate(examples)
    print(f"  After dedup: {len(examples)}")

    # Quality filter
    examples = filter_quality(examples)
    print(f"  After quality filter: {len(examples)}")

    # Split
    train, eval_set = split_train_eval(examples)
    print(f"  Train: {len(train)}, Eval: {len(eval_set)}")

    # Write final files
    _write_jsonl(train, TRAIN_FILE)
    _write_jsonl(eval_set, EVAL_FILE)
    print(f"[Step 4b] Wrote {TRAIN_FILE.name} and {EVAL_FILE.name}")


if __name__ == "__main__":
    run()
