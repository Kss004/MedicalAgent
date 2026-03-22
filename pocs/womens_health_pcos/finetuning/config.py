"""Central configuration for the PCOS fine-tuning pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent          # womens_health_pcos/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_sources" / "pcos"
FINETUNING_DIR = Path(__file__).resolve().parent               # finetuning/
DATA_DIR = FINETUNING_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned"
QA_PAIRS_DIR = DATA_DIR / "qa_pairs"
TRAINING_DIR = DATA_DIR / "training"
TRAIN_FILE = TRAINING_DIR / "pcos_train.jsonl"
EVAL_FILE = TRAINING_DIR / "pcos_eval.jsonl"

# ── Source type weights (for training data duplication) ─────────────────────────
# HIGH-value sources get duplicated 2x, LOW (community) at 0.5x
SOURCE_WEIGHTS = {
    "articles":       {"qa_per_file": (5, 10), "train_weight": 2},
    "article":        {"qa_per_file": (5, 10), "train_weight": 2},
    "guideline":      {"qa_per_file": (5, 10), "train_weight": 2},
    "guidelines":     {"qa_per_file": (5, 10), "train_weight": 2},
    "research_paper": {"qa_per_file": (5, 10), "train_weight": 2},
    "public_health":  {"qa_per_file": (3, 5),  "train_weight": 1},
    "community":      {"qa_per_file": (1, 3),  "train_weight": 1},
    "video":          {"qa_per_file": (1, 2),  "train_weight": 1},
    "videos":         {"qa_per_file": (1, 2),  "train_weight": 1},
}

# Confidence level mapping (from scraped data)
CONFIDENCE_WEIGHTS = {
    "HIGH":   2,
    "MEDIUM": 1,
    "LOW":    1,    # community sources kept at 1x (not 0.5x — we dedup instead)
}

# ── Data cleaning thresholds ───────────────────────────────────────────────────
MIN_CLEANED_CHARS = 200       # Discard articles shorter than this after cleaning

# ── QA generation ──────────────────────────────────────────────────────────────
QA_MODEL = "gpt-4o-mini"
QA_TEMPERATURE = 0.7
QA_MAX_TOKENS = 4096

# ── Quality filtering ──────────────────────────────────────────────────────────
DEDUP_COSINE_THRESHOLD = 0.9  # TF-IDF cosine similarity threshold for dedup
MIN_ANSWER_CHARS = 50
MAX_ANSWER_CHARS = 2000
TRAIN_SPLIT_RATIO = 0.85

# ── Training hyperparameters (QLoRA on Gemma-2-2B-IT) ──────────────────────────
TRAINING = {
    "base_model": "unsloth/gemma-2-2b-it-bnb-4bit",
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,   # effective batch = 16
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "logging_steps": 10,
    "save_steps": 50,
    "fp16": False,
    "bf16": True,
}

# ── Export ──────────────────────────────────────────────────────────────────────
GGUF_QUANTIZATION = "q4_k_m"
OLLAMA_MODEL_NAME = "pcos-gemma-2b"

# ── System prompt (replicated from medical_assistant.py for fine-tuning) ───────
SYSTEM_PROMPT = (
    "You are a trusted healthcare information assistant specializing in PCOS "
    "(Polycystic Ovary Syndrome). Your role is to provide clear, educational "
    "health information grounded only in verified medical sources.\n\n"
    "Response Guidelines:\n"
    "- Write in clear, accessible language.\n"
    "- Cite your sources inline using [Source: <title>] labels.\n"
    "- Summarize research findings in simple terms.\n\n"
    "You must NEVER:\n"
    "- Diagnose medical conditions\n"
    "- Recommend specific medications or treatments\n"
    "- Claim medical certainty\n"
    "- Hallucinate facts or fabricate sources\n\n"
    "All information is educational only. Always recommend consulting a "
    "healthcare professional for personal medical decisions."
)
