"""Step 7: Export fine-tuned model to GGUF and register with Ollama.

Converts the QLoRA-trained model to GGUF format (q4_k_m quantization)
and creates an Ollama Modelfile for local serving.
"""

import subprocess
from pathlib import Path

from config import GGUF_QUANTIZATION, OLLAMA_MODEL_NAME, SYSTEM_PROMPT


def export_to_gguf():
    """Export the Unsloth model to GGUF format."""
    from unsloth import FastLanguageModel

    model_dir = Path(__file__).parent / "model_final"
    gguf_dir = Path(__file__).parent / "model_gguf"
    gguf_dir.mkdir(exist_ok=True)

    print(f"[Export] Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=2048,
        load_in_4bit=True,
    )

    print(f"[Export] Saving as GGUF ({GGUF_QUANTIZATION})...")
    model.save_pretrained_gguf(
        str(gguf_dir),
        tokenizer,
        quantization_method=GGUF_QUANTIZATION,
    )

    # Find the generated GGUF file
    gguf_files = list(gguf_dir.glob("*.gguf"))
    if gguf_files:
        print(f"[Export] GGUF saved: {gguf_files[0]}")
        return gguf_files[0]
    else:
        print("[Export] Warning: No GGUF file found after export")
        return None


def create_ollama_modelfile(gguf_path: Path) -> Path:
    """Create an Ollama Modelfile for the exported model."""
    modelfile_path = Path(__file__).parent / "Modelfile"

    # Escape the system prompt for the Modelfile
    escaped_prompt = SYSTEM_PROMPT.replace('"', '\\"')

    content = f"""FROM {gguf_path}

PARAMETER temperature 0
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER stop "<end_of_turn>"

SYSTEM \"\"\"{escaped_prompt}\"\"\"
"""

    modelfile_path.write_text(content, encoding="utf-8")
    print(f"[Export] Modelfile created: {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: Path):
    """Register the model with Ollama."""
    print(f"[Export] Registering model as '{OLLAMA_MODEL_NAME}' with Ollama...")
    try:
        result = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"[Export] Model '{OLLAMA_MODEL_NAME}' registered successfully!")
            print(f"  Test with: ollama run {OLLAMA_MODEL_NAME}")
        else:
            print(f"[Export] Ollama registration failed: {result.stderr}")
    except FileNotFoundError:
        print("[Export] Ollama not found. Install it from https://ollama.com")
        print(f"  Then run: ollama create {OLLAMA_MODEL_NAME} -f {modelfile_path}")
    except subprocess.TimeoutExpired:
        print("[Export] Ollama registration timed out")


def run():
    """Full export pipeline: model → GGUF → Ollama."""
    gguf_path = export_to_gguf()
    if gguf_path is None:
        return

    modelfile_path = create_ollama_modelfile(gguf_path)
    register_with_ollama(modelfile_path)


if __name__ == "__main__":
    run()
