#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX format and extract vocab.txt."""
import shutil
from pathlib import Path


def main():
    assets_dir = Path(__file__).parent.parent / "internal" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    model_path = assets_dir / "model.onnx"
    vocab_path = assets_dir / "vocab.txt"

    if model_path.exists() and vocab_path.exists():
        print("Assets already exist, skipping export.")
        return

    print("Downloading all-MiniLM-L6-v2 ONNX model from Hugging Face Hub...")

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Download pre-built ONNX model (single self-contained file)
    onnx_file = hf_hub_download(
        repo_id=model_name,
        filename="onnx/model.onnx",
    )

    shutil.copy(onnx_file, model_path)
    print(f"Wrote {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Extract vocab
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_path, "w") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    print(f"Wrote {vocab_path} ({len(sorted_vocab)} tokens)")


if __name__ == "__main__":
    main()
