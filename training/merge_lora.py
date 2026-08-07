"""
Merge a trained LoRA adapter into base weights (Week 1, Docs/TODO.md: GGUF
serving path). PEFT's merge_and_unload() folds the LoRA delta
((alpha/r) * B@A) directly into the base weight matrices, producing a plain
HF checkpoint indistinguishable from a full fine-tune — llama.cpp's
converter doesn't need to know LoRA was ever involved.

Loaded in bf16 on CPU (not 4-bit) for the merge itself: merging into an
already-quantized base compounds quantization error on top of the LoRA
delta. Quantization happens afterward, once, during GGUF conversion.

Usage:
    python training/merge_lora.py --adapter training/adapters/medieval_r8_gutonly
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MERGED_DIR = Path(__file__).resolve().parent / "merged_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    adapter_path = Path(args.adapter)
    out_dir = MERGED_DIR / adapter_path.name

    print(f"loading base model (bf16, CPU): {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map="cpu")

    print(f"loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    print("merging LoRA delta into base weights...")
    merged = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"merged model saved -> {out_dir}")


if __name__ == "__main__":
    main()
