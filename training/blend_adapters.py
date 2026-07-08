"""
Adapter blending (Specs.md section 7 Phase 3 / Appendix B, Contribution 2) —
interpolates two LoRA adapters to test whether the result is a coherent
hybrid persona rather than two conflicting voices.

Uses PEFT's built-in `LoraModel.add_weighted_adapter(combination_type="linear")`
rather than hand-rolling the spec's simplified `alpha*A + (1-alpha)*B` example
directly on raw tensors — PEFT's version correctly accounts for each source
adapter's own alpha/rank scaling (LoRA's effective weight update is
`(alpha/r) * B @ A`, so two adapters with different alpha need to be
normalized before a naive tensor-average would mean what you'd expect).
Requires both adapters to share the same rank (`r`) — "linear" combination
does not support blending across different ranks; that needs "cat" or "svd".

Motivation for this specific pair: `medieval_r8_gutonly` (Gutenberg-only
training, 100% archaic-dialect-dense) produces real archaic markers but
over-inserts them even where the reference expected plain prose (see
Docs/TODO.md Phase 3 — its drift score got *worse* than baseline on the
206 held-out entries with non-archaic references). `medieval_r8_a32`
(full 1003-entry mix) has broader archetype coverage and more natural
register but weak archaic-marker production. Blending tests whether a
mixture gets some of both.

Usage:
    python blend_adapters.py --adapter-a training/adapters/medieval_r8_gutonly \\
                              --adapter-b training/adapters/medieval_r8_a32 \\
                              --alpha 0.5
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTERS_DIR = Path(__file__).resolve().parent / "adapters"


def check_compatible(adapter_a: str, adapter_b: str):
    cfg_a = json.loads((Path(adapter_a) / "adapter_config.json").read_text())
    cfg_b = json.loads((Path(adapter_b) / "adapter_config.json").read_text())
    if cfg_a["r"] != cfg_b["r"]:
        raise ValueError(
            f"rank mismatch: {adapter_a} has r={cfg_a['r']}, {adapter_b} has r={cfg_b['r']} — "
            f"'linear' combination needs matching rank. Use combination_type='cat' or 'svd' instead "
            f"if you need to blend different ranks (not implemented here)."
        )
    print(f"adapter A: r={cfg_a['r']} alpha={cfg_a['lora_alpha']}")
    print(f"adapter B: r={cfg_b['r']} alpha={cfg_b['lora_alpha']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-a", required=True)
    parser.add_argument("--adapter-b", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="weight for adapter A; adapter B gets (1-alpha)")
    parser.add_argument("--output-name", default=None, help="override output dir name under training/adapters/")
    args = parser.parse_args()

    check_compatible(args.adapter_a, args.adapter_b)

    print(f"loading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16
    )

    print(f"loading adapter A: {args.adapter_a}")
    model = PeftModel.from_pretrained(base_model, args.adapter_a, adapter_name="adapter_a")
    print(f"loading adapter B: {args.adapter_b}")
    model.load_adapter(args.adapter_b, adapter_name="adapter_b")

    blend_name = "blended"
    model.base_model.add_weighted_adapter(
        adapters=["adapter_a", "adapter_b"],
        weights=[args.alpha, 1 - args.alpha],
        adapter_name=blend_name,
        combination_type="linear",
    )
    model.set_adapter(blend_name)

    a_name = Path(args.adapter_a).name
    b_name = Path(args.adapter_b).name
    output_name = args.output_name or f"blend_{a_name}_{b_name}_a{args.alpha}"
    output_dir = ADAPTERS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # save_pretrained on a multi-adapter PeftModel nests each adapter under
    # its own subfolder (output_dir/blended/...) even with selected_adapters
    # restricting which ones get saved — flatten it so this adapter loads
    # the same way as every other one in this repo (adapter_config.json /
    # adapter_model.safetensors directly in output_dir).
    model.save_pretrained(str(output_dir), selected_adapters=[blend_name])
    nested_dir = output_dir / blend_name
    for f in nested_dir.iterdir():
        f.rename(output_dir / f.name)
    nested_dir.rmdir()

    tokenizer.save_pretrained(str(output_dir))
    print(f"blended adapter saved -> {output_dir}")


if __name__ == "__main__":
    main()
