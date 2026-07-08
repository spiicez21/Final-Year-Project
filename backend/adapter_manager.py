"""
Adapter Manager (Specs.md section 4) — loads TinyLlama-1.1B-Chat once, then
loads multiple LoRA adapters onto that single base model and switches the
*active* one via PEFT's `set_adapter()`. This is the actual "dynamic adapter
loading... no full model reload" architecture claim: the base model's 1.1B
params stay resident in VRAM the whole time; switching persona is just
pointing at a different ~1M-param LoRA delta, which is near-instant.

Only handles LoRA adapters here (Condition B), not full fine-tune checkpoints
(Condition D) — a full fine-tune is a complete standalone model and can't
share a base with anything else, so it doesn't fit this "swap the delta"
architecture at all. That's part of the real deployability tradeoff already
documented in Docs/TODO.md (2.3MB adapter vs 11.86GB full model).
"""

import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# domain -> adapter directory. "medieval" is the only trained domain so far
# (healthcare/education datasets don't exist yet — see Docs/TODO.md Phase 2).
ADAPTER_PATHS = {
    "medieval": REPO_ROOT / "training" / "adapters" / "medieval_r8_gutonly",
}

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a medieval RPG world. Respond in an archaic, "
                   "period-appropriate voice consistent with your role. Never break character.")


class AdapterManager:
    """Loads the base model once; adapters are loaded on first use and then
    kept resident, switched via set_adapter() rather than reloaded."""

    def __init__(self):
        self.tokenizer = None
        self.model = None  # PeftModel once at least one adapter is loaded
        self._loaded_adapters = set()
        self._active_adapter = None

    def _ensure_base_loaded(self):
        if self.model is not None:
            return
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16
        )

    def ensure_adapter_loaded(self, domain: str):
        if domain not in ADAPTER_PATHS:
            raise ValueError(f"unknown domain '{domain}', available: {list(ADAPTER_PATHS)}")
        if domain in self._loaded_adapters:
            return

        self._ensure_base_loaded()
        adapter_path = str(ADAPTER_PATHS[domain])
        if self.model is None:
            # first adapter: wraps the base model into a PeftModel
            self.model = PeftModel.from_pretrained(self.base_model, adapter_path, adapter_name=domain)
        else:
            # subsequent adapters: loaded onto the *same* already-wrapped model
            self.model.load_adapter(adapter_path, adapter_name=domain)
        self.model.eval()
        self._loaded_adapters.add(domain)

    def switch_to(self, domain: str):
        """Near-instant — just changes which LoRA delta is active, no reload."""
        self.ensure_adapter_loaded(domain)
        if self._active_adapter != domain:
            self.model.set_adapter(domain)
            self._active_adapter = domain

    def generate(self, domain: str, archetype: str, message: str, max_new_tokens: int = 80) -> dict:
        start = time.perf_counter()
        self.switch_to(domain)
        switch_ms = round((time.perf_counter() - start) * 1000, 1)

        system = SYSTEM_TEMPLATE.format(archetype=archetype)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": message}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
        gen_ms = round((time.perf_counter() - gen_start) * 1000, 1)

        response = self.tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        return {
            "response": response,
            "domain": domain,
            "archetype": archetype,
            "adapter_switch_ms": switch_ms,  # ~0 after first load of that domain
            "generation_ms": gen_ms,
        }

    def available_domains(self) -> list:
        return list(ADAPTER_PATHS.keys())

    def loaded_domains(self) -> list:
        return sorted(self._loaded_adapters)
