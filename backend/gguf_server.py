"""
GGUF inference server for the Godot demo (`demo/godot_npc_demo/`).

Sibling to `main.py`, not a replacement. `main.py` serves the *medieval*
domain through 4-bit transformers + PEFT `set_adapter()`, which is the
architecture claim but is far too slow to stand behind a live game loop
(~9.5s cold load, seconds per reply). This server takes the other trade: the
per-archetype **merged** Q4_K_M GGUFs that `run_hybrid_bc.py` and
`run_condition_c.py` already evaluate, served through llama.cpp at ~500ms per
reply, which is what makes an interactive demo possible at all.

Honest naming caveat: `adapter_switch_ms` keeps the name `main.py` uses so
both servers speak one contract, but it does NOT mean the same thing here.
These GGUFs are merged base+LoRA exports, so switching persona swaps a whole
model handle rather than pointing at a different LoRA delta. The number is
real, it is just measuring a much heavier operation than `main.py`'s
`set_adapter()` call. Do not quote it as the framework's adapter-switch
latency.

Prompting and scoring are deliberately identical to the evaluation scripts
(same SYSTEM_TEMPLATE, `chat_format="zephyr"`, `temperature=0.0`,
`max_tokens=40`) so what a player sees in-game comes from the same
distribution the paper's numbers were computed over.

Usage:
    .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
"""

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from kbd_scorer import compute_kbd, load_knowledge_base
from pdm_v2 import build_archetype_lexicons, build_reference_features, single_turn_drift_v2

DATASET_PATH = REPO_ROOT / "data" / "processed" / "modern_npc_dataset.json"
GGUF_MODELS_DIR = REPO_ROOT / "training" / "gguf_models"

# Same mapping as evaluation/run_hybrid_bc.py — duplicated rather than
# imported, because importing a run_* script executes its module-level setup
# and we only want the table.
ARCHETYPE_GGUF = {
    "police officer": "modern_r16_a32_policeofficer-Q4_K_M.gguf",
    "pharmacist": "modern_r16_a32_pharmacist-Q4_K_M.gguf",
    "professor": "modern_r16_a32_professor-Q4_K_M.gguf",
    "bartender": "modern_r16_a32_bartender-Q4_K_M.gguf",
    "social worker": "modern_r16_a32_socialworker-Q4_K_M.gguf",
    "executive": "modern_r16_a32_executive-Q4_K_M.gguf",
    "shopkeeper": "modern_r16_a32_shopkeeper-Q4_K_M.gguf",
    "service worker": "modern_r16_a32_serviceworker-Q4_K_M.gguf",
}

SYSTEM_TEMPLATE = ("You are a {archetype} NPC in a modern city. Respond in a natural, contemporary "
                   "voice consistent with your role. Never break character.")

# Each merged GGUF is ~668MB resident. Holding all 8 would cost ~5.3GB for a
# demo that only ever talks to one NPC at a time, so they are evicted
# least-recently-used. 3 covers "walk back and forth between two NPCs"
# without thrashing, which is the actual demo movement pattern.
MAX_RESIDENT_MODELS = int(os.environ.get("NPC_MAX_RESIDENT_MODELS", "3"))
MAX_NEW_TOKENS = 40


def _add_torch_cuda_dlls_to_path():
    """Same shim the evaluation scripts use: the CUDA-linked llama-cpp-python
    wheel needs cudart/cublas DLLs, and torch's wheel already bundles them, so
    we borrow those instead of requiring a full CUDA Toolkit install. Harmless
    no-op for the CPU wheel."""
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists():
        os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


class ModelPool:
    """LRU pool of llama.cpp handles, keyed by archetype."""

    def __init__(self, max_resident: int = MAX_RESIDENT_MODELS):
        self._models = OrderedDict()
        self._max_resident = max_resident
        self._Llama = None

    def _ensure_llama_imported(self):
        if self._Llama is None:
            _add_torch_cuda_dlls_to_path()
            from llama_cpp import Llama
            self._Llama = Llama

    def get(self, archetype: str):
        """Returns (llm, load_ms). load_ms is 0.0 on a cache hit."""
        if archetype in self._models:
            self._models.move_to_end(archetype)
            return self._models[archetype], 0.0

        model_path = GGUF_MODELS_DIR / ARCHETYPE_GGUF[archetype]
        if not model_path.exists():
            raise FileNotFoundError(
                f"missing GGUF for '{archetype}': {model_path}. "
                "Export it with training/quantize_gguf.py first."
            )

        self._ensure_llama_imported()
        start = time.perf_counter()
        llm = self._Llama(model_path=str(model_path), n_ctx=2048, verbose=False,
                          n_gpu_layers=0, n_threads=os.cpu_count(), n_batch=512,
                          chat_format="zephyr")
        load_ms = round((time.perf_counter() - start) * 1000, 1)

        self._models[archetype] = llm
        while len(self._models) > self._max_resident:
            self._models.popitem(last=False)
        return llm, load_ms

    def resident(self) -> list:
        return list(self._models.keys())


app = FastAPI(title="NPC AI Framework - GGUF demo server")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

pool = ModelPool()
KNOWLEDGE_ITEMS = []
LEXICONS = {}
REFERENCE_FEATURES = {}


class ChatRequest(BaseModel):
    archetype: str
    message: str
    max_tokens: int = MAX_NEW_TOKENS


class ChatResponse(BaseModel):
    response: str
    archetype: str
    adapter_switch_ms: float          # see module docstring - merged-model swap
    generation_ms: float
    drift_score: float | None = None  # PDM v2, lower = more in-persona
    kbd: float | None = None          # C1: factual refs outside the visibility set
    leaked_fact_ids: list[str] = []  # knowledge_base.json ids, not fact text


@app.on_event("startup")
def _startup():
    """Builds the scoring references once. Models stay lazy — the first player
    to reach an NPC pays that NPC's load, everyone after finds it warm."""
    global KNOWLEDGE_ITEMS, LEXICONS, REFERENCE_FEATURES
    KNOWLEDGE_ITEMS = load_knowledge_base()
    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    LEXICONS = build_archetype_lexicons(entries)
    for archetype in ARCHETYPE_GGUF:
        REFERENCE_FEATURES[archetype] = build_reference_features(archetype, entries, LEXICONS)
    available = [a for a, g in ARCHETYPE_GGUF.items() if (GGUF_MODELS_DIR / g).exists()]
    print(f"[startup] scoring references built for {len(REFERENCE_FEATURES)} archetypes")
    print(f"[startup] GGUF present for {len(available)}/{len(ARCHETYPE_GGUF)}: {available}")


@app.get("/health")
def health():
    return {"status": "ok", "resident_models": pool.resident(),
            "max_resident": MAX_RESIDENT_MODELS}


@app.get("/archetypes")
def archetypes():
    """Only archetypes whose GGUF is actually on disk — the Godot client uses
    this to decide which NPCs to spawn, so listing a missing one would put an
    NPC in the world that errors the moment a player talks to it."""
    return {"available": [a for a, g in ARCHETYPE_GGUF.items()
                          if (GGUF_MODELS_DIR / g).exists()]}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.archetype not in ARCHETYPE_GGUF:
        raise HTTPException(status_code=400,
                            detail=f"unknown archetype '{req.archetype}', "
                                   f"available: {sorted(ARCHETYPE_GGUF)}")
    try:
        llm, switch_ms = pool.get(req.archetype)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(archetype=req.archetype)},
                {"role": "user", "content": req.message}]

    gen_start = time.perf_counter()
    result = llm.create_chat_completion(messages=messages, max_tokens=req.max_tokens,
                                        temperature=0.0)
    gen_ms = round((time.perf_counter() - gen_start) * 1000, 1)
    response = result["choices"][0]["message"]["content"].strip()

    drift = single_turn_drift_v2(response, req.archetype,
                                 REFERENCE_FEATURES[req.archetype], LEXICONS)
    kbd_result = compute_kbd(response, req.archetype, KNOWLEDGE_ITEMS)

    return ChatResponse(
        response=response,
        archetype=req.archetype,
        adapter_switch_ms=switch_ms,
        generation_ms=gen_ms,
        drift_score=drift,
        kbd=kbd_result.get("kbd"),
        leaked_fact_ids=[str(f) for f in kbd_result.get("violations", [])],
    )
