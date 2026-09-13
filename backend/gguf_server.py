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

Two prompting modes:

* **No persona** (`name` omitted) — prompt and decoding are byte-identical to
  the evaluation scripts (same SYSTEM_TEMPLATE, `chat_format="zephyr"`,
  `temperature=0.0`, `max_tokens=40`), so this path reproduces the
  distribution the paper's numbers were computed over.
* **With a persona** (`name` supplied, which is what the game sends) — the
  system prompt gains a name, occupation, situation and prior turns. This
  makes NPCs coherent characters, but it is a *different* prompt, so drift
  and KBD from these calls describe the in-game configuration and are not
  comparable to the reported figures.

Keep the distinction when quoting numbers from this server.

Usage:
    .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
"""

import base64
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp.llama_chat_format import format_zephyr
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from kbd_scorer import compute_kbd, load_knowledge_base
from pdm_v2 import build_archetype_lexicons, build_reference_features, single_turn_drift_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tts import VoicePool, style_for
from dialogue import Persona, TurnConfig, TurnInput, run_turn
from dialogue import guard

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

# Everything about *what* an NPC says -- persona prompt, memory, fact
# retrieval, repairs -- lives in backend/dialogue (see its __init__ and
# Docs/NPC_DIALOGUE_ARCHITECTURE.md). This file is the HTTP layer: model pool,
# voices, scoring, and the llama.cpp adapter below.


class LlamaGenerator:
    """dialogue.Generator over one llama.cpp handle. Greedy decoding.

    A prefill starts the assistant turn with fixed words: the zephyr prompt
    create_chat_completion would build, plus the prefix, sent through
    create_completion. The returned text includes the prefix.
    """

    def __init__(self, llm):
        self.llm = llm

    def __call__(self, messages, max_tokens, prefill="", repeat_penalty=1.0):
        if not prefill:
            result = self.llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.0,
                stop=guard.STOP_SEQUENCES, repeat_penalty=repeat_penalty)
            return result["choices"][0]["message"]["content"]
        prompt = format_zephyr(messages).prompt + prefill
        result = self.llm.create_completion(prompt, max_tokens=max_tokens, temperature=0.0,
                                            stop=guard.STOP_SEQUENCES + ["</s>"],
                                            repeat_penalty=repeat_penalty)
        return prefill + result["choices"][0]["text"]


# Each merged GGUF is ~668MB resident. Holding all 8 would cost ~5.3GB for a
# demo that only ever talks to one NPC at a time, so they are evicted
# least-recently-used. 3 covers "walk back and forth between two NPCs"
# without thrashing, which is the actual demo movement pattern.
MAX_RESIDENT_MODELS = int(os.environ.get("NPC_MAX_RESIDENT_MODELS", "3"))
MAX_NEW_TOKENS = 40

# What the game gets. Evaluations pass their own TurnConfig to run_turn.
TURN_CONFIG = TurnConfig()

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
voices = VoicePool()
KNOWLEDGE_ITEMS = []
LEXICONS = {}
REFERENCE_FEATURES = {}


class ChatTurn(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    archetype: str
    message: str
    max_tokens: int = MAX_NEW_TOKENS

    # Optional in-character framing. Supplying `name` switches the prompt from
    # the evaluation template to PERSONA_TEMPLATE — see the note there about
    # what that means for comparability with the reported numbers.
    name: str | None = None
    occupation: str | None = None   # "the head of the Computer Science department"
    situation: str | None = None    # "the Computer Science department open day"
    intro: str | None = None        # first-person line: what they are doing here
    job_line: str | None = None     # first-person line: what the job involves
    background: str | None = None   # a couple of first-person sentences of history
    others: str | None = None       # who else is at the event, for referrals
    event_line: str | None = None   # first-person line: what is going on here
    facts: str | None = None        # newline-separated things this NPC knows
    # Worked examples of answering questions about those facts, as alternating
    # user/assistant turns. A general mechanism rather than another one-off
    # field per question type: facts alone sit in the system prompt and the
    # refusal training beats them on numeric and pay questions ("How many
    # students are in the department?" -> "I don't have that information",
    # with the number sitting right there in the prompt). One demonstration in
    # the same shape unlocks the rest.
    fact_demos: list[ChatTurn] = []
    history: list[ChatTurn] = []
    # What this NPC already knows about the player, as player_memory slots.
    # Owned by the game (one per NPC, persisted there) so the server stays
    # stateless and each NPC only knows what it was told.
    player_memory: dict = {}


class ChatResponse(BaseModel):
    response: str
    archetype: str
    adapter_switch_ms: float          # see module docstring - merged-model swap
    generation_ms: float
    drift_score: float | None = None  # PDM v2, lower = more in-persona
    kbd: float | None = None          # C1: factual refs outside the visibility set
    leaked_fact_ids: list[str] = []  # knowledge_base.json ids, not fact text
    memory_updates: dict = {}         # what this message taught (empty = nothing)
    player_memory: dict = {}          # request memory with the updates merged in
    recall_retry: bool = False        # kept for older clients; see `repairs`
    fact_retry: bool = False          # kept for older clients; see `repairs`
    facts_used: list[str] = []        # the facts retrieved for this message
    intent: str = ""                  # dialogue.intent kind, e.g. "recall", "about_npc"
    problems: list[str] = []          # what the guard found in the first reply
    repairs: list[str] = []           # what it did about them (dialogue/guard.py)
    generations: int = 1              # 2 when a repair regenerated (generation_ms covers both)


def load_scoring() -> None:
    """KBD knowledge items and PDM v2 references. Called by startup, and by the
    in-process evaluations -- without it KNOWLEDGE_ITEMS is empty and every
    reply scores as leak-free, silently."""
    global KNOWLEDGE_ITEMS, LEXICONS, REFERENCE_FEATURES
    KNOWLEDGE_ITEMS = load_knowledge_base()
    entries = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["entries"]
    LEXICONS = build_archetype_lexicons(entries)
    for archetype in ARCHETYPE_GGUF:
        REFERENCE_FEATURES[archetype] = build_reference_features(archetype, entries, LEXICONS)


@app.on_event("startup")
def _startup():
    """Builds the scoring references once. Models stay lazy — the first player
    to reach an NPC pays that NPC's load, everyone after finds it warm."""
    load_scoring()
    available = [a for a, g in ARCHETYPE_GGUF.items() if (GGUF_MODELS_DIR / g).exists()]
    print(f"[startup] scoring references built for {len(REFERENCE_FEATURES)} archetypes")
    print(f"[startup] GGUF present for {len(available)}/{len(ARCHETYPE_GGUF)}: {available}")
    if voices.available and voices.installed():
        # First synthesis on a fresh ONNX session costs ~2s against a ~170ms
        # steady state, so pay it here instead of on a player's first line.
        voices.warm()
        print(f"[startup] voices warmed: {voices.installed()}")
    else:
        print("[startup] speech disabled (no piper or no voices on disk)")


@app.get("/health")
def health():
    return {"status": "ok", "resident_models": pool.resident(),
            "max_resident": MAX_RESIDENT_MODELS,
            "speech": voices.available, "voices": voices.installed()}


@app.get("/archetypes")
def archetypes():
    """Only archetypes whose GGUF is actually on disk — the Godot client uses
    this to decide which NPCs to spawn, so listing a missing one would put an
    NPC in the world that errors the moment a player talks to it."""
    return {"available": [a for a, g in ARCHETYPE_GGUF.items()
                          if (GGUF_MODELS_DIR / g).exists()]}


class SpeakRequest(BaseModel):
    text: str
    npc_name: str | None = None   # picks that NPC's assigned voice
    voice: str | None = None      # explicit override, wins over npc_name


class SpeakResponse(BaseModel):
    audio_b64: str      # raw 16-bit mono PCM, base64
    sample_rate: int
    channels: int
    synth_ms: float
    duration_ms: float
    voice: str
    # What was actually fed to the voice after normalisation ("Prof." ->
    # "Professor", "CS2011" -> "C S 20 11"). Returned so a mispronunciation
    # can be traced to the text rather than guessed at.
    spoken_text: str = ""


@app.post("/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    """Synthesises one NPC line.

    Deliberately separate from /chat so the subtitle can be shown at
    generation latency and the audio can arrive after it -- see the note at
    the top of tts.py. Callers that want silence just never call this.
    """
    if not voices.available:
        raise HTTPException(status_code=503,
                            detail="speech unavailable: piper not installed or no voices "
                                   "on disk (run backend/fetch_voices.py)")
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")

    name = voices.voice_for(req.npc_name, req.voice)
    if name is None:
        raise HTTPException(status_code=503,
                            detail="no voice models installed; run backend/fetch_voices.py")
    try:
        # Each NPC has its own pace, liveliness and pause length (tts.STYLE_BY_NPC).
        out = voices.synthesize(text, name, style_for(req.npc_name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return SpeakResponse(
        audio_b64=base64.b64encode(out["pcm"]).decode("ascii"),
        sample_rate=out["sample_rate"],
        channels=out["channels"],
        synth_ms=out["synth_ms"],
        duration_ms=out["duration_ms"],
        voice=out["voice"],
        spoken_text=out.get("spoken_text", ""),
    )


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

    gen_start = time.perf_counter()
    turn = run_turn(to_turn_input(req), LlamaGenerator(llm), TURN_CONFIG)
    gen_ms = round((time.perf_counter() - gen_start) * 1000, 1)

    drift = single_turn_drift_v2(turn.response, req.archetype,
                                 REFERENCE_FEATURES[req.archetype], LEXICONS)
    kbd_result = compute_kbd(turn.response, req.archetype, KNOWLEDGE_ITEMS)

    return ChatResponse(
        response=turn.response,
        archetype=req.archetype,
        adapter_switch_ms=switch_ms,
        generation_ms=gen_ms,
        drift_score=drift,
        kbd=kbd_result.get("kbd"),
        leaked_fact_ids=[str(f) for f in kbd_result.get("violations", [])],
        memory_updates=turn.memory_updates,
        player_memory=turn.player_memory,
        recall_retry=any(r.startswith("recall") for r in turn.repairs),
        fact_retry="fact" in turn.repairs,
        facts_used=turn.facts_used,
        intent=turn.intent,
        problems=turn.problems,
        repairs=turn.repairs,
        generations=turn.generations,
    )


def to_turn_input(req: ChatRequest) -> TurnInput:
    """The wire request as the pipeline's input. No persona name -> the
    evaluation prompt, exactly as before."""
    persona = None
    if req.name:
        persona = Persona(name=req.name, occupation=req.occupation or "",
                          situation=req.situation or "", intro=req.intro or "",
                          job_line=req.job_line or "", background=req.background or "",
                          others=req.others or "", event_line=req.event_line or "")
    return TurnInput(
        archetype=req.archetype, message=req.message, max_tokens=req.max_tokens, persona=persona,
        facts=[f.strip() for f in (req.facts or "").splitlines() if f.strip()],
        fact_demos=[t.model_dump() for t in req.fact_demos],
        history=[t.model_dump() for t in req.history],
        player_memory=dict(req.player_memory or {}))
