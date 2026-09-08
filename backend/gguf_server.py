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

import json
import os
import re
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

# In-character prompt used when the caller supplies a persona (the game does;
# the evaluation scripts do not).
#
# This is DELIBERATELY not the template above. SYSTEM_TEMPLATE is kept
# byte-identical to the one in evaluation/run_*.py so a /chat call with no
# persona reproduces the paper's prompt exactly. The moment a persona is
# supplied we are no longer sampling that distribution, so drift/KBD numbers
# from persona calls are NOT comparable to the reported results — they
# describe the in-game configuration instead. Both paths are kept so the
# comparison stays available rather than being quietly replaced.
#
# Kept short on purpose — see the note on _priming_turns() below for why the
# persona is established by demonstration rather than by a longer instruction
# block. Residual assistant-prior leakage ("Sure, here's a sample response:")
# is stripped by _clean_reply(); that leakage is itself an instance of the
# Instruct-prior-vs-persona competition the project documents as a drift
# mechanism.
PERSONA_TEMPLATE = (
    "You are {name}, a {archetype} — {occupation}. You are at {situation}.{background}{others}"
    "{facts} "
    "Reply in one or two short spoken sentences. Never break character."
)

# Optional clauses. Kept as separate fragments so an NPC with no background,
# or an event with nobody else at it, produces a prompt with no dangling
# sentence rather than an empty gap.
BACKGROUND_CLAUSE = " {background}"
OTHERS_CLAUSE = " Also here today: {others}."

# Game-world facts (department size, timetable, the NPC's own pay and
# workload). These come from the game, not from knowledge_base.json — see the
# header of Systems/NPC/campus_facts.gd for why the two must stay separate.
#
# Put after the persona and before the instruction so the last thing in the
# system message is still "reply in one or two sentences", which is the part
# the model most needs to keep hold of.
FACTS_CLAUSE = "\nThings you know:\n{facts}\n"

## Why the persona is taught by example rather than by instruction.
##
## The obvious approach — a long system prompt listing rules ("always answer
## questions about who you are", "never mention being an AI") — was tried and
## measurably backfired on this model. TinyLlama-1.1B answered "Who are you?"
## with "I'm not allowed to tell you that." and "What is your job like?" with
## "I don't have a job, I'm just a machine.": a flat persona break produced by
## the very prompt meant to prevent one.
##
## Two reasons. The adapters were fine-tuned against the short SYSTEM_TEMPLATE
## above, so a long structured instruction block is out-of-distribution for
## them; and a 1.1B model follows demonstrations far more reliably than
## negative rules, which mostly serve to put the forbidden words in context.
##
## So the system prompt stays short and close to the training distribution,
## and identity is established by seeding the conversation with two turns the
## NPC has already answered correctly. The model continues a pattern it can
## see instead of obeying rules it cannot hold.
def _priming_turns(name: str, occupation: str, intro: str, job_line: str,
                   background: str = "", event_line: str = "",
                   situation: str = "") -> list:
    turns = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": f"I'm {name}, {occupation}."},
        {"role": "user", "content": "What are you doing here today?"},
        {"role": "assistant", "content": intro},
    ]
    # The third demonstration is not decorative. Several adapters were trained
    # with hand-authored refusal examples, and without a worked example of a
    # *permitted* question about their own work they generalise the refusal to
    # it — the executive answered "What is your job actually like?" with "I'm
    # not allowed to talk about my job." Showing one answered job question
    # scopes the refusal back to genuinely out-of-bounds topics.
    if job_line:
        turns += [
            {"role": "user", "content": "What is your job actually like?"},
            {"role": "assistant", "content": job_line},
        ]
        # A second demonstration in a casual, unpunctuated register. One
        # formal example was not enough: the executive answered "What is your
        # job actually like?" fine but still refused "whats your work like"
        # and "r job like" (2 of 6 phrasings tested). The refusal training in
        # some adapters keys partly on register, so the demonstrations have to
        # cover more than one. The short answer is the first sentence of
        # job_line, so the two examples agree on the facts while differing in
        # length and formality.
        first_sentence = re.split(r"(?<=[.!?])\s+", job_line.strip())[0]
        turns += [
            {"role": "user", "content": "whats your work like"},
            {"role": "assistant", "content": first_sentence},
        ]
    # Background as a demonstration too, not only as a line in the system
    # prompt. Stated in the system prompt alone it was mostly ignored: the
    # professor whose background says "came back to teach after four years in
    # industry" answered "Have you always worked here?" with "about a year",
    # and "Did you work outside academia?" with "No, I'm not allowed to work
    # outside academia" — flatly contradicting it. Same lesson as the identity
    # turns: on this model, shown beats told.
    if background:
        turns += [
            {"role": "user", "content": "How did you end up here?"},
            {"role": "assistant", "content": background},
        ]
    # "What is happening here?" is about the *event*, not the person, and
    # without a demonstration for it the adapter answers from its own training
    # topic instead: the police officer replied "we have received reports of a
    # group of people breaking into the computer science department", inventing
    # an incident at what is supposed to be an open day. (KBD scored null on
    # that reply — it matched no knowledge_base fact, so it was a hallucination
    # shaped by the archetype's topic prior, not a visibility-set leak.)
    if event_line:
        turns += [
            {"role": "user", "content": "what is happening here"},
            {"role": "assistant", "content": event_line},
        ]
        # Second phrasing, asking for the *reason* rather than the scene. One
        # demonstration covered "what is happening here" but left "Why is
        # everyone here?" answered with "I don't know" — and the host of the
        # event answering "I've never seen this before" is worse than a bland
        # line. Same register-coverage lesson as the job questions.
        if situation:
            turns += [
                {"role": "user", "content": "why is everyone here"},
                {"role": "assistant", "content": f"Everyone's here for {situation}."},
            ]
    return turns

# Cut generation at the point the model starts writing the player's next line
# or drifting into document formatting.
STOP_SEQUENCES = ["\nStudent:", "\nYou:", "\nyou:", "Student:", "###", "\n\n\n"]

# Assistant-prior boilerplate that survives the prompt instructions. Matched at
# the START of a reply only, so an NPC can still legitimately use these words
# mid-sentence.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|okay|ok|here)\b[^\n:]{0,60}:\s*",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"[.!?…][\"')\]]*\s*$")


def _clean_reply(text: str) -> str:
    """Strips assistant-prior artefacts and mid-sentence truncation.

    Generation is capped at a low token count to keep latency inside the
    real-time budget, so replies regularly stop mid-clause ("We've been
    working hard to improve the app and"). Rather than raise the cap for every
    turn, the dangling tail is dropped back to the last completed sentence —
    the cost is a shorter line, not a longer wait.
    """
    cleaned = text.strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned, count=1).strip()

    # Models often wrap a persona line in quotes; drop them only when they
    # enclose the whole reply, so quoted speech inside a line survives.
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()

    if not _SENTENCE_END_RE.search(cleaned):
        # Truncated mid-sentence: keep everything up to the last terminator.
        cut = max(cleaned.rfind(c) for c in ".!?…")
        if cut > 0:
            cleaned = cleaned[:cut + 1]

    # Never hand back an empty string: a short odd reply is more debuggable
    # than an NPC that appears to say nothing at all.
    return cleaned.strip() or text.strip()


def _select_history(history: list) -> list:
    """Picks which prior turns to replay when the conversation outruns the cap.

    Plain truncation to the last N drops the opening exchange, which is
    usually the one that established what the conversation is *about* — ask
    four questions about someone's research and the fifth answer has forgotten
    the topic. So the first exchange is always kept and the window slides over
    the rest, which costs one exchange of recency for a stable subject.
    """
    if len(history) <= MAX_HISTORY_TURNS:
        return history
    return history[:2] + history[-(MAX_HISTORY_TURNS - 2):]


def _build_messages(req: "ChatRequest") -> list:
    """Eval-identical prompt by default; in-character prompt with a persona."""
    if not req.name:
        messages = [{"role": "system",
                     "content": SYSTEM_TEMPLATE.format(archetype=req.archetype)}]
        messages.append({"role": "user", "content": req.message})
        return messages

    occupation = req.occupation or f"a {req.archetype}"
    system = PERSONA_TEMPLATE.format(
        name=req.name,
        archetype=req.archetype,
        occupation=occupation,
        situation=req.situation or "a college department open day",
        background=BACKGROUND_CLAUSE.format(background=req.background.strip())
                   if req.background else "",
        # Naming the other guests is what lets an NPC hand a question on
        # ("you'd want to ask Ms. Okafor about that") instead of either
        # inventing an answer or dead-ending the player.
        others=OTHERS_CLAUSE.format(others=req.others.strip()) if req.others else "",
        facts=FACTS_CLAUSE.format(facts=req.facts.strip()) if req.facts else "",
    )
    messages = [{"role": "system", "content": system}]
    messages += _priming_turns(
        req.name, occupation,
        req.intro or "I'm here for the open day, meeting students.",
        req.job_line or "",
        req.background or "",
        req.event_line or "",
        req.situation or "",
    )
    for turn in req.fact_demos:
        if turn.role in ("user", "assistant"):
            messages.append({"role": turn.role, "content": turn.content})
    # Prior turns are what let an NPC answer "and how long have you done that?"
    # — without them every question is heard in isolation.
    for turn in _select_history(req.history):
        if turn.role in ("user", "assistant"):
            messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": req.message})
    return messages

# Each merged GGUF is ~668MB resident. Holding all 8 would cost ~5.3GB for a
# demo that only ever talks to one NPC at a time, so they are evicted
# least-recently-used. 3 covers "walk back and forth between two NPCs"
# without thrashing, which is the actual demo movement pattern.
MAX_RESIDENT_MODELS = int(os.environ.get("NPC_MAX_RESIDENT_MODELS", "3"))
MAX_NEW_TOKENS = 40

# Turns of prior dialogue replayed into the prompt. Four exchanges, not two:
# two was enough for an immediate follow-up but lost the subject of the
# conversation as soon as you asked a third question, so an NPC would forget
# what you were talking about halfway through.
MAX_HISTORY_TURNS = 8


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

    messages = _build_messages(req)

    gen_start = time.perf_counter()
    result = llm.create_chat_completion(messages=messages, max_tokens=req.max_tokens,
                                        temperature=0.0, stop=STOP_SEQUENCES)
    gen_ms = round((time.perf_counter() - gen_start) * 1000, 1)
    response = _clean_reply(result["choices"][0]["message"]["content"])

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
