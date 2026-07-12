"""
FastAPI app (Specs.md section 4/11). Thin HTTP wrapper around AdapterManager
— the actual "framework" contribution is the adapter-switching + PDM scoring,
this just exposes it over a network boundary for the demo frontend.
"""

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from pdm_scorer import extract_features, single_turn_drift, DIALECT_PATTERNS

from adapter_manager import AdapterManager

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "processed" / "medieval_npc_dataset.json"

app = FastAPI(title="NPC AI Framework")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

manager = AdapterManager()


def _build_archetype_references() -> dict:
    """Per-archetype reference dialect-feature set for live PDM scoring —
    same aggregation `run_stress_test.py` uses, filtered to pdm_scorer's
    actual detectable vocabulary so drift scores are internally consistent."""
    import json
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    valid = set(DIALECT_PATTERNS.keys())
    refs = {}
    for e in data["entries"]:
        arch = e["persona"]["archetype"]
        refs.setdefault(arch, set()).update(e["linguistic_markers"]["dialect_features"])
    return {arch: sorted(feats & valid) for arch, feats in refs.items()}


ARCHETYPE_REFERENCES = _build_archetype_references()


class ChatRequest(BaseModel):
    domain: str = "medieval"
    archetype: str = "peasant"
    message: str
    reference_features: list[str] | None = None  # optional, enables drift scoring


class ChatResponse(BaseModel):
    response: str
    domain: str
    archetype: str
    adapter_switch_ms: float
    generation_ms: float
    features: list[str]
    drift_score: float | None = None


@app.get("/domains")
def list_domains():
    return {"available": manager.available_domains(), "loaded": manager.loaded_domains()}


@app.get("/archetypes")
def list_archetypes():
    return {arch: feats for arch, feats in sorted(ARCHETYPE_REFERENCES.items())}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = manager.generate(req.domain, req.archetype, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    features = extract_features(result["response"])
    # Auto-scores drift against the archetype's known reference vocabulary
    # unless the caller overrides it — this is what makes PDM a "live score"
    # in the UI rather than something the frontend has to compute itself.
    reference = req.reference_features if req.reference_features is not None \
        else ARCHETYPE_REFERENCES.get(req.archetype, [])
    drift = single_turn_drift(result["response"], set(reference)) if reference else None

    return ChatResponse(**result, features=sorted(features), drift_score=drift)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_adapters": manager.loaded_domains()}
