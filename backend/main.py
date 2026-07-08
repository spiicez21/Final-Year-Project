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
from pdm_scorer import extract_features, single_turn_drift

from adapter_manager import AdapterManager

app = FastAPI(title="NPC AI Framework")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

manager = AdapterManager()


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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        result = manager.generate(req.domain, req.archetype, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    features = extract_features(result["response"])
    drift = None
    if req.reference_features is not None:
        drift = single_turn_drift(result["response"], set(req.reference_features))

    return ChatResponse(**result, features=sorted(features), drift_score=drift)


@app.get("/health")
def health():
    return {"status": "ok", "loaded_adapters": manager.loaded_domains()}
