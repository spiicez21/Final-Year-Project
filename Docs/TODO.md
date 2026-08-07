# Project TODO

Tracks the **11 build-week schedule** (see `PROJECT_UPDATE_BRIEF` §6, mirrored below) against actual repo state. Update this file as tasks land — don't let it drift.

*Last rewritten: 2026-08-07, against the August 2026 scope decisions.*

## Key dates

| Date | Event |
|------|-------|
| **6 Sep 2026** | Modern-city dataset freeze — no additions after this date |
| **23 Oct 2026** | **Results freeze** — no new experiments after this date, writing only |
| **mid-Nov 2026** | Submission deadline (IEEE CoG) |

**Scope note:** healthcare and education domains, and the human-evaluation study, are **cut** (see README). Phases for them have been deleted from this file. The project reports **automatic metrics only**; the absence of a human study is a stated limitation.

---

## Work already banked (carries forward)

Complete and reusable — most is domain-agnostic and survives the modern-city pivot. Full blow-by-blow history is in git and prior revisions of this file.

**Framework / systems (domain-agnostic — carries over untouched):**
- [x] `training/train_adapter.py` — QLoRA pipeline (transformers + peft + trl SFTTrainer). Real fixes banked: bf16-end-to-end training (fp16+bf16 GradScaler crash), step-checkpointing after a thermal-TDR crash, and a **completion-only-loss bug** (loss was computed over the whole templated sequence, not just the assistant response — fixed by emitting `prompt`/`completion` columns).
- [x] `training/blend_adapters.py` — adapter interpolation via PEFT `add_weighted_adapter(combination_type="linear")` (correctly accounts for each adapter's alpha/rank scaling).
- [x] `backend/adapter_manager.py` — `AdapterManager` loads adapters onto one resident base and hot-swaps via `set_adapter()`. **Verified live:** second call with a different archetype reported `adapter_switch_ms: 0` — real proof of the no-reload claim.
- [x] `backend/main.py` — FastAPI `POST /chat`, `GET /domains`, `GET /archetypes`, `GET /health`. Model preloaded at startup.
- [x] `frontend/` — Next.js demo (domain/archetype dropdowns, live chat, live PDM bar, per-message latency). Verified end-to-end in-browser.

**Medieval control dataset + results (now the control condition, frozen):**
- [x] 1,003-pair medieval set, validated (1003/1003 schema-conformant, 0 duplicate ids). Sources: Gutenberg (Shakespeare/Chaucer/Malory, 626), chimbiwide rule-based rewrite (150), hand-authored (227).
- [x] 50-entry stress-test corpus, held out.
- [x] **Key medieval finding (reportable RQ3 result):** training-data *composition* matters more than LoRA rank/alpha/epochs for lexically-specific persona control. Eight configs on the full mix produced near-zero archaic markers; training only on the 626 dialect-dense Gutenberg entries (`medieval_r8_gutonly`) fixed it. Held-out (uncontaminated) A-vs-B: marker rate 4.7% → 14.0%, mean drift 0.9766 → 0.9396.
- [x] **Blending sweep (RQ2 precursor):** α=0.2 beat pure gutonly on both marker rate and over-insertion — a genuine sweet spot.
- [x] **Full fine-tune reference:** best raw persona (46.8% marker rate) but worst over-insertion (0.8932) and 11.86 GB on disk vs. 2.3 MB LoRA — a clean systems tradeoff.

> These medieval numbers are the **control**. The old PDM baseline mean drift of **0.9833 is saturated and reports nothing useful** — do **not** carry it forward as a headline result (see PDM v2 note in Week 5).

---

## The 11 build weeks

| Weeks | Focus | Exit condition |
|-------|-------|----------------|
| 1 (Aug 10–16) | llama.cpp/GGUF inference path. Re-measure TinyLlama latency. Qwen3-0.6B setup. | Latency < 500 ms |
| 2 (Aug 17–23) | Train one adapter (guard, 189 pairs). End-to-end generation. | Adapter generates in-character text |
| 3–4 (Aug 24–Sep 6) | Modern dataset, 600 pairs. Annotate visibility sets in the same pass. | **Frozen Sep 6** |
| 5 (Sep 7–13) | PDM v2 against real generations. | Baseline drift shows variance |
| 6 (Sep 14–20) | KBD implementation + planted forbidden facts. | KBD catches ≥1 real violation |
| 7–8 (Sep 21–Oct 4) | Train 8 modern adapters × 2 base models. | 16 adapters exist |
| 9–10 (Oct 5–18) | **α-sweep.** KBD + PDM across α for 3 adapter pairs. | This is the paper — protect these weeks |
| 11 (Oct 19–23) | Flat-RAG baseline at matched token budget. Latency/VRAM benchmarks. Figures. | Freeze |

**Slip absorbs into weeks 3–8. Never out of 9–10.**

---

### Week 1 (Aug 10–16) — Inference path + Qwen3 setup · exit: latency < 500 ms

- [ ] Merge a trained LoRA adapter into base weights once (`merge_and_unload()`, offline), export to **GGUF Q4_K_M** (llama.cpp / `llama-cpp-python`), serve via llama.cpp/Ollama. This is the `backend/inference.py` the spec always intended.
- [ ] **Re-measure TinyLlama latency** on the GGUF path. The 3486 ms figure is an unoptimized-path artifact — do not cite it as a model-size result until re-measured.
- [ ] Stand up **Qwen3-0.6B-Base** and **Qwen3-0.6B-Instruct**. ChatML template. **Set `enable_thinking=False`** (dual-mode thinking otherwise leaks reasoning traces, inflates latency, pollutes metric scoring).
- [ ] Ensure dataset formatting + any tokenizer-dependent metric code handles **both** Llama-chat and ChatML templates.

### Week 2 (Aug 17–23) — One adapter, end-to-end · exit: in-character generation

- [ ] Train one adapter (guard, 189 pairs) end-to-end on the current pipeline against at least one base model. Confirm in-character generation through the GGUF serving path.
- [ ] LoRA config for the new runs: `r=16`, `alpha=32`, target `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

### Weeks 3–4 (Aug 24 – Sep 6) — Modern dataset, 600 pairs · **exit: frozen Sep 6**

- [ ] Build the modern-city set: **600 pairs (75 × 8 roles)**, persona re-voiced from cleared sources (SGD, Taskmaster, MultiWOZ, SODA, PersonaChat, Synthetic-Persona-Chat). Record `provenance` per entry. See [`DATA_PIPELINE.md`](DATA_PIPELINE.md).
  - Load SGD/Taskmaster/MultiWOZ as **raw JSON from GitHub**, not via `load_dataset`.
  - **Do not use Cornell Movie-Dialogs** (no open licence) or **DailyDialog** (NC + share-alike). Both rejected — see `DATA_PIPELINE.md`.
- [ ] Use the 1:1 role mapping: guard→police officer, merchant→shopkeeper, scholar→professor, innkeeper→bartender, clergy→social worker, herbalist→pharmacist, noble→executive, peasant→service worker.
- [ ] **Annotate visibility sets in the same pass** — flat visibility tags per knowledge item (set intersection, not a tree/graph). Also populate `persona_features[]` per entry.
- [ ] Extend `stress_test_corpus.json` with **15–20 planted forbidden facts** (probes whose correct answer is a refusal / expression of ignorance) — required for KBD to have signal.
- [ ] **Freeze the dataset on 6 Sep 2026.**

### Week 5 (Sep 7–13) — PDM v2 · exit: baseline drift shows variance

- [ ] Rebuild PDM around **domain-agnostic feature families**: domain lexicon, register markers, formality score, syntactic profile, stance. (The old PDM used Early Modern English markers and produces no signal on modern dialogue.)
- [ ] **Calibrate against real generations from a trained adapter**, not against the dataset. The 0.9833 figure is what happens when a metric is calibrated without model output.
- [ ] **Acceptance:** baseline drift must show **variance across turns**. A flat near-1.0 score means the metric is still broken.
- [ ] Do **not** write PDM v2 before an adapter exists to calibrate against (repeats the 0.9833 mistake).

### Week 6 (Sep 14–20) — KBD · exit: catches ≥1 real violation

- [ ] Implement KBD: `KBD = (references to out-of-visibility knowledge items) / (total factual references in response)`.
- [ ] Flat visibility-tagged knowledge items; visibility resolution is a **set intersection**. Not a tree, not Neo4j, no propagation/decay.
- [ ] Measure **primarily against the stress corpus + planted forbidden facts**, not normal dialogue (normal dialogue → near-zero violations, no signal).
- [ ] **Exit:** KBD catches ≥1 real out-of-visibility violation.

### Weeks 7–8 (Sep 21 – Oct 4) — Train 16 adapters · exit: 16 adapters exist

- [ ] Train **8 modern archetype adapters × 2 base models** (TinyLlama-1.1B, Qwen3-0.6B) = 16 adapters.
- [ ] Train both **Qwen3-0.6B-Base and Qwen3-0.6B-Instruct** for ≥1 archetype; report the Base-vs-Instruct difference as an **ablation** (the Instruct "helpful assistant" prior competes with the NPC persona — itself a drift mechanism).

### Weeks 9–10 (Oct 5–18) — **α-sweep (the paper)** · protect these weeks

- [ ] For **3 adapter pairs**, sweep α ∈ {0, 0.25, 0.5, 0.75, 1.0}. Measure **KBD and PDM v2** against **both parent visibility sets**.
- [ ] Answer RQ2: does the epistemic boundary **interpolate** (each parent's facts with reduced confidence) or **collapse to the union** (a leak)?
- [ ] This is the headline result. **No slip into these weeks.**

### Week 11 (Oct 19–23) — Baselines, benchmarks, figures · exit: freeze

- [ ] **Flat-RAG baseline at matched token budget** (Condition C) — essential; without it a reviewer attributes gains to shorter prompts, not visibility structure.
- [ ] Latency / peak-RAM / adapter-storage benchmarks across conditions and both base models.
- [ ] Figures: architecture diagram, PDM v2 curve, α-sweep KBD/PDM heatmap.
- [ ] **Results freeze 23 Oct.**

---

## Evaluation conditions (reference)

| Condition | Description |
|-----------|-------------|
| A | Base model, no adapter (floor) |
| B | Base + persona adapter (primary claim) |
| C | Base + **flat-RAG at matched token budget** |
| D | GPT-4o few-shot (upper-bound reference) |

Metrics: KBD, PDM v2, BERTScore F1, adapter-routing accuracy, latency, peak RAM, adapter storage. For any rubric requiring a >95% figure, use **adapter routing accuracy** or **schema conformance** — never a PDM/KBD threshold pass-rate.

> The extensive **full-fine-tune** results already banked (previously "Condition D") no longer map to a lettered condition — D is now GPT-4o. Keep the full-fine-tune numbers as the systems/deployability tradeoff evidence for Contribution 4, but report them as an explicit ablation, not as Condition D.

---

## Paper (writing-only after 23 Oct)

- [ ] Results section first, then Method, then Related Work (from [`RELATED_WORK.md`](RELATED_WORK.md)), then Intro + Abstract last.
- [ ] Retrieve and assess **IEEE arnumber 11419836** before the draft (currently inaccessible — see `RELATED_WORK.md`).
- [ ] Submit to IEEE CoG; upload ArXiv preprint the same week.

---

## Reconciliation notes (conflicts flagged, not silently overwritten)

- **Crime-city → modern-city.** The 2026-08-06 pivot targeted a GTA-inspired *crime-city* setting with a street-slang PDM lexicon. The August 2026 decision is a **general modern city** with a different 1:1 mapping (see `DATA_PIPELINE.md`). Code still encoding the crime-city direction — `MODERN_ARCHETYPES` in `training/train_adapter.py`, the modern branch of `data/scripts/chimbiwide_converter.py`, `DIALECT_PATTERNS_MODERN` in `evaluation/pdm_scorer.py` — **predates and conflicts with** the current decision and needs reconciling before the modern adapters are trained. Not changed in this doc pass (code, not docs).
- **Cornell Movie-Dialogs** was previously listed as the next modern source; it is now **rejected** (no open licence). The DATA_PIPELINE task list reflects the rejection.
- **PDM lexicon in code is still domain-specific.** PDM v2 (domain-agnostic feature families) is new work for Week 5; the existing `pdm_scorer.py` is not it.

---

## Known issues / environment notes

- System Python 3.13 is broken on this machine (`0x80070003`). Scripts run via the Python 3.10 install. `Lib/site-packages` was found **empty** on 2026-08-06 — reinstall from `requirements.txt` before running anything locally.
- Local GPU: NVIDIA MX450, **2.15 GB VRAM** — tight even for 4-bit QLoRA on TinyLlama 1.1B, and cannot sustain multi-hour runs (thermal/power throttling to a 5 W cap observed). **Budget Colab/cloud GPU** for real training. Full fine-tune (~8–12 GB optimizer state) cannot run locally at all.
- **Resume-from-checkpoint is broken below torch 2.6** (CVE-2025-32434 blocks `torch.load` of optimizer state). `train_adapter.py` skips resume below 2.6 and restarts fresh. Not upgrading torch (risks destabilizing the working bitsandbytes/peft/trl stack).
- `ollama serve` is unreliable under sustained load (`WinError 10013`, socket exhaustion). Eval/serving scripts use a persistent session, retry-with-backoff, and per-entry incremental save (`--resume` skips written ids). Expect the same flakiness on the GGUF serving path.
