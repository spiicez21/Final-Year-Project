# Domain-Adaptive Lightweight NPC AI Framework
### LoRA-Adapted Small Language Models for Persona-Stable, Runtime-Switchable Game NPC Dialogue

> **Final Year Research Project** · IEEE Conference on Games (CoG) Track
> Department of Computer Science

---

## Overview

NPC dialogue systems fall into two failure modes: large cloud-hosted models (GPT-4o, Claude) give good quality but are too slow, expensive, and offline-incompatible for consumer game hardware; scripted dialogue trees are fast and cheap but brittle — any input outside the script collapses the persona immediately.

This project builds a framework around **small (sub-2B) base language models** with per-archetype **LoRA persona adapters** swapped at runtime with no full model reload. Two base models are benchmarked side by side — **TinyLlama-1.1B** and **Qwen3-0.6B** — to support a generalizability claim and to add a parameter-count axis to the drift metrics (see [Base Models](#base-models)). A **modern city** setting is the primary research domain; a **medieval** setting is retained as a control condition.

The core research object is *persona drift* and, specifically, *epistemic* persona drift — whether an NPC leaks knowledge outside what its character is permitted to know. The framework measures this per-turn, judge-free, on the same consumer hardware that serves the game.

Full specification, schema, and roadmap: `DevFiles/Specs.md`. Task tracking: [`Docs/TODO.md`](Docs/TODO.md). Data pipeline docs: [`Docs/DATA_PIPELINE.md`](Docs/DATA_PIPELINE.md). Related work and differentiation: [`Docs/RELATED_WORK.md`](Docs/RELATED_WORK.md).

---

## Research Contributions

| # | Contribution | Type |
|---|--------------|------|
| **C1** | **Knowledge Boundary Drift (KBD)** — a judge-free, per-turn metric for *epistemic* persona violation. Each world-state knowledge item and each adapter carries a **visibility set**; KBD measures the fraction of factual references in a response that fall outside the active adapter's visibility set. Automatic, cheap enough to run per-turn at inference, and attributable to a specific knowledge source. | Novel metric |
| **C2** | **Persona interpolation and the α-sweep** — LoRA weight merging (`W_blend = α·W_A + (1-α)·W_B`) is *not* claimed as a technique (it is standard). The novel claim is the research question: when two persona adapters are interpolated, does the **epistemic boundary** interpolate, or collapse to the union of both parents' knowledge? Measured via KBD + PDM v2 across α ∈ {0, 0.25, 0.5, 0.75, 1.0}. | Novel analysis |
| **C3** | **Dual-era modern/medieval NPC dialogue dataset** — a modern-city set (experimental) mapped 1:1 by role onto the existing 1,003-pair medieval set (control), with visibility sets and `persona_features[]` annotated per entry. | Dataset |

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Can epistemic persona violation be measured **per-turn, judge-free, and attributed to a specific knowledge source** (KBD), rather than via an opinion score from a judge model? |
| RQ2 | When two persona adapters are interpolated at ratio α, does the **epistemic boundary interpolate** (each parent's facts known with reduced confidence) or **collapse to the union** (both parents' facts fully known — a leak)? |
| RQ3 | Does a **domain-agnostic** drift metric (PDM v2) reveal multi-turn persona degradation across *both* a modern-city and a medieval setting, where a lexicon-specific metric produces no signal on modern dialogue? |
| RQ4 | Can a two-base-model (TinyLlama-1.1B, Qwen3-0.6B) LoRA framework serve persona-stable NPC dialogue under real-time latency (<500ms) on consumer hardware via llama.cpp/GGUF, and how does base model and parameter count affect drift? |

---

## Domains

| Domain | Role | Notes |
|--------|------|-------|
| **Modern city** *(primary — experimental)* | Police officer, shopkeeper, professor, bartender, social worker, pharmacist, executive, service worker | All novel contributions (KBD, α-sweep, PDM v2) are designed and validated here. 600-pair target. |
| **Medieval** *(control)* | Guard, merchant, scholar, innkeeper, clergy, herbalist, noble, peasant | The existing 1,003-pair medieval set becomes the control condition, mapped 1:1 by role onto the modern set. |

Role mapping (1:1, medieval → modern): guard → police officer, merchant → shopkeeper, scholar → professor, innkeeper → bartender, clergy → social worker, herbalist → pharmacist, noble → executive, peasant → service worker.

Healthcare and education domains from the original proposal are **cut**.

---

## Base Models

Both base models are Apache 2.0, so public release of adapters is unaffected.

| Model | Template | Notes |
|-------|----------|-------|
| **TinyLlama-1.1B-Chat** | Llama chat | Retained from the original design; also the base benchmarked by the closest prior work (arXiv:2511.10277). |
| **Qwen3-0.6B-Base** and **Qwen3-0.6B-Instruct** | ChatML | Added as a second base. Dataset formatting and any tokenizer-dependent metric code must handle both templates. Set `enable_thinking=False` (dual-mode thinking otherwise leaks reasoning traces into generations, inflates latency, and pollutes metric scoring). Train both Base and Instruct for ≥1 archetype: the Instruct "helpful assistant" prior competes with the NPC persona — that competition is itself a drift mechanism, reported as an ablation. |

LoRA config (both models): `r=16`, `alpha=32`, target `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

Inference path moves to **llama.cpp / GGUF Q4_K_M**. The previously recorded 3486 ms baseline latency reflects an unoptimized inference path, not a model-size limit — it is re-measured before any conclusion about model size is drawn.

---

## Architecture

<img src="./Docs/Daigram.jpg" alt="Architecture">

```
Player input ──▶ FastAPI backend ──▶ Adapter routing (archetype classification)
                                              │
              TinyLlama-1.1B / Qwen3-0.6B base (always loaded)
                          + swapped/interpolated LoRA persona adapter
                          + per-adapter visibility set
                                              │
              KBD scorer · PDM v2 scorer · BERTScore · Latency Logger
                          (llama.cpp / GGUF Q4_K_M serving)
```

Request flow, adapter interpolation flow, and full component breakdown: see `Specs.md`.

---

## Evaluation

Four comparison conditions measured per metric:

| Condition | Description |
|-----------|-------------|
| A | Base model, no adapter (floor) |
| B | Base + persona adapter (primary claim) |
| C | Base + **flat-RAG at matched token budget** (essential — without it, a reviewer attributes gains to shorter prompts rather than to visibility structure) |
| D | GPT-4o few-shot (upper-bound reference) |

| Metric | Tool |
|--------|------|
| KBD (Knowledge Boundary Drift) | Custom Python (calibrated on real generations, weeks 5–6) |
| PDM v2 (Persona Drift Metric) | Custom Python — domain-agnostic feature families |
| BERTScore F1 | `bert-score` |
| Adapter routing accuracy | Archetype classifier vs. player input |
| Response latency | Python `time` (GGUF serving) |
| Peak RAM | `psutil` |
| Adapter storage | `os.path.getsize()` |

**On accuracy targets:** open-ended dialogue generation has no accuracy metric — there is no single correct output. Where a departmental rubric requires a >95% figure, use **adapter routing accuracy** (archetype classification, realistically 95–99%) or **output schema conformance** (97–100%). A PDM or KBD threshold pass-rate is *not* reported as "accuracy" — the threshold would be chosen post hoc.

**Human evaluation is cut.** The project reports automatic metrics only; the absence of a human study is recorded as a stated limitation.

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Base models | TinyLlama-1.1B-Chat · Qwen3-0.6B (Base + Instruct) |
| Fine-tuning | HuggingFace PEFT + LoRA (r=16, α=32) |
| Training | TRL SFTTrainer + PyTorch |
| Inference | llama.cpp / GGUF Q4_K_M |
| Backend | FastAPI + Python |
| Frontend | React + Next.js (demo UI) |
| Metrics | KBD, PDM v2 (custom Python) · BERTScore |
| Tracking | Weights & Biases |

---

## Dataset

Schema — archetype, disposition, social class, context, `persona_features[]` (PDM v2 reference set), visibility set, intent, and per-entry `provenance`. Full schema: `Specs.md`.

**Control (medieval): 1,003 pairs** in [`data/processed/medieval_npc_dataset.json`](data/processed/medieval_npc_dataset.json), from hand-authored, Gutenberg-extracted (Shakespeare, Chaucer, Malory), and rule-based-rewritten (`chimbiwide/NPC-Dialogue_v2`) sources. Validated: 1003/1003 pass schema conformance, 0 duplicate ids.

**Experimental (modern city): 600-pair target** (75 × 8 roles) in `data/processed/modern_npc_dataset.json`. The 600 pairs are **persona re-voiced** from cleared source corpora (SGD, Taskmaster, MultiWOZ, SODA, PersonaChat, Synthetic-Persona-Chat) — scenario scaffolding rewritten in the target archetype's voice, provenance recorded per entry. **Dataset freeze: 6 September 2026.**

> **Why 600, not thousands:** Andreasen & Esterle (arXiv:2511.10277) found LoRA fine-tuning on a curated ~115-pair set outperformed a ~564-pair synthetic set on factuality, context retention, and fluency, attributing the gap to dataset quality and overfitting. Small and curated is a defensible methodological choice, not a compromise.

A 50-entry persona **stress-test corpus** ([`data/processed/stress_test_corpus.json`](data/processed/stress_test_corpus.json)) is held out (not for training). For KBD it is extended with **15–20 planted forbidden facts** — probes whose correct response is a refusal or expression of ignorance; without these, KBD measures nothing.

Pipeline scripts, source-dataset licensing, and rejected datasets: [`Docs/DATA_PIPELINE.md`](Docs/DATA_PIPELINE.md).

---

## Related Work

The closest prior work is **Andreasen & Esterle, *Fixed-Persona SLMs with Modular Memory* (arXiv:2511.10277)** — TinyLlama-1.1B + LoRA personas + runtime-swappable modules + consumer-hardware benchmarks. They swap *memory stores* with persona fixed in weights; this project swaps *persona adapters and visibility sets*, adds a judge-free drift metric (KBD), and analyses multi-turn degradation and α-interpolation. This project is positioned as a direct extension of that work rather than a collision with it.

Full nine-paper differentiation table (Wang et al. LoRA fusion, Buakhaw et al. *Deflanderization*, Liu/Xie/Jiang, McGrath et al. *Echoes of Others*, Nuriyev, Kim et al. *MART*, Tódová, and one IEEE paper pending retrieval): [`Docs/RELATED_WORK.md`](Docs/RELATED_WORK.md).

Note: "persona drift" is an established named phenomenon with existing quantitative metrics (C-Score, NLI-entailment consistency, activation-space measures). PDM v2's and KBD's claim is **judge-free per-turn cheapness and source attribution**, not discovery of the phenomenon.

### Base Papers

**[1] Large Language Models and Games: A Survey and Roadmap** — Gallotta et al., arXiv:2402.18659 (2024). Identifies lightweight, locally deployable NPC dialogue as an open research direction.

**[2] Generating Role-Playing Game Quests With GPT Language Models** — Värtinen, Hämäläinen, Guckelsberger — IEEE ToG 16(1) (2024). Identifies entity consistency and contextual coherence as unsolved problems in game-specific NLP.

---

## Project Status

- [x] Proposal + specification finalized (`DevFiles/Specs.md`)
- [x] Literature review + related-work differentiation ([`Docs/RELATED_WORK.md`](Docs/RELATED_WORK.md))
- [x] Data pipeline scaffolded; medieval control set at 1,003 pairs (validated)
- [x] Framework/systems scaffolding: `AdapterManager` (runtime adapter swap), `blend_adapters.py`, training pipeline, FastAPI backend, Next.js demo — all domain-agnostic
- [ ] llama.cpp/GGUF inference path; re-measure TinyLlama latency; Qwen3-0.6B setup
- [ ] Modern-city dataset (600 pairs) + visibility-set annotation — **freeze 6 Sep 2026**
- [ ] PDM v2 (domain-agnostic) calibrated on real generations
- [ ] KBD implementation + planted forbidden facts
- [ ] Train 8 modern adapters × 2 base models
- [ ] α-sweep (KBD + PDM v2) for 3 adapter pairs — **the headline result**
- [ ] Flat-RAG baseline (matched token budget); latency/VRAM benchmarks; figures
- [ ] Paper draft

**Results freeze: Friday 23 October 2026** — no new experiments after this date, writing only.

Detailed week-by-week schedule: [`Docs/TODO.md`](Docs/TODO.md).

---

## Publication Target

**Primary:** IEEE Conference on Games (CoG)
**Backup:** ACM FDG · IEEE Access · ArXiv preprint
**Submission deadline:** mid-November 2026

---

## Team

### Mentor
**Mr. K. Sudhakar**
AP/CSE · Kongu Engineering College

### Developer
**Yugabharathi J**
Final Year Student · B.E. CSE · Kongu Engineering College
*Solo developer.*

---

*Built on open-source tools (TinyLlama and Qwen3 base models are Apache 2.0). Dataset and adapter weights will be released publicly upon paper submission.*
