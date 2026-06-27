# Domain-Adaptive Lightweight NPC AI Framework
### Parameter-Efficient Fine-Tuning for Real-Time Game NPC Dialogue

> **Final Year Research Project** · Yugabharathi · AI / NLP / Game Development  
> Department of Computer Science

---

## Overview

Modern games rely on either hand-authored NPC dialogue (expensive, static) or large cloud-hosted LLMs (slow, costly, impractical for consumer hardware). This project proposes a middle path: a single lightweight base model dynamically adapted with interchangeable **LoRA adapters** for different NPC personas and game domains.

The framework is directly motivated by two open problems identified in the literature:

- Gallotta et al. (2024) survey the roles of LLMs in games and note that running AAA games and LLMs in parallel on consumer hardware is currently infeasible due to computational requirements — calling for lightweight, locally deployable alternatives.
- Värtinen et al. (2022) demonstrate that fine-tuned GPT-2 can generate acceptable RPG quest descriptions, but highlight that entity consistency and contextual coherence remain unsolved — problems our game-state injection approach directly targets.

---

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Can a sub-2B parameter model with domain-specific LoRA adapters produce NPC dialogue comparable in persona fidelity to larger general-purpose models? |
| RQ2 | Does structured game-state context injection (location, inventory, quest status, relationship score) improve persona consistency across multi-turn NPC conversations? |
| RQ3 | What is the tradeoff between model size and domain accuracy when adapters are swapped dynamically at inference time? |
| RQ4 | Can dynamic adapter loading maintain a ≤300ms first-token latency on consumer CPU hardware — the practical real-time constraint for game dialogue? |

---

## Novelty

Prior work falls into two categories. Large model approaches (GPT-3/4, Claude) achieve good dialogue quality but are too slow and expensive for real-time game deployment. Small model approaches (fine-tuned GPT-2, TinyLlama baseline) are fast but suffer from persona drift and game-state blindness.

This project combines, for the first time:

1. **Sub-2B LoRA adapters** trained for NPC-specific persona fidelity (not general style transfer)
2. **Structured game-state injection** — location, inventory, quests, relationship scores passed as context at inference time
3. **Persona fidelity metric** — a probe classifier that tracks whether NPC responses stay in-character across conversation turns
4. **Real-time latency constraint** — ≤300ms first token on laptop CPU, a game-specific requirement absent from academic NLP benchmarks

---

## Domains

| Domain | NPC Type | Key Challenge |
|--------|----------|---------------|
| **Medieval RPG** *(primary)* | Blacksmith, quest-giver, merchant | Archaic speech register, lore consistency, world-state awareness |
| **Sci-Fi Merchant** *(secondary)* | Trader, informant | Technical jargon, bartering logic, future-setting vocabulary |
| **CS Tutor** *(baseline)* | Educational assistant | Explanatory tone, non-game domain — used to test adapter specificity |

> Healthcare was excluded as a domain. It introduces ethics review risk and dilutes the game AI narrative. Mentioned as future work only.

---

## Architecture

<img src="./Docs/Daigram.jpg" alt="Architecture">

---

## Evaluation

### Ablation Study

Four conditions measured across four metrics. This is the core experimental contribution.

| Condition | Description |
|-----------|-------------|
| A | Base model, no adapter |
| B | Base model + generic style prompt |
| C | Base model + LoRA adapter |
| D | Base model + LoRA adapter + game-state injection *(expected best)* |

### Metrics

| Metric | Measurement Method | Target |
|--------|--------------------|--------|
| **Persona Fidelity Score** | Probe classifier: response vs. persona profile | Primary metric |
| **Style Consistency** | Human evaluation (5-point Likert) | Domain match |
| **Persona Drift** | Fidelity score across turns 1, 3, 5, 10 | Stability over time |
| **Latency** | Time-to-first-token on laptop CPU | ≤ 300ms |
| **Memory** | Peak RAM during inference | ≤ 4GB |
| **Model Size** | Adapter file size vs. full model | Storage efficiency |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Base model | TinyLlama 1.1B (fallback: Qwen 0.5B) |
| Fine-tuning | HuggingFace PEFT + LoRA |
| Training | TRL + PyTorch |
| Inference | llama.cpp / Ollama |
| Backend | FastAPI + Python |
| Frontend | React (demo only) |
| Evaluation | GPT-4-as-judge + custom probe classifier |

---

## Dataset

A curated dataset of NPC dialogue pairs with structured persona tags and game-state annotations will be created and released as part of this project. Inspired by Värtinen et al.'s open quest dataset methodology, all data will be publicly available.

**Target:** 500+ annotated dialogue pairs per domain  
**Sources:** Public domain literature, fantasy game wikis, RPG dialogue corpora, educational QA datasets

---

## Base Papers

This project builds directly on the following two papers included in this repository:

**[1] Large Language Models and Games: A Survey and Roadmap**  
Gallotta, Todd, Zammit, Earle, Liapis, Togelius, Yannakakis — arXiv:2402.18659 (2024)  
→ Provides the typology of LLM roles in games and identifies lightweight, locally deployable NPC dialogue as an open research direction. Notes that running LLMs alongside games on consumer hardware is currently infeasible — the core problem this project addresses.

**[2] Generating Role-Playing Game Quests With GPT Language Models**  
Värtinen, Hämäläinen, Guckelsberger — IEEE Transactions on Games, Vol. 16, No. 1 (2024)  
→ Demonstrates GPT-2 fine-tuning for RPG quest generation and identifies entity consistency and contextual coherence as key unsolved problems in game-specific NLP. The quest dataset methodology and placeholder substitution technique directly inform our data pipeline.

---

## Related Work

| Paper | Relevance |
|-------|-----------|
| Hu et al., *LoRA* (2021) | Core PEFT technique used for all adapters |
| Dettmers et al., *QLoRA* (2023) | Quantized fine-tuning for consumer hardware |
| Park et al., *Generative Agents* (2023) | NPC social simulation; motivation for game-state injection |
| Zhang et al., *TinyLlama* (2024) | Base model; open-source, sub-2B, consumer deployable |

---

## Project Status

- [x] Proposal finalized
- [x] Literature review complete
- [ ] Dataset curation (in progress)
- [ ] Baseline experiments (Conditions A & B)
- [ ] LoRA adapter training (Condition C)
- [ ] Game-state injection (Condition D)
- [ ] Evaluation pipeline
- [ ] Paper draft

---

## Publication Target

**Primary:** IEEE Conference on Games (CoG) — Short Paper  
**Secondary:** ACL Student Research Workshop

---

## Author

### Mentor
**Mr.K.Sudhakar**
AP/CSE
Kongu Engineering College 

**Yugabharathi J**
**Soumya K**
**Soundariya M**
Final Year Student · B.E. CSE
Kongu Engineering College

---

*Built on open-source tools. Dataset and adapter weights will be released publicly upon paper submission.*