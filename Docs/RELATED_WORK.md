# Related Work

Differentiation table for the nine papers most adjacent to this project. One row per paper: what they do, and how this project differs. This is the source material for the paper's Related Work section — keep it current as papers are read.

The two overarching differentiators that apply across almost every row:

1. **Judge-free, per-turn, source-attributable measurement.** Prior work that measures epistemic scope or persona fidelity almost universally uses a judge model (GPT-4o, Openchat-3.6) producing an opinion score. KBD and PDM v2 are automatic, cheap enough to run per-turn at inference on the same hardware serving the game, and KBD is attributable to a specific knowledge source.
2. **This project measures rather than mitigates.** Several papers below *fix* character drift (via prompting, protective fine-tuning, or SFT). This project's contribution is a metric for it, and an analysis of what happens to the epistemic boundary under adapter interpolation.

Neither of these is a claim that persona drift is an *unmeasured* phenomenon — it is a named, quantified phenomenon (C-Score, NLI-entailment consistency, prompt-to-line / line-to-line / Q&A consistency, activation-space measures). The claim is the specific combination of judge-free, per-turn, cheap, and source-attributable.

---

## Differentiation table

| # | Paper | What they do | How this project differs |
|---|-------|--------------|--------------------------|
| 1 | **Andreasen & Esterle** — *Fixed-Persona SLMs with Modular Memory* (arXiv:2511.10277) | **Closest prior work.** TinyLlama-1.1B + LoRA personas + runtime-swappable memory modules + consumer-hardware benchmarks. Measure appropriate refusal for out-of-scope questions using **Openchat-3.6 as judge**. Find a curated ~115-pair set beats a ~564-pair synthetic set on factuality/context/fluency. | They swap **memory stores** with persona fixed in weights; this project swaps **persona adapters *and* visibility sets**. They have **no drift metric and no multi-turn degradation analysis**. KBD replaces their judge-based refusal check with an automatic, per-turn, source-attributable metric. This project is framed as a **direct extension** of theirs (same TinyLlama base, same consumer-hardware framing, same small-curated-dataset finding adopted as the 600-pair rationale). |
| 2 | **Wang et al.** — *Model Fusion with Multi-LoRA Inference* (arXiv:2509.24229) | Multi-LoRA serving + **LoRA parameter fusion in game NPC dialogue**. Their fusion averages adapters trained for the **same function** across different data sources. | Establishes that LoRA weight merging is **standard prior art** — so this project does **not** claim the `W_blend = α·W_A + (1-α)·W_B` technique. This project interpolates **different personas** with tunable α and asks whether the **epistemic boundary** interpolates or collapses to the union — a question about knowledge leakage, not about serving throughput. |
| 3 | **Buakhaw et al.** — *Deflanderization* (arXiv:2510.13586) | Names character drift ("flanderization") in NPC dialogue; **mitigates** it via prompting + LoRA SFT on Qwen3-14B. | They **mitigate**, this project **measures**. Different model scale (Qwen3-14B vs. sub-2B TinyLlama/Qwen3-0.6B for consumer hardware). |
| 4 | **Liu, Xie & Jiang** (MDPI *AI* 6(5):93) | Character hallucination via knowledge graph + AMR + protective fine-tuning; train protective scenarios so models express ignorance outside a character's cognitive scope. Scored as "Hallucination" on a **7-point Likert scale by GPT-4o**. | Judge-based Likert scoring vs. **KBD's automatic, judge-free attribution** to a specific knowledge source. Flat visibility-tagged knowledge items (set intersection) vs. their KG+AMR machinery. |
| 5 | **McGrath, Lorandi & Belz** — *Echoes of Others* (INLG 2025 demos) | LoRA 4-bit local RPG dialogue, guard/merchant roles, latency vs. GPT-4o Mini. | Two-page demo with **shallow evaluation** (3 scenarios, LLM-as-judge). This project contributes a metric (KBD/PDM v2), a full four-condition evaluation, and the α-sweep analysis. |
| 6 | **Nuriyev** — *Efficient Tool-Calling Multi-Expert NPC Agent* (arXiv:2511.01720) | Three LoRA adapters routed **by function** (tool / persona / direct). | Routing by **function** vs. this project's routing by **archetype** and analysis of interpolation across **epistemic access** (visibility sets), not tool vs. persona. |
| 7 | **Kim et al.** — *MART* (arXiv:2412.11189) | Merchant NPC pricing and negotiation. | Adjacent domain, **different problem** (negotiation/pricing, not persona/epistemic drift). |
| 8 | **Tódová** — *A Quest for Information* (Masaryk University MSc, 2025) | LLM-driven NPCs for game-based learning. | **Different application** (educational NPCs), no drift metric. |
| 9 | **IEEE** — arnumber 11419836 | **Not yet reviewed** — PDF inaccessible at time of writing. | **Action: retrieve and assess before the paper draft.** Placeholder row; revisit. |

---

## Established persona-drift metrics to cite (context, not competitors)

These situate PDM v2 as a *cheaper* member of an existing family, not a discovery:

- **C-Score** — persona consistency scoring.
- **NLI-entailment consistency** — persona statements as entailment premises.
- **Prompt-to-line / line-to-line / Q&A consistency** — turn-level consistency checks.
- **Activation-space measures** — internal-representation drift.

PDM v2's differentiator against all of these: domain-agnostic feature families (domain lexicon, register markers, formality score, syntactic profile, stance), judge-free, per-turn, and calibrated against real generations from a trained adapter rather than against the dataset.

---

## Open action items

- [x] **8 of 9 source PDFs retrieved and archived locally** (2026-08-15) under `Base Papers/`: `RW01_andreasen_esterle_2511.10277.pdf`, `RW02_wang_2509.24229.pdf`, `RW03_buakhaw_deflanderization_2510.13586.pdf`, `RW05_mcgrath_lorandi_belz_inlg2025_demos.pdf`, `RW06_nuriyev_2511.01720.pdf`, `RW07_kim_mart_2412.11189.pdf`, `ai-06-00093-v2.pdf` (row 4, Liu/Xie/Jiang), `A-Quest-for-Information-Tereza-Todova.pdf` (row 8, Tódová thesis).
- [ ] **Row 9, IEEE arnumber 11419836, still not retrieved.** Both direct `curl` (with a browser user-agent) and the browser-automation tooling failed — the IEEE Xplore stamp/download endpoint is JS-gated and returned an empty response either way. Not a transient failure specific to this session; genuinely needs manual retrieval (e.g. via institutional library access) before the paper draft. Row 9 in the differentiation table above stays a placeholder until then.
- [ ] Confirm exact venue/volume strings for each citation when the bibliography is built.
- [ ] Now that the other 8 PDFs are local, worth a pass re-verifying the differentiation-table summaries (rows 1–8) against the actual paper text rather than abstracts/prior notes — not done in this pass, flagged for before the draft.
