# Project TODO

Tracks the roadmap in [`DevFiles/Specs.md`](../DevFiles/Specs.md) section 9 against actual repo state. Update this file as tasks land — don't let it drift.

*Last synced: 2026-07-02*

---

## Phase 1 — Foundation (Weeks 1–3)

- [x] Set up Python environment: PyTorch, HuggingFace Transformers, PEFT, TRL (torch 2.5.1+cu121, transformers 5.12.1, peft 0.19.1, trl 1.7.0, accelerate 1.14.0 — see root `requirements.txt`)
- [x] Install TinyLlama via Ollama — confirm it runs locally (`ollama pull tinyllama`, 637MB, verified via 326-call baseline run)
- [x] Run baseline evaluation — record all outputs and latency (Condition A, 326/326 entries, `evaluation/results/baseline_outputs.json` + `baseline_metrics.csv`; mean latency 3486ms, mean drift 0.9833 — see `Docs/DATA_PIPELINE.md` note below on the run)
- [x] Define and implement PDM formula (`evaluation/pdm_scorer.py`, ported from `Specs.md` Appendix A, validated against the spec's worked example — 0.9 drift on a persona-collapse conversation)
- [x] Set up Weights & Biases project for experiment tracking — logged in as `spicez21` (`spicez21-kongu-engineering-college` org), verified via `wandb.Api()`. No project created yet — pass `report_to="wandb"` + `project="npc-ai-framework"` in `TrainingArguments` when adapter training starts; W&B auto-creates the project on first run.
- [x] Create Git repo with `.gitignore` (`BlenderFiles`, `DevFiles` currently ignored — revisit once `training/adapters/` exists, per spec: `adapters/`, `__pycache__/`, `*.bin`)

## Phase 2 — Dataset (Weeks 4–6)

- [x] First gap-fill batch, hand-authored in-session (no API cost): 27 entries — guard +7, herbalist +5, merchant +5, innkeeper +5, scholar +5 (`SYN-0327`..`SYN-0353`, `source: synthetic_claude`).
- [x] Second gap-fill batch, hand-authored in-session: 21 entries — merchant +6, scholar +5, innkeeper +5, herbalist +5 (`SYN-0804`..`SYN-0824`). Checked CRD3 as an alternative first — dead end, HF dropped script-based dataset loading and CRD3 has no Parquet conversion.
- [x] Third gap-fill batch, hand-authored in-session: 20 entries — merchant +5, scholar +5, innkeeper +5, herbalist +5 (`SYN-0825`..`SYN-0844`).
- [x] Fourth gap-fill batch, hand-authored in-session: 20 entries — merchant +5, scholar +5, innkeeper +5, herbalist +5 (`SYN-0845`..`SYN-0864`).
- [x] Fifth gap-fill batch, hand-authored in-session: 20 entries — merchant +5, scholar +5, innkeeper +5, herbalist +5 (`SYN-0865`..`SYN-0884`).
- [x] Sixth gap-fill batch, hand-authored in-session: 20 entries — merchant +7, scholar +7, innkeeper +6, herbalist skipped this round (gap nearly closed at 25) (`SYN-0885`..`SYN-0904`).
- [x] Seventh gap-fill batch, hand-authored in-session: 19 entries — merchant +6, scholar +6, innkeeper +4, herbalist +3 (`SYN-0905`..`SYN-0923`).
- [x] Eighth gap-fill batch, hand-authored in-session: 17 entries — merchant +5, scholar +5, innkeeper +5, herbalist +2 (`SYN-0924`..`SYN-0940`).
- [x] Ninth gap-fill batch, hand-authored in-session: 17 entries — merchant +5, scholar +5, innkeeper +5, herbalist +2 (`SYN-0941`..`SYN-0957`).
- [x] Tenth gap-fill batch, hand-authored in-session: 17 entries — merchant +5, scholar +5, innkeeper +4, herbalist +3 (`SYN-0958`..`SYN-0974`).
- [x] Eleventh gap-fill batch, hand-authored in-session: 19 entries — merchant +6, scholar +4, innkeeper +4, herbalist +3, guard +2 (`SYN-0975`..`SYN-0993`).
- [x] Twelfth gap-fill batch, hand-authored in-session: 10 entries — merchant +4, scholar +3, innkeeper +3 (`SYN-0994`..`SYN-1003`). **1,000-entry milestone crossed — 1003 total, 227 hand-authored across twelve batches.**
- [x] Run Gutenberg extractor on Shakespeare (Hamlet, Macbeth, Julius Caesar) — 126 pairs
- [x] Run Gutenberg extractor on Chaucer (Canterbury Tales) — 200 pairs
- [x] Run Gutenberg extractor on Malory (*Le Morte Darthur*, Rhys ed., Gutenberg #46853) — 300 pairs, quality≥5. Required a new extraction mode: this edition has **no quotation marks at all**, dialogue is only marked by an inline `<clause>, said <name>` tag (Early Modern English convention). Added `parse_tagged_dialogue()` + `TAGGED_SOURCES` to `gutenberg_extractor.py`. Known limitation: only the first `said X` tag per clause is stripped, so a few multi-speaker clauses leave a stray embedded tag in the output text — not worth over-engineering, flagged for the eventual `dataset_validator.py` pass.
- [x] Build chimbiwide conversion pipeline: download + medieval-plausibility filter working (255/300 rows pass)
- [x] Implement `register_rewrite()` in `chimbiwide_converter.py` — rule-based archaic rewriter (contractions, you/your/yours → thee/thy/thine/thou, irregular thou-verb + inverted-question fixups, sentence-capitalization, small vocab swap list). No LLM/API call, deterministic. 150 entries merged (`CHM-0354`..`CHM-0503`, `source: chimbiwide`, `quality_score: 5`, tagged `register_rewritten` for later review — grammar is "good enough," not Shakespeare-quality).
- [x] Filter `microsoft/crd3` for fantasy dialogue entries — **dead end, not pursued.** HF's `datasets` library (5.0.0) no longer supports script-based dataset loading at all (`RuntimeError: Dataset scripts are no longer supported`), and CRD3 has no official Parquet conversion (the dataset viewer explicitly refuses to auto-convert it). Would require downgrading `datasets` or hand-fetching raw GitHub dumps. Even then: real actual-play D&D transcripts are modern spoken English (same heavy-rewrite burden as chimbiwide) and dominated by adventurer PCs, not merchant/innkeeper NPCs — poor effort-to-payoff for our specific gaps. Skipped in favor of hand-authored batches.
- [ ] GPT-4o augmentation pass targeting underrepresented archetypes (`gpt4o_augmentor.py` gap-report works via `--dry-run`; generation path untested — needs `OPENAI_API_KEY`; superseded in practice by hand-authored batches, see above)
- [ ] Write `dataset_validator.py` (schema conformance, duplicate detection, archetype balance report)
- [x] Reach 1,000 total entries — **1003/1000, done.** Archetype balance is not perfect (see table) but the spec's raw count target is met.
- [ ] Build 50-entry stress test corpus (`data/processed/stress_test_corpus.json`) — not started

### Archetype gap (current 1003 entries vs. spec target — remaining imbalance, not a blocker for Phase 3)

| Archetype | Target | Current | Gap |
|-----------|-------:|--------:|----:|
| Merchant | 150 | 87 | 63 |
| Scholar | 150 | 100 | 50 |
| Innkeeper | 100 | 63 | 37 |
| Herbalist | 50 | 38 | 12 |
| Guard | 200 | 189 | 11 |
| Noble | 150 | 182 | 0 (over) |
| Peasant | 150 | 221 | 0 (over) |
| Clergy | 50 | 123 | 0 (over) |

Total count target met. Remaining per-archetype imbalance (merchant/scholar/innkeeper still under target, noble/peasant/clergy over) is a training-data-quality concern for Phase 3, not a blocker — LoRA training can proceed on the current distribution, and further balancing can happen in parallel if archetype-specific persona drift looks weak in evaluation.

**Deliverable:** `medieval_npc_dataset_v1.json` with 1,000+ entries — **not yet met** (844/1000).

## Phase 3 — Training (Weeks 7–10)

- [ ] Train medieval adapter (r=8, α=16, 3 epochs)
- [ ] Train healthcare adapter
- [ ] Train education adapter
- [ ] Run LoRA rank ablation (r = 4, 8, 16, 32) on medieval adapter
- [ ] Implement adapter blending function (`training/blend_adapters.py` — reference impl in `Specs.md` Appendix B)
- [ ] Test blending at α = 0.2, 0.5, 0.8 (medieval × scholar)
- [ ] Save all adapter checkpoints with W&B run IDs

**Blocked on:** Phase 2 dataset completion (need enough per-archetype coverage before training is meaningful).

## Phase 4 — Evaluation (Weeks 11–14)

- [ ] Run all 4 conditions on full evaluation set (PDM, BERTScore, latency, RAM)
- [ ] Run stress test corpus through all conditions
- [ ] Conduct human evaluation (recruit 20–30 participants, run Google Form)
- [ ] Calculate Cohen's Kappa
- [ ] Build results tables (automated + human)
- [ ] Profile adapter load/switch latency specifically

## Phase 5 — Paper + Submission (Weeks 15–18)

- [ ] Write Results section first
- [ ] Write Method section
- [ ] Write Related Work section
- [ ] Write Introduction and Abstract last
- [ ] Prepare figures: architecture diagram, PDM curve, blending heatmap
- [ ] Submit to IEEE CoG (check current deadline at ieee-cog.org)
- [ ] Upload ArXiv preprint same week as CoG submission

---

## Known issues / environment notes

- System Python 3.13 install is broken (`0x80070003` launch error on this machine). All scripts run via the Python 3.10 install at `C:\Users\spicez\AppData\Local\Programs\Python\Python310\python.exe`. Fix or reinstall 3.13 before relying on the `python3`/`python` shell aliases.
- Local GPU is an NVIDIA MX450 with **2.15GB VRAM** — CUDA confirmed working (torch 2.5.1+cu121), but this is tight even for 4-bit QLoRA on TinyLlama 1.1B. Budget for cloud/Colab GPU time for anything beyond small-scale local iteration.
- `ollama serve` is unreliable under sustained load on this hardware — intermittent `500` errors and `WinError 10013` connection failures during the 326-entry baseline run (~13% of first-pass calls failed). `evaluation/run_baseline.py` now uses a persistent session, retry-with-backoff (3 retries, 3/6/9s), a 0.5s inter-request delay, and saves to disk after every entry (not batched) so a crash loses at most one call. `--resume` skips ids already written. Expect the same flakiness during actual adapter training — plan checkpointing accordingly.
- `chimbiwide_converter.py` needs the `datasets` package (`pip install datasets`) — not in a committed requirements file yet since `requirements.txt` doesn't exist at repo root.
- README.md content predates `Specs.md` v1.0 (different RQs/domains/contributions) — reconciled 2026-07-02, see [README.md](../README.md).