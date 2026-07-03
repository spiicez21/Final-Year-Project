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

- [x] First gap-fill batch, hand-authored in-session (no API cost): 27 entries — guard +7, herbalist +5, merchant +5, innkeeper +5, scholar +5 (`SYN-0327`..`SYN-0353`, `source: synthetic_claude`). Gaps still large (see table) — more passes needed, target was 50 total hand-authored.
- [x] Run Gutenberg extractor on Shakespeare (Hamlet, Macbeth, Julius Caesar) — 126 pairs
- [x] Run Gutenberg extractor on Chaucer (Canterbury Tales) — 200 pairs
- [ ] Run Gutenberg extractor on Malory (*Le Morte d'Arthur*) — raw text not yet downloaded
- [x] Build chimbiwide conversion pipeline: download + medieval-plausibility filter working (255/300 rows pass)
- [ ] Implement `register_rewrite()` in `chimbiwide_converter.py` (currently raises `NotImplementedError` — needs LLM backend + IP scrub before merge, per spec's "Medium IP Risk" note)
- [ ] Filter `microsoft/crd3` for fantasy dialogue entries — not started
- [ ] GPT-4o augmentation pass targeting underrepresented archetypes (`gpt4o_augmentor.py` gap-report works via `--dry-run`; generation path untested — needs `OPENAI_API_KEY`)
- [ ] Write `dataset_validator.py` (schema conformance, duplicate detection, archetype balance report)
- [ ] Reach 1,000 total entries with balanced archetype distribution (currently 353, see gap table)
- [ ] Build 50-entry stress test corpus (`data/processed/stress_test_corpus.json`) — not started

### Archetype gap (current 353 entries vs. spec target)

| Archetype | Target | Current | Gap |
|-----------|-------:|--------:|----:|
| Guard | 200 | 13 | 187 |
| Scholar | 150 | 23 | 127 |
| Merchant | 150 | 28 | 122 |
| Innkeeper | 100 | 10 | 90 |
| Peasant | 150 | 63 | 87 |
| Noble | 150 | 88 | 62 |
| Herbalist | 50 | 5 | 45 |
| Clergy | 50 | 123 | 0 (over target — don't add more from Gutenberg) |

**Deliverable:** `medieval_npc_dataset_v1.json` with 1,000+ entries — **not yet met** (353/1000).

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