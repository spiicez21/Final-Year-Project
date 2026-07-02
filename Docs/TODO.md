# Project TODO

Tracks the roadmap in [`DevFiles/Specs.md`](../DevFiles/Specs.md) section 9 against actual repo state. Update this file as tasks land — don't let it drift.

*Last synced: 2026-07-02*

---

## Phase 1 — Foundation (Weeks 1–3)

- [ ] Set up Python environment: PyTorch, HuggingFace Transformers, PEFT, TRL
- [ ] Install TinyLlama via Ollama — confirm it runs locally
- [ ] Run baseline evaluation — record all outputs and latency
- [ ] Define and implement PDM formula (reference implementation exists in `Specs.md` Appendix A — port to `evaluation/pdm_scorer.py`)
- [ ] Set up Weights & Biases project for experiment tracking
- [x] Create Git repo with `.gitignore` (`BlenderFiles`, `DevFiles` currently ignored — revisit once `training/adapters/` exists, per spec: `adapters/`, `__pycache__/`, `*.bin`)

## Phase 2 — Dataset (Weeks 4–6)

- [ ] Expand hand-authored entries from 10 → 50 (focus on guard, merchant, herbalist archetypes — biggest gaps, see table below)
- [x] Run Gutenberg extractor on Shakespeare (Hamlet, Macbeth, Julius Caesar) — 126 pairs
- [x] Run Gutenberg extractor on Chaucer (Canterbury Tales) — 200 pairs
- [ ] Run Gutenberg extractor on Malory (*Le Morte d'Arthur*) — raw text not yet downloaded
- [x] Build chimbiwide conversion pipeline: download + medieval-plausibility filter working (255/300 rows pass)
- [ ] Implement `register_rewrite()` in `chimbiwide_converter.py` (currently raises `NotImplementedError` — needs LLM backend + IP scrub before merge, per spec's "Medium IP Risk" note)
- [ ] Filter `microsoft/crd3` for fantasy dialogue entries — not started
- [ ] GPT-4o augmentation pass targeting underrepresented archetypes (`gpt4o_augmentor.py` gap-report works via `--dry-run`; generation path untested — needs `OPENAI_API_KEY`)
- [ ] Write `dataset_validator.py` (schema conformance, duplicate detection, archetype balance report)
- [ ] Reach 1,000 total entries with balanced archetype distribution (currently 326, see gap table)
- [ ] Build 50-entry stress test corpus (`data/processed/stress_test_corpus.json`) — not started

### Archetype gap (current 326 entries vs. spec target)

| Archetype | Target | Current | Gap |
|-----------|-------:|--------:|----:|
| Guard | 200 | 6 | 194 |
| Scholar | 150 | 18 | 132 |
| Merchant | 150 | 23 | 127 |
| Innkeeper | 100 | 5 | 95 |
| Peasant | 150 | 63 | 87 |
| Noble | 150 | 88 | 62 |
| Herbalist | 50 | 0 | 50 |
| Clergy | 50 | 123 | 0 (over target — don't add more from Gutenberg) |

**Deliverable:** `medieval_npc_dataset_v1.json` with 1,000+ entries — **not yet met** (326/1000).

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
- `chimbiwide_converter.py` needs the `datasets` package (`pip install datasets`) — not in a committed requirements file yet since `requirements.txt` doesn't exist at repo root.
- README.md content predates `Specs.md` v1.0 (different RQs/domains/contributions) — reconciled 2026-07-02, see [README.md](../README.md).
