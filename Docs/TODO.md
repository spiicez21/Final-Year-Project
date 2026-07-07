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
- [x] GPT-4o augmentation pass targeting underrepresented archetypes — **superseded, not pursued.** `gpt4o_augmentor.py` gap-report tooling works (`--dry-run`), but the actual generation was replaced by hand-authored batches (same schema-conformant output, zero API cost, twelve batches covering the identical archetype gaps). Script kept in the repo for future use if `OPENAI_API_KEY` becomes available.
- [x] Write `dataset_validator.py` (schema conformance, duplicate detection, archetype balance report). Ran clean: 0 schema errors, 1003/1003 valid, no duplicate ids or duplicate input/output pairs. 19 warnings, all the already-known Malory stray-tag artifact (~6% of the 300 Malory entries) — nothing new surfaced. `--strict` flag exits 1 on errors for future CI use.
- [x] Reach 1,000 total entries — **1003/1000, done.** Archetype balance is not perfect (see table) but the spec's raw count target is met.
- [x] Build 50-entry stress test corpus (`data/processed/stress_test_corpus.json`) — held out, NOT for training. 15 identity_challenge, 15 out_of_world_reference, 12 modern_language, 8 extended_pressure (10-11 turns each, one per archetype). All 8 archetypes covered. Schema differs from the main dataset: `{id, archetype, stress_test_type, turns: [...]}` — no pre-written NPC output, since these are player-side scripts meant to be run against whichever condition (A/B/C/D) is under test per `Specs.md` section 8's protocol (record the turn number where PDM > 0.7 and archaic markers vanish).

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

- [x] Set up `training/` pipeline: `train_adapter.py` (QLoRA via `transformers`+`peft`+`trl` SFTTrainer), `configs/lora_config.yaml`, `configs/training_args.yaml`. Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (same variant as the Ollama baseline, for a fair Condition A vs B comparison). 4-bit QLoRA (nf4, double quant) — the 2.15GB MX450 can't fit fp16 full-model training.
  - **Real bug hit and fixed:** fp16 training crashed (`_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`) because TinyLlama-Chat's checkpoint has some layers natively in bf16, and mixing fp16-scaled gradients with bf16 params breaks torch's GradScaler. Fixed by training in bf16 end-to-end instead (no loss scaler needed) — MX450 reports `torch.cuda.is_bf16_supported() == True` (software-emulated, Turing has no native bf16 tensor cores, but it works).
  - Smoke-tested at 5 and 50 samples before committing to a full run — 50-sample timing (~55s/step, cold) badly overestimated the real full-dataset speed (~17s/step warm) — don't trust small-sample timing extrapolations on this hardware.
- [x] Train medieval adapter (r=8, α=16, 3 epochs) — **done.** 1003 samples, 189 steps, 108.8 min actual runtime (GPU thermal-throttled a few times mid-run, 63°C observed — added time beyond the ~54min best-case estimate, still nowhere near the 3hr worst case). Loss 3.36 → 1.58-1.83, mean token accuracy 0.42 → 0.69. Adapter: `training/adapters/medieval_r8/adapter_model.safetensors`, 2.26MB (spec budget: <500MB — comfortably under). W&B run `medieval-r8-adapter` (project defaulted to `huggingface`, not `npc-ai-framework` — fix `project=` in `SFTConfig` for future runs, cosmetic only).
- [ ] Train healthcare adapter
- [ ] Train education adapter
- [ ] Run LoRA rank ablation (r = 4, 8, 16, 32) on medieval adapter — **motivated by a real finding, not just checkbox-following**: r=8 sanity check (`training/quick_inference.py`) showed loss converging fine (3.36→1.6) and content shifting medieval-thematic, but **zero archaic dialect markers** (no thee/thou/hath) in 5 test generations, confirmed with both sampled and greedy decoding (ruled out sampling noise). Training data has plenty of signal (797/1003 entries, 79.5%, have real dialect_features) — so r=8's capacity, not the data, is the suspected bottleneck. r=16 running now.
  - r=8: done, see Phase 3 entry above.
  - r=16: **first attempt crashed** at step 56/189 (~1hr in) — no Python traceback, process just died (exit code 4). Per-step time had degraded badly beforehand (17s → 106s/step over that hour), strongly suggesting GPU thermal throttling escalated into a Windows WDDM driver TDR reset. `nvidia-smi` confirmed the GPU was healthy afterward (idle, 48°C, 0 processes) — a transient crash, not hardware failure. Lost the full hour since `save_strategy: epoch` hadn't hit its first checkpoint (step 63) yet. **Fix applied:** `train_adapter.py` now hardcodes `save_strategy="steps"`, `save_steps=15` (~4-5min between saves) plus auto-resume-from-latest-checkpoint on restart, regardless of `training_args.yaml`'s `save_strategy` field (kept for spec-alignment reference only). **Retry succeeded**, 106min, no crash.
  - **r=16 result — same null finding as r=8.** Loss/accuracy curves nearly identical to r=8 (3.37→1.82 vs 3.36→1.81, token accuracy 0.68 both). `quick_inference.py` sanity check: still zero `thee/thou/hath/dost` markers across 5 test generations. One output ("Aye, aye, a sword for a knight") is genuinely medieval-flavored but "aye" isn't in `pdm_scorer.py`'s `DIALECT_PATTERNS` list — a real gap in the scorer's vocabulary, worth widening (add aye, ye, morrow, milord, etc.) before trusting PDM=0 readings at face value. Doubling rank (8→16) did not meaningfully change either the loss curve or the archaic-marker outcome — this is genuine evidence that **rank is likely not the bottleneck**, contrary to the working hypothesis. Candidate real bottlenecks: (a) `DIALECT_PATTERNS` is too narrow and undercounts real archaic style already present, (b) 3 epochs / this LR may be capped regardless of rank, (c) TinyLlama-Chat's RLHF-aligned "helpful assistant" prior may need a stronger system-prompt intervention (explicit "use words like thee, thou, hath") rather than relying purely on implicit stylistic transfer from fine-tuning data alone. r=32 not yet run — likely to repeat the same null pattern based on this trend, so pausing the rank sweep here pending a decision on which alternative to chase.
  - **Cheap follow-up test (no retraining, `quick_inference.py --directive`):** added an explicit system-prompt instruction directly naming the target words ("use archaic words such as thee, thou, thy, hath, dost, doth, wilt, nay, art, 'tis, prithee, forsooth"), tested on the existing r=16 adapter. Result: **worse, not better** — one output (merchant) degenerated into 27x repeated "aye" (a greedy-decoding repetition-loop failure mode), and none of the explicitly-named words appeared even once across all 5 prompts. This rules out "the model just needs to be told" as an easy fix.
  - **Net conclusion so far:** rank (8→16) and directive prompting are both ruled out as quick fixes for near-zero archaic-marker production. This matters for the paper's core claim (RQ1/Contribution 4: "comparable persona consistency to GPT-4o" is the primary claim) — a null Condition B isn't a footnote, it's the central result failing, so decided to fix it before writing anything up rather than reframe as a limitation prematurely.
  - **alpha=32 (r=8): also null.** Trained clean, 90min (survived two thermal stalls this time thanks to the step-checkpoint fix, didn't crash). Loss slightly better than r=8/r=16 (3.15→1.72 vs ~3.36→1.8) — alpha=32 does help the loss curve marginally. But `quick_inference.py` sanity check: still **zero** thee/thou/hath/etc markers across all 5 prompts, and one output (innkeeper) degenerated into a 27x repeated "a room" loop under greedy decoding — the same repetition-loop failure mode seen earlier at r=16/directive-prompt ("aye" x27). Two different configs now show this greedy-decoding degeneracy, suggesting it's a property of the small model/dataset combination, not a one-off.
  - **Three full training runs (r=8, r=16, r=8_a32), three null results on archaic-marker production.** Loss curves all converge to roughly the same ~1.6-1.8 plateau regardless of rank or alpha — strong evidence the bottleneck isn't LoRA capacity or scaling at all. Likely real cause: archaic pronouns are rare tokens even in a dataset where most entries contain them (a handful of occurrences among many tokens per example) — cross-entropy loss is dominated by common words, so it can drop a lot without meaningfully raising the odds of those specific rare tokens.
  - **8 epochs (r=8) running** to test the rare-token-reinforcement hypothesis directly. Hit two real bugs launching this one, both fixed before the run that's now in progress:
    1. **Directory-naming collision:** output dir only encoded rank+alpha (`medieval_r8`), not epoch count — a naive 8-epoch run at r=8 would've reused the *original 3-epoch r=8 run's directory*, and the auto-resume-from-checkpoint logic would've tried to resume from that unrelated run's final checkpoint. Fixed: dir suffix now includes epoch count when overridden (`medieval_r8_e8`).
    2. **Resume-from-checkpoint is fundamentally broken on this machine's torch (2.5.1):** transformers refuses to `torch.load` optimizer/scheduler state below torch 2.6 (CVE-2025-32434 restriction) — confirmed by the collision above instantly crashing on `trainer.train(resume_from_checkpoint=...)` before any training happened. This means the crash-resilience checkpointing added after the r=16 crash was **saving fine but could never actually resume** — a future crash would've crashed again immediately on the resume attempt. Fixed: `train_adapter.py` now checks torch version and skips resume entirely below 2.6 (falls back to a fresh start, with a printed note) instead of attempting a resume that's guaranteed to fail. Not upgrading torch itself — real risk of destabilizing the working bitsandbytes/peft/trl stack for an already-working pipeline.
    - Verified the original `medieval_r8` adapter's `adapter_model.safetensors` was untouched (unchanged mtime) after the collision — no data lost, the crash happened before any weights were touched.
  - **8-epoch run stopped manually at step 120/504 (~1.9 epochs), not completed.** After the naming/resume fixes, the run itself went clean for the first ~90 steps, then degraded far worse than any previous run: by step 120 (3h42m elapsed) `nvidia-smi` showed the GPU **power-throttled to a 5W cap** (normal MX450 TDP ~25-30W), 66-67°C, stuck at ~110-117s/step (vs ~17s/step baseline) — tqdm's remaining-time estimate was 11h46m. This is a sustained-load thermal/power ceiling this laptop GPU hits on runs longer than ~1.5-2hrs; the three prior ~90-110min runs had brief stalls but recovered, this one didn't. Manually killed (PID 56084) rather than wait it out.
  - **Tested checkpoint-120 anyway (~1.9 epochs in) — same null result as all three prior full runs.** Zero archaic dialect markers, and another degenerate repetition loop ("aye" x27, merchant prompt — third occurrence of this failure mode across different configs). This is now **four independent confirmations** (r=8/3ep, r=16/3ep, r=8_a32/3ep, r=8/~1.9ep-partial) of the same outcome: loss converges fine, thematic content shifts medieval, archaic lexical markers never appear.
  - **Decision: pausing further local training experiments.** This hardware (2.15GB MX450) cannot sustain the multi-hour runs needed to properly test the epoch-count hypothesis, on top of the crash/throttle history already logged above. **Recommendation: migrate remaining Phase 3 training (epoch sweep, healthcare/education adapters once their datasets exist, rank r=32, adapter blending experiments) to Google Colab** — free-tier T4/A100 GPUs would run each ~90-110min local experiment in a few minutes and don't share this thermal ceiling. `train_adapter.py` is already portable (no Windows-specific paths beyond the hardcoded Python 3.10 interpreter note elsewhere in this doc, which doesn't apply on Colab) — would need a small Colab notebook wrapper (`!pip install`, mount Drive or clone repo for `data/processed/medieval_npc_dataset.json`, run the same script) rather than a rewrite.
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