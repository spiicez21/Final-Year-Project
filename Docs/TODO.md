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

**Scope note, 2026-08-08 — medieval cut too, project is modern-city only now.** The dual-era (modern + medieval control) design from 2026-08-07 lasted one day. The medieval dataset/adapters/eval results below are kept on disk, **not deleted**, but are no longer part of the active plan — no medieval-vs-modern comparison, no "control condition" framing anywhere going forward. A Week 2 guard-adapter training run using the medieval dataset was **killed mid-run** (step 3/36) when this decision landed — see Week 2 below for the real scheduling consequence this creates.

---

## Work already banked (carries forward)

Complete and reusable — most is domain-agnostic and survives the modern-city pivot. Full blow-by-blow history is in git and prior revisions of this file.

**Framework / systems (domain-agnostic — carries over untouched):**
- [x] `training/train_adapter.py` — QLoRA pipeline (transformers + peft + trl SFTTrainer). Real fixes banked: bf16-end-to-end training (fp16+bf16 GradScaler crash), step-checkpointing after a thermal-TDR crash, and a **completion-only-loss bug** (loss was computed over the whole templated sequence, not just the assistant response — fixed by emitting `prompt`/`completion` columns).
- [x] `training/blend_adapters.py` — adapter interpolation via PEFT `add_weighted_adapter(combination_type="linear")` (correctly accounts for each adapter's alpha/rank scaling).
- [x] `backend/adapter_manager.py` — `AdapterManager` loads adapters onto one resident base and hot-swaps via `set_adapter()`. **Verified live:** second call with a different archetype reported `adapter_switch_ms: 0` — real proof of the no-reload claim.
- [x] `backend/main.py` — FastAPI `POST /chat`, `GET /domains`, `GET /archetypes`, `GET /health`. Model preloaded at startup.
- [x] `frontend/` — Next.js demo (domain/archetype dropdowns, live chat, live PDM bar, per-message latency). Verified end-to-end in-browser.

**Medieval-fantasy dataset + results (archived 2026-08-08 — out of scope, kept for reference, not deleted):**
- [x] 1,003-pair medieval set, validated (1003/1003 schema-conformant, 0 duplicate ids). Sources: Gutenberg (Shakespeare/Chaucer/Malory, 626), chimbiwide rule-based rewrite (150), hand-authored (227).
- [x] 50-entry stress-test corpus, held out.
- [x] **Key medieval finding (methodology evidence, not a current-scope result):** training-data *composition* matters more than LoRA rank/alpha/epochs for lexically-specific persona control. Eight configs on the full mix produced near-zero archaic markers; training only on the 626 dialect-dense Gutenberg entries (`medieval_r8_gutonly`) fixed it. Held-out (uncontaminated) A-vs-B: marker rate 4.7% → 14.0%, mean drift 0.9766 → 0.9396. Worth keeping in mind for the modern-city training runs (Weeks 7–8) — the same composition effect may recur.
- [x] **Blending sweep:** α=0.2 beat pure gutonly on both marker rate and over-insertion on the medieval set — a genuine sweet spot there. Not assumed to transfer to the modern α-sweep (Weeks 9–10); that sweep is run fresh on modern adapters.
- [x] **Full fine-tune reference:** best raw persona (46.8% marker rate) but worst over-insertion (0.8932) and 11.86 GB on disk vs. 2.3 MB LoRA — a clean systems tradeoff, general enough to still cite for Contribution 4 even though the domain it was measured on is now out of scope.

> These medieval numbers are **archived, not a current result** — do not report them as project findings without noting they predate the modern-only scope decision. The old PDM baseline mean drift of **0.9833 is saturated and reports nothing useful** regardless of domain — do **not** carry it forward (see PDM v2 note in Week 5).

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

- [x] **Re-measured TinyLlama latency on the GGUF path — real numbers, done 2026-08-07.** Built `evaluation/run_gguf_latency.py`: downloads `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (Q4_K_M, 668.8MB) via `huggingface_hub`, serves via `llama_cpp.Llama` (zephyr chat format — verified TinyLlama-Chat's actual template, not ChatML), measures wall-clock per response.
  - **CPU-only** (prebuilt wheel, no compiler toolchain on this machine): **1352ms mean** at `max_tokens=40`, down from Ollama's 3486ms — **2.6x faster**, but still over the 500ms target.
  - **Tried GPU offload** (installed the CUDA `llama-cpp-python` wheel, `n_gpu_layers=-1`) — needed the CUDA runtime DLLs (cudart/cublas), which torch's `cu121` wheel already had bundled (`torch/lib/`), avoided a full CUDA-toolkit install by pointing `PATH` at that directory instead. Verified via verbose load log: **23/23 layers genuinely offloaded to CUDA0 (MX450)**, not a silent CPU fallback. **Result: no speedup** (~1350ms, identical to CPU) — `nvidia-smi` shows this card power-capped at 5W even under load, and for a model this small the GPU's kernel-launch overhead + low throughput at that power ceiling doesn't beat 8-core CPU AVX2 inference. A real, reportable hardware-constraint finding, not a bug left unfixed.
  - **At `max_tokens=20`** (a more game-realistic NPC bark length, not a 40-token paragraph): **816ms mean** — **4.3x faster than the old baseline**, but still misses 500ms. Timing is roughly linear (~27ms/token + ~270ms fixed overhead) — hitting 500ms via token-count alone would mean an ~8-token response, too short to be a usable NPC line.
  - **Honest conclusion:** GGUF quantized CPU inference is a large, real win over the old Ollama path, but TinyLlama-1.1B does not hit <500ms on this hardware at a usable response length regardless of CPU/GPU. Qwen3-0.6B (roughly half the params) was proposed as the next test for this — **explicitly parked by user decision, 2026-08-07: "leave qwen, use tinyllama itself."** Staying single-base (TinyLlama only) for now; revisit Qwen3 later if needed.
  - Noted separately, not fixed (quality issue, not latency): some GGUF outputs leak `"Never break character."` / narrator-voice artifacts into the response — a prompt-template issue, tracked for later, out of scope for the latency task.
  - **Thread/batch tuning** (`n_threads=os.cpu_count()`, `n_batch=512`, explicit `n_gpu_layers=0` instead of relying on defaults) shaved default CPU latency from 1352ms to **1186ms mean** at `max_tokens=40` (~12% faster) — small but real, baked into `run_gguf_latency.py`'s defaults now. Script also self-contained: adds torch's `cu121` DLL dir to `PATH` internally so the CUDA-linked `llama-cpp-python` wheel loads without a manual env-var step every run.
- [x] **Merge a trained LoRA adapter into base weights, export to GGUF Q4_K_M, serve via llama.cpp — done end-to-end, 2026-08-07**, using the existing `medieval_r8_gutonly` control adapter as the vehicle (no modern adapter exists yet).
  - `training/merge_lora.py` (new): loads base in bf16 on CPU (not 4-bit — merging into an already-quantized base would compound error on top of the LoRA delta), `PeftModel.from_pretrained` + `merge_and_unload()`, saves a plain HF checkpoint indistinguishable from a full fine-tune.
  - GGUF conversion: shallow-cloned `ggml-org/llama.cpp` for `convert_hf_to_gguf.py` (pure Python, no compiler needed) into `training/llama.cpp/`. Installed only `gguf`+`sentencepiece` rather than its full pinned `requirements-convert_hf_to_gguf.txt` (which demands `transformers==4.57.6`/`torch==2.11.0`/`numpy~=1.26` — would have downgraded the pinned training stack). Ran clean against the newer versions already installed.
  - Quantization to Q4_K_M: **no compiled `llama-quantize.exe` on this machine** (no C/C++ toolchain — see Known issues). Instead called `llama_cpp.llama_model_quantize()` directly, the same C function the binary would call, exposed through `llama-cpp-python`'s ctypes bindings — `training/quantize_gguf.py` (new). 2.2GB f16 -> 667.8MB Q4_K_M.
  - **Benchmarked the real merged+quantized adapter** (`evaluation/run_gguf_latency.py --model medieval_gutonly`, added a `local_path` option alongside the HF-download path): **777ms mean** at `max_tokens=20`, CPU. Outputs show genuine archaic markers (`"Nay, I will not have a room for the night..."`) with no `"Never break character."` leakage — confirms the merge preserved the trained persona correctly, not just a speed number.
  - Still misses <500ms, consistent with the stock-model finding above — the ceiling is model size + this hardware, not the merge/quantize/serve pipeline itself, which now works correctly end-to-end.
- [ ] Stand up Qwen3-0.6B — **parked**, see above.
- [ ] Ensure dataset formatting + any tokenizer-dependent metric code handles **both** Llama-chat and ChatML templates.

### Week 2 (Aug 17–23) — One adapter, end-to-end · exit: in-character generation

- [x] **LoRA config updated for the new runs** — `training/configs/lora_config.yaml` now `r=16`, `alpha=32`, target `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (was `r=8`, `q_proj, v_proj` only). Applies to all runs going forward; the archived medieval adapters above were trained under the old config and aren't affected retroactively.
- [x] **Added `--archetype` filtering to `train_adapter.py`** (`build_dataset()` now takes an `archetype` param, filters `persona.archetype == X`) — needed either way, for training single-archetype adapters instead of whole-domain ones.
- [x] **Started training a guard adapter (189 medieval entries) as the Week 2 pipeline-validation run — killed mid-run (step 3/36) when the modern-only decision landed 2026-08-08.** This wasn't wasted: it proved the new r=16/7-module config + `--archetype` filter work mechanically (model loaded, dataset built to exactly 189 entries, training started, wandb run `medieval-r16_a32_guard-adapter` logged) before being stopped on purpose, not because of a bug.
- [ ] **Real blocker exposed by the modern-only decision: Week 2 has no data to train on.** The plan sequences dataset-building in Weeks 3–4 (Aug 24–Sep 6), *after* Week 2 — that ordering only worked when Week 2 could borrow the already-existing medieval set to validate the pipeline. With medieval out of scope, Week 2's exit condition ("adapter generates in-character text") has nothing to train on until Weeks 3–4 land. Options, not yet decided:
  - (a) Hand-author a small pilot batch now (~20–30 entries, one modern archetype, e.g. police officer) purely to unblock pipeline validation — real dataset work still happens on schedule in Weeks 3–4.
  - (b) Let Week 2 slip into Weeks 3–4 and validate the training pipeline as soon as the first real modern entries exist, accepting the schedule's own "slip absorbs into weeks 3–8" clause.
  - Needs a decision before Week 2 can actually close out.

### Weeks 3–4 (Aug 24 – Sep 6) — Modern dataset, 600 pairs · **exit: frozen Sep 6**

- [ ] Build the modern-city set: **600 pairs (75 × 8 roles)**, persona re-voiced from cleared sources (Taskmaster, MultiWOZ, SODA, Synthetic-Persona-Chat). Record `provenance` per entry. See [`DATA_PIPELINE.md`](DATA_PIPELINE.md).
  - Load Taskmaster/MultiWOZ as **raw JSON from GitHub**, not via `load_dataset`.
  - **Rejected, do not retry:** SGD (CC BY-SA 4.0 share-alike, not the Apache 2.0 assumed originally), PersonaChat (no formal licence on the HF mirrors), Cornell Movie-Dialogs (no open licence), DailyDialog (NC + share-alike). Full reasoning in `DATA_PIPELINE.md`.
- [ ] Eight archetypes, standalone (no longer framed as a mapping from anything): police officer, shopkeeper, professor, bartender, social worker, pharmacist, executive, service worker.
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

## Reconciliation notes (conflicts flagged, then fixed)

- **Dual-era (medieval control + modern experimental) → modern-only — 2026-08-08.** The dual-era design in this file's 2026-08-07 rewrite is superseded one day later by a user decision to drop medieval entirely and run this as a single-domain modern-city project. Medieval dataset/adapters/eval results are kept on disk, not deleted, but are archived/out-of-scope — see the "Work already banked" section above and the killed Week 2 run. README.md, this file, and `Docs/DATA_PIPELINE.md` were all updated (the latter's section header and medieval-dataset framing changed; its modern-city sourcing tables — Taskmaster/MultiWOZ/SODA/Synthetic-Persona-Chat, rejected sources — did not need to change). No code changes were needed beyond what the 2026-08-07 reconciliation already did — `train_adapter.py`'s `DATASET_PATHS`/`MODERN_ARCHETYPES` were already modern-city-only in naming, just no longer described as "mapped from medieval."
- **Crime-city → modern-city — reconciled 2026-08-07.** `training/train_adapter.py`'s `MODERN_ARCHETYPES` now reads `police officer/shopkeeper/professor/bartender/social worker/pharmacist/executive/service worker` (was cop/dealer/boss/...), its modern `SYSTEM_PROMPTS` entry dropped the crime-city framing, and the unused `healthcare`/`education` dataset-path stubs were removed (cut per README). `data/scripts/chimbiwide_converter.py`'s `--domain modern` branch was reverted entirely — chimbiwide isn't part of the modern pipeline anymore (Taskmaster/MultiWOZ/SODA/Synthetic-Persona-Chat only), so the file is back to medieval-only, matching what `DATA_PIPELINE.md` documents. `evaluation/pdm_scorer.py`'s `DIALECT_PATTERNS_MODERN` (crime slang) was removed rather than left as a wrong stand-in for PDM v2. `modern_npc_dataset.json`'s metadata now carries the correct 8-role list and an explicit "extractor not built yet" note instead of a stale crime-city archetype list.
- **Cornell Movie-Dialogs** was previously listed as the next modern source; it is now **rejected** (no open licence). The DATA_PIPELINE task list reflects the rejection.
- **PDM lexicon in code is still medieval-only.** PDM v2 (domain-agnostic feature families, calibrated on real generations) is new work for Week 5 — not built, and `pdm_scorer.py`'s docstring now says so explicitly instead of carrying a wrong modern lexicon.

---

## Known issues / environment notes

- System Python 3.13 is broken on this machine (`0x80070003`). Scripts run via the Python 3.10 install. `Lib/site-packages` was found **empty** on 2026-08-06 — reinstall from `requirements.txt` before running anything locally.
- Local GPU: NVIDIA MX450, **2.15 GB VRAM** — tight even for 4-bit QLoRA on TinyLlama 1.1B, and cannot sustain multi-hour runs (thermal/power throttling to a 5 W cap observed). **Budget Colab/cloud GPU** for real training. Full fine-tune (~8–12 GB optimizer state) cannot run locally at all.
- **Resume-from-checkpoint is broken below torch 2.6** (CVE-2025-32434 blocks `torch.load` of optimizer state). `train_adapter.py` skips resume below 2.6 and restarts fresh. Not upgrading torch (risks destabilizing the working bitsandbytes/peft/trl stack).
- `ollama serve` is unreliable under sustained load (`WinError 10013`, socket exhaustion). Eval/serving scripts use a persistent session, retry-with-backoff, and per-entry incremental save (`--resume` skips written ids). Expect the same flakiness on the GGUF serving path.
- **No C/C++ compiler toolchain on this machine** (`nmake`/MSVC missing) — `pip install llama-cpp-python` fails trying to build from source. Fix: install from the prebuilt wheel index instead (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` or `/cu121` for the CUDA build), not from PyPI directly.
- **CUDA `llama-cpp-python` wheel needs CUDA runtime DLLs (cudart64_12/cublas64_12) that aren't installed standalone** on this machine (driver only, no toolkit). Workaround used: add torch's `cu121` wheel's bundled DLL directory to `PATH` (`.../site-packages/torch/lib`) instead of installing the full CUDA Toolkit — confirmed via verbose load log that all layers genuinely offload to `CUDA0`, not a silent fallback.
- **Background `Bash` tasks in this Claude Code session die silently** if the session/turn boundary interrupts them mid-run (no error, no completion notification, output file just stops updating) — happened twice with `pip install torch`. If a background install looks stuck, verify with `Get-Process`/`pip show` rather than trusting the log file alone; rerun in the foreground if genuinely dead.
- **No `llama-quantize.exe`** (compiled binary, needs the same missing C/C++ toolchain) — worked around by calling `llama_cpp.llama_model_quantize()` directly via `llama-cpp-python`'s ctypes bindings (`training/quantize_gguf.py`), same underlying C function, no binary needed.
