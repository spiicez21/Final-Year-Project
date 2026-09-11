# Where every number in `main.tex` comes from

Nothing in the draft is illustrative, rounded-for-effect, or invented. This
table is the audit trail. If a number in the paper cannot be found here, treat
it as a bug and remove it.

Sources are given as repository paths. `TODO.md` refers to `Docs/TODO.md`,
which is the project's running lab notebook and records the results of runs
whose raw output files store responses but not scores.

## Framework and dataset

| Claim in paper | Value | Source |
|---|---|---|
| Base model | TinyLlama-1.1B-Chat | `training/train_adapter.py`, `README.md` |
| LoRA config | r=16, alpha=32, 7 target modules | `README.md`, adapter configs |
| Adapter size on disk | 25.3 MB | TODO.md "Work already banked" — measured from `adapter_model.safetensors` |
| Full fine-tune size | 11.86 GB (~470x larger) | TODO.md "Work already banked" |
| Adapter switch cost | 0 ms on second call | TODO.md, `backend/adapter_manager.py` verification note |
| Dataset size | 1,160 pairs | `data/processed/modern_npc_dataset.json` (counted) |
| Per-archetype counts | 325 / 140 / 120 / 115x5 | same file, counted by archetype |
| Knowledge base | 17 facts, 2–3 per archetype | `data/processed/knowledge_base.json` |
| Signature phrases | 39 | same file, counted |

## RQ1 — KBD

| Claim | Value | Source |
|---|---|---|
| First calibration run | 1/14 forbidden flagged | TODO.md Week 6 |
| Corrected result | **5/14 (36%)** forbidden violated | TODO.md Week 6; `evaluation/results/kbd_calibration_results.json` |
| Control probes | 3/3 correctly unflagged | TODO.md Week 6 |
| Per-archetype table | social worker 2/2, executive 2/2, police officer 1/2, bartender 1/2 | TODO.md Week 6 revisit (2026-08-15) |

## RQ2 — α-sweep

| Claim | Value | Source |
|---|---|---|
| α-sweep table (8 probes/cell) | see paper Table II | TODO.md Weeks 9–10, second table (2026-08-15 rerun) |
| Earlier 4-probe table | 100/100/100/25/0 etc. | TODO.md Weeks 9–10, first table — cited only as the sample-size caution |
| PDM v2 across α (police/pharmacist) | 0.407–0.419 | TODO.md Weeks 9–10 |
| Raw per-probe responses | — | `evaluation/results/alpha_sweep_results.json` |

> **Note.** `alpha_sweep_results.json` stores responses only — it has no
> `violation` or `kbd` field. Scoring it naively yields 0% everywhere, which is
> an artifact of the missing field, not a result. The scored table lives in
> TODO.md. Worth re-emitting scores into the JSON before submission so the
> paper's headline table is reproducible from a result file rather than from
> the notebook.

## RQ3 — PDM v2

| Claim | Value | Source |
|---|---|---|
| Per-archetype baseline vs adapter, all 7 | see paper Table III | `evaluation/results/pdm_v2_multi_results.json` |
| Mean gap | 0.0660 | computed from that file |
| Ordering correct | 7/7 (`order_ok: true`) | same file |
| Old saturated PDM v1 | 0.9833 | TODO.md, cited as the motivation for v2 |
| BERTScore baseline / adapter | 0.8530 / 0.9049 (+0.0519) | `evaluation/results/bertscore_results.json` |

## Conditions

| Claim | Value | Source |
|---|---|---|
| Condition C violation | 4/4 (100%) | TODO.md Week 11; `evaluation/results/condition_c_results.json` |
| Condition B police officer | 0/4 (0%) | TODO.md Week 11 |
| Condition B pharmacist | 100% | TODO.md Week 11 |
| Condition C token overhead | 59 (police), 45 (pharmacist) | TODO.md Week 11 |
| Hybrid B+C per archetype | 0%–100% forbidden, 0%–100% control | `evaluation/results/hybrid_bc_results.json` |

## RQ4 — latency

| Claim | Value | Source |
|---|---|---|
| Old machine, 40 tokens | 1186 ms mean (fail) | TODO.md Week 1 |
| Old machine, 20 tokens | 816 ms mean | TODO.md Week 1 |
| Old machine, fitted cost | ~27 ms/token + ~270 ms fixed | TODO.md Week 1 |
| Old machine, GPU offload | no speedup, 23/23 layers offloaded, 5 W cap | TODO.md Week 1 |
| Ollama baseline | 3486 ms | TODO.md Week 1 |
| **Current machine, 40 tokens** | **411.4 / 448.8 ms mean; p50 312.5 / 331.0 ms; pass** | `evaluation/run_gguf_latency.py --model modern_police --max-tokens 40`, two runs, 2026-09-08 |
| Current machine, 20 tokens | 345.7 ms mean, p50 338.5 ms | same script, `--max-tokens 20` |
| Current machine spec | 20 cores, RTX 5060 laptop; CPU inference | `os.cpu_count()`, `nvidia-smi` |

> `n=5` per run in `run_gguf_latency.py`, and the first call of each run carries
> warm-up, which is why the paper gives a range rather than a single mean.
> Increasing `n` before submission would be cheap and worth doing.

## Playable integration

| Claim | Value | Source |
|---|---|---|
| Median warm generation in play | 333 ms | measured 2026-09-08 over 7 turns, `SampleGame` + `backend/gguf_server.py` |
| Instruction-vs-demonstration failure | "I don't have a job, I'm just a machine" | `SampleGame/new-game-project/Systems/NPC/README.md` |
| Persona-break rate after fix | 0 / 25 probes | same README, regression probe |
| Break-in hallucination, KBD undefined | `kbd: null`, 0 leaked ids | same README |

## Training and the held-out split (added 2026-09-11)

| Claim | Value | Source |
|---|---|---|
| Held-out split | 173 held-out / 987 train, stratified by archetype | `data/processed/modern_split.json`, from `training/make_split.py` (seeded; byte-identical on rerun) |
| Police officer, held-out loss minimum | 1.7105 at step 34 of 54 | `training/adapters/modern_r16_a32_policeofficer_ho_legacy_mb16/training_summary.json` |
| Police officer, held-out loss at end of training | 1.7425 | same file, last entry of `eval_loss_curve` |
| Archetype size range | 115–325 examples | `modern_npc_dataset.json`, counted |

> **Train-set evaluation.** PDM v2 (Table III) and BERTScore were computed on
> prompts drawn from the adapters' own training data, with PDM v2's reference
> features built from the same entries. Both are reported in the paper with
> that caveat stated in the text and the table caption. They must not be
> quoted elsewhere as generalisation results.

## Citations (verified 2026-09-11)

Every entry in `refs.bib` was resolved against its DOI (`doi-mcp`) or, for the
INLG demo, Google Scholar and the ACL Anthology (`scholar_mcp`). Claims made
about each paper in Related Work were then checked against the PDFs in
`Base Papers/`. Corrections made in that pass:

| Entry | Was | Is |
|---|---|---|
| closest prior work | "Andreasen & Esterle" | **Braas** & Esterle |
| Braas & Esterle, multi-turn | "no multi-turn analysis" | they do evaluate multi-turn *context retention*; not drift |
| Braas & Esterle, models | "on TinyLlama-1.1B" | DistilGPT-2, TinyLlama-1.1B **and** Mistral-7B |
| Wang et al., "fusion" | averaging adapters across data sources | averaging LoRA checkpoints across epochs of one run |
| Nuriyev, experts | tool / persona / direct | tool calling / tool-response interpretation / direct dialogue |
| Liu et al. | wrong title, wrong first names | *Personalized Non-Player Characters: …*, Xiao Liu, Zhenping Xie, Senlin Jiang |
| McGrath et al. | invented title, wrong first name | *…Real-Time LLM Dialogue Generation for Immersive NPC Interaction*, James McGrath |
| Buakhaw, Wang, Nuriyev | wrong first names | Pasin, Kangxu, Mahammad |

Verbatim-overlap check (7-word spans) against all seven source PDFs: two
shared spans, both legitimate — a quoted phrase from Liu et al. in quotation
marks with citation, and the names of Nuriyev's three system components.

## Not in the paper, and why

- **Adapter routing accuracy 47.4% (91/192)** — `evaluation/results/adapter_routing_results.json`.
  Real and interesting, but it is a different contribution (automatic archetype
  selection) and does not bear on RQ1–RQ4. Left out to keep the argument
  narrow; a candidate for the journal version.
- **Medieval-domain results** — archived, predate the modern-city-only scope
  decision of 2026-08-08. Must not be reported as project findings.
- **Qwen3-0.6B** — not trained. This is why RQ4's base-model comparison is
  marked `[PENDING]` in the draft.
