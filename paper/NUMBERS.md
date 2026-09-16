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

## Player memory — Section "A first dynamic case" (added 2026-09-11)

All from `evaluation/results/player_memory_results.json` (`evaluation/run_player_memory.py`)
unless noted. 5 NPCs, greedy decoding, keyword scoring. "Reported" pools the
`test` and `test2` probe sets (8 + 8 questions × 5 NPCs = 80).

| Claim | Value | Source key in `summary` |
|---|---|---|
| Same visit: transcript / +memory / +retry | 22/80, 33/80, 56/80 | `session/reported/{none,memory,memory+retry}` |
| Returning: transcript / +memory / +retry | 0/80, 37/80, 64/80 | `returning/reported/...` (none is trivially 0) |
| Retry fired | 54 of 160 reported questions | sum of `retries` over `{session,returning}/{test,test2}/memory+retry` (14+9+15+16) |
| Retry latency | first gen p50 339.5 ms; retry p50 244.7 ms | `latency_ms` (first p90 is inflated by model swaps across 5 NPCs in a 3-model pool; not quoted) |
| False "you told me" on unrelated questions | 0/40 | `{session,returning}/test2/memory+retry` → `false_recall` (20 each) |
| Never-told NPC knew the player | 0/16; claimed "you told me" 0/16 | `isolation` |
| KB-leaking replies | 0 | `kbd_leaking_replies` |
| Intent filter covers test2 | 6/8 | `test2_gate_coverage` |
| Hand audit of retried replies | 54 read: 48 faithful, 2 invented detail (keyword hits), 4 wrong/non-answer | `rows` with `retried: true`, split test/test2 — audited by hand 2026-09-11 |
| Placement, design set | none 13/40, clause only 11/40, greeting demo early 14/40, demo early 14/40, demo late 21/40 | `evaluation/results/memory_placement_results.json` (`evaluation/run_memory_placement.py`) |

> The `dev` set was used for every design decision and is not reported.
> `test` results were seen before the retry was built; `test2` was written
> after the retry was designed and before it was run. Both were written by the
> same person who wrote the retry's intent filter, so neither is blind — the
> paper says so in Limitations.

## Fact grounding — Playable Integration, third finding (added 2026-09-11)

From `evaluation/results/grounding_results.json` (`evaluation/run_grounding.py`,
default run: `block` vs `server` through the real `/chat` handler) unless noted.
5 NPCs × 10 questions; facts parsed from `campus_facts.gd`. Reported numbers
are the `test` phrasings; `dev` was used for design.

| Claim | Value | Source |
|---|---|---|
| All facts in system prompt | 31/50 grounded, 7 refusals | `summary["test/block"]` |
| Retrieval + late turn + retry | 42/50 grounded, 0 refusals | `summary["test/server"]` |
| Canteen on the wrong floor | 5/5 NPCs ("second floor") | `rows`, dev, variant block, "where's the canteen?" |
| Fixed / broke | 16 fixed (10 wrong + 6 refusals → correct), 5 correct → wrong | verdict transitions block→server over `rows`, test |
| Regressions explained | 3 × "how long is the open day on for?" (retrieval "long → years"), 1 baseline scorer false positive (Reyes, "near the ground floor") | same rows, read by hand |
| KB-leaking replies | 0 in both layouts | `summary["kbd_leaking_replies"]` |
| Retrieved-only system block + retry | 39/50 (test) | `run_grounding.py test sys_retry` (placement ablation; table in the script docstring) |
| Latency, 8-turn conversation, median | block 620 ms; retrieved-only system block 2416 ms; server 906 ms (incl. retries) | `evaluation/results/grounding_latency.json` (`run_grounding.py latency`) |

> **Bug found and fixed while doing this.** The in-process evaluations call
> the server's KBD scorer, whose knowledge base is loaded at server startup.
> Run in-process, it was empty, so the first player-memory run's "0 KB-leaking
> replies" checked nothing. `gguf_server.load_scoring()` now loads it, both
> evaluations call it and assert it loaded, and the memory evaluation was
> rerun: still 0 leaks, recall unchanged, first-generation median 339.5 ms
> (was 354.6 ms; the paper uses 340).

## Understanding the player — learned extraction and intent (added 2026-09-15)

Not yet in `main.tex`. Replaces the pattern-based extraction and intent that the
player-memory numbers above were measured with; those sections must be rerun
before they are quoted again.

| Claim | Value | Source |
|---|---|---|
| Reported failure (pattern rules) | "my name is yugabharathi" then "i am class cse d" → name "Class"; also "Tamil", "Hosteller", "Cr" as names | reproduced 2026-09-15; `evaluation/memory_extraction_cases.json` lines |
| TinyLlama-1.1B JSON-schema extraction (tried, rejected) | dev 8/12 lines fully right with grounding, ~1.3 s/line, slot confusions ("madurai" as year) | scratch prototype, 2026-09-15 — not kept as code |
| Extractor | GLiNER small (`urchade/gliner_small-v2.1`, Apache-2.0, 166M) fine-tuned on 6,000 generated lines, 3.0 min on RTX 5060, val loss 10.53 → 0.316 | `training/extractor/player_facts_gliner/training_summary.json` |
| Value disjointness | generator aborts if any case value is in its lists (it caught "cse" and "trichy") | `training/extractor/make_extraction_data.py`, `check_no_leakage` |
| Zero-shot GLiNER, test | 16/36 lines fully right, 9/37 slots, 1 forbidden slot learned, scenarios 3/5, 56 ms/line | `evaluation/results/memory_extraction_results.json` → `base` |
| **Fine-tuned, test** | **35/36 lines, 36/37 slots, 0 forbidden, scenarios 5/5, 58 ms/line**; threshold 0.3 chosen on dev (11/12) | same file → `fine-tuned` |
| Only miss | "i am doing mini project on iot" → department "iot" | same file, test rows |
| Intent, pattern rules | dev 14/14, test 35/40 | `evaluation/results/intent_results.json` → `rules` |
| **Intent, learned heads** | **dev 14/14, test 40/40, ~51 ms/line** (encoder forward, CPU) | same file → `learned`; heads from `training/extractor/train_intent.py` |

> Caveats: 36 extraction and 40 intent test lines, one author; generator
> templates were written after the case lines, so sentence shapes may overlap
> even though values cannot. Synthetic validation (100% intent, 0.316 extraction
> loss) says nothing about real lines and is not quoted as a result.

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

## Reported news (evaluation/run_events.py, 2026-09-15)

3 report lines × 3 (told NPC, other NPC) pairs, real pipeline, fine-tuned extractor at 0.7.

| | result |
|---|---|
| report captured as an event | 9/9 (1 report line kept as the player's quoted words: no incident span ≥ 0.7) |
| unheard NPC mentions the incident (leak, KBD) | 0/9 |
| greeted NPC passes heard news on | 9/9 — **all 9 needed the `news` restart**; 0/9 before the repair existed |
| NPC answers "is anything happening?" with heard news | 9/9 — **all 9 needed the restart**; 0/9 before |
| authored fallback line used | 0 |
| told-NPC reply repaired (`report_refusal`) | 6/9; 2 unrepaired replies still off ("I'll go check it out.", "That's not a good idea.") |

Honest reading: the prompt carried the news every time and the 1.1B model ignored it;
passing news on works only because the guard restarts the reply on the news line's
opening words and the model completes it. An incident threshold of 0.4 was tried and
reverted: it made "the canteen food was bad today" an incident on test lines and had
been chosen from a run_events report line, not dev.

Conversations at threshold 0.7 (with news repairs): reported 15/15 memory slots, dev 20/20,
test 25/25, test2 20/25 (department "it" in "1st year it" missed for all 5 NPCs); spurious slots 0 in all.
