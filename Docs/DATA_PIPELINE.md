# Data Pipeline

Documents `data/` — how it's structured, what each script does, and how to run the pipeline. Schema reference: `DevFiles/Specs.md` section 6.

## Directory layout

```
data/
├── raw/
│   ├── gutenberg/           # Plain-text public domain sources (medieval, archived)
│   │   ├── hamlet.txt       # Gutenberg #1524
│   │   ├── macbeth.txt      # Gutenberg #1533
│   │   ├── caesar.txt       # Gutenberg #1785
│   │   ├── canterbury.txt   # Gutenberg #2383 (Purves ed.)
│   │   └── malory.txt       # Gutenberg #46853, Le Morte Darthur (Rhys ed.)
│   ├── modern/               # gitignored, large — see "Modern-city dataset" below
│   │   ├── taskmaster/        # git clone, google-research-datasets/Taskmaster (1.1GB)
│   │   ├── multiwoz/          # git clone, budzianowski/multiwoz (378MB)
│   │   ├── soda/               # HF snapshot_download, allenai/soda (817MB)
│   │   └── synthetic-persona-chat/  # HF snapshot_download, google/Synthetic-Persona-Chat (37MB)
│   └── huggingface/         # HF `datasets` cache (gitignored-worthy, large)
│
├── processed/
│   ├── medieval_npc_dataset.json   # Archived control dataset, schema v1.0
│   ├── modern_npc_dataset.json     # Active dataset, schema v1.0
│   └── stress_test_corpus.json     # 50 held-out persona-breaking conversations (NOT for training)
│
└── scripts/
    ├── gutenberg_extractor.py      # Play + poem + tagged-dialogue extraction (medieval, archived)
    ├── chimbiwide_converter.py     # HF chimbiwide/NPC-Dialogue_v2 filter + rewrite (medieval-only)
    ├── soda_extractor.py           # allenai/soda speaker-role extraction — the active modern-city source
    ├── gpt4o_augmentor.py          # Gap-fill synthesis for underrepresented archetypes (superseded by hand-authored batches)
    └── dataset_validator.py        # Schema conformance, duplicate detection, archetype balance report (medieval-only currently)
```

Python interpreter used for all of the above: `C:\Users\spicez\AppData\Local\Programs\Python\Python310\python.exe` (system 3.13 is broken on this machine — see [TODO.md](TODO.md)).

## The dataset: modern city (600 pairs)

The project is **modern-city only** as of 2026-08-08 (see `Docs/TODO.md` reconciliation notes for the full history — an earlier dual-era design with a medieval control condition, and before that a crime-city direction, both superseded). A **600-pair** set (75 × 8 roles), **persona re-voiced** from cleared source corpora. **Dataset freeze: 6 September 2026.** See [Modern-city dataset](#modern-city-dataset-600-pairs) below.

Eight archetypes: police officer, shopkeeper, professor, bartender, social worker, pharmacist, executive, service worker.

### Rationale for the 600-pair target

Andreasen & Esterle (arXiv:2511.10277) found that LoRA fine-tuning on a curated **~115-pair** set outperformed a **~564-pair** synthetic set on factuality, context retention, and fluency, attributing the gap to dataset quality and overfitting. Small and curated is a **defensible methodological choice, not a compromise** — this is worth a sentence in the paper's dataset section.

## Archived: medieval-fantasy dataset (out of scope, kept on disk)

**1,003 entries** in `data/processed/medieval_npc_dataset.json` — 1,000-entry target met (2026-07-04). Not deleted, not part of the active project as of the 2026-08-08 modern-only decision; kept as reference (methodology evidence, systems/deployability tradeoff numbers for Contribution 4 — see `Docs/TODO.md`).

| Source | Pairs | Method |
|--------|------:|--------|
| Hamlet | 60 | Play speaker-cue extraction |
| Julius Caesar | 45 | Play speaker-cue extraction |
| Macbeth | 21 | Play speaker-cue extraction |
| Canterbury Tales | 200 | Frame-narrative quote extraction |
| Le Morte Darthur (Malory) | 300 | Inline dialogue-tag extraction (no quotation marks in this edition) |
| chimbiwide/NPC-Dialogue_v2 | 150 | HF filter + rule-based archaic rewrite (no LLM) |
| Hand-authored (Claude, in-session, no API cost) | 227 | Direct schema-conformant writing across 12 batches, targeted at worst archetype gaps |

Final archetype distribution: peasant 221, guard 189, noble 182, clergy 123, scholar 100, merchant 87, innkeeper 63, herbalist 38. `gutenberg_extractor.py`, `chimbiwide_converter.py` (medieval-only), `dataset_validator.py` below still operate on this set if ever needed again — none of that tooling was removed.

## Modern-city dataset (600 pairs)

**Target: 600 pairs (75 × 8 roles)** in `data/processed/modern_npc_dataset.json`. **Freeze: 6 September 2026 — no additions after that date.**

**State as of 2026-08-08: 659 entries, target already reached.**

| Source | Entries | Method |
|--------|--------:|--------|
| Hand-authored pilot (Week 2, `POL-*`) | 59 | Original content, police officer only — see `Docs/TODO.md` |
| SODA (`SOD-*`) | 600 | `data/scripts/soda_extractor.py`, 75 per archetype — see below |

Archetype distribution: `police officer 134` (59 pilot + 75 SODA), all other 7 archetypes at 75 each. Not trimmed to exactly 600 — the police overage is real training data, not a defect.

Taskmaster and MultiWOZ are downloaded but **not yet extracted from** — SODA alone reached the numeric target on its own (see `soda_extractor.py` below), so pulling additional variety from them is an optional quality enhancement, not a blocker. Revisit once PDM v2/KBD (Weeks 5–6) can actually evaluate whether the current set needs more register diversity.

**Persona re-voicing, not copying** was the original plan (source corpora supply scenario scaffolding, rewritten in the target archetype's voice). In practice, SODA's raw dialogue is already natural contemporary register for a role whose *name* already matches the target archetype (e.g. a speaker literally named "Police Officer") — so `soda_extractor.py` does direct extraction + relabeling, not a rewrite pass, and says so honestly in each entry's `conversion_note` rather than overclaiming. Provenance (source corpus + original row index) is recorded per entry in the schema `provenance` field.

**Annotate in the same pass:**

- **`persona_features[]`** — the PDM v2 reference feature set for each entry (domain lexicon, register markers, formality, syntactic profile, stance). PDM v2 is calibrated against real generations, but the dataset carries the reference features.
- **Visibility set** — the archetypes permitted to know each knowledge item referenced in the entry. KBD (README C1) resolves violations as a **set intersection**; flat visibility tags, **not** a hierarchical tree, **not** Neo4j, **not** propagation delays or decay.

### Source datasets — cleared and downloaded

All licences below were **verified from each source directly** on 2026-08-07 (not taken from the brief — see the SGD lesson under [rejected](#source-datasets--rejected-licence-diligence)). Raw sources live in **gitignored** `data/raw/modern/`.

| Dataset | Verified licence | How pulled | On disk | Use |
|---------|------------------|-----------|---------|-----|
| Taskmaster (TM-1/2/3/4) | CC BY 4.0 | `git clone --depth 1` (GitHub raw JSON) | `data/raw/modern/taskmaster/` (~1.1 GB) | Task-oriented scaffolding; covers shopkeeper / service-worker / transaction scenarios |
| MultiWOZ 2.2 | MIT | `git clone --depth 1` (GitHub raw JSON) | `data/raw/modern/multiwoz/` (378 MB) | Multi-domain service dialogue structure. **Correction, 2026-08-08:** actual downloaded service distribution is `restaurant 4728, hotel 4182, train 3931, attraction 3485, taxi 1872, hospital 108, bus 6` — **no `police` domain at all** despite this row's original claim (verified by loading all 21 `dialogues_*.json` files directly, not assumed). Not used for police officer; see the SODA row instead. |
| SODA | CC BY 4.0 | `huggingface_hub.snapshot_download` (parquet) | `data/raw/modern/soda/` (817 MB) | 1.19M social dialogues. **The actual source used for all 8 archetypes** (`data/scripts/soda_extractor.py`) — its `speakers` field carries real in-narrative role names (e.g. "Police Officer", "CEO"), searched directly rather than relying on domain tags. Verified real per-archetype match counts before extracting: `executive 50252, professor 18425, shopkeeper 16510, service worker 11681, social worker 9920, police officer 8606, bartender 1758, pharmacist 99`. |
| Synthetic-Persona-Chat | CC BY 4.0 | `huggingface_hub.snapshot_download` (parquet) | `data/raw/modern/synthetic-persona-chat/` (~38 MB) | Persona-grounded turn structure (covers the role the brief assigned to PersonaChat) |

**Loading notes:**
- **Taskmaster + MultiWOZ**: pulled as **raw JSON from GitHub**, *not* via `load_dataset` — HF deprecated script-based loading and these have no reliable Parquet conversion (the **same failure mode that killed `microsoft/crd3`**, see [Not yet built](#not-yet-built)). Do not retry them through `load_dataset`.
- **SODA + Synthetic-Persona-Chat**: pulled via `huggingface_hub.snapshot_download(repo_type="dataset")` (they ship real Parquet, so this works where `crd3`/SGD-style script loading does not). This avoids needing the full `datasets` library.
- **Environment note, superseded 2026-08-08:** the paragraph above (only Python 3.14, `truststore` needed for SSL) described a *different* machine than the one used for the rest of this project — that machine's `data/raw/modern/` never made it here since the folder is gitignored. All 4 sources were re-downloaded from scratch on **this** machine (the `spicez`/Python 3.10 box every other doc references) using the same `huggingface_hub`/`git clone` commands, no `truststore` workaround needed here — plain SSL verification worked fine on this network.

### Source datasets — rejected (licence diligence)

Recorded so they are not retried, and worth a paragraph in the paper's dataset section:

| Dataset | Reason rejected |
|---------|-----------------|
| **Schema-Guided Dialogue (SGD)** | Brief assumed *Apache 2.0*; the canonical `google-research-datasets/dstc8-schema-guided-dialogue` repo's `LICENSE.txt` is actually **CC BY-SA 4.0 (share-alike)** — the same contamination property that rejected DailyDialog. **Dropped** (user decision, 2026-08-07). Its roles (shopkeeper, pharmacist, service worker) are covered by Taskmaster + MultiWOZ + SODA. The clone was deleted. |
| **PersonaChat** | HF mirrors (`bavard/personachat_truecased`, `AlekseyKorshuk/persona-chat`) declare **no licence** — same red flag as Cornell. **Not downloaded.** `google/Synthetic-Persona-Chat` (CC BY 4.0, cleared above) covers the identical "persona-grounded turn structure" role. Revisit only if a clearly-licensed PersonaChat source is found. |
| **DailyDialog** | CC BY-NC-SA 4.0. **Share-alike** would contaminate the released dataset's licence, and **NC** conflicts with public release. |
| **Cornell Movie-Dialogs Corpus** | **No formal open licence.** Unsafe for a dataset intended for public release on paper submission. *(Supersedes the earlier TODO note that proposed Cornell as the next modern source under "research-use precedent" — the current decision rejects it outright.)* |

## `gutenberg_extractor.py`

```
python data/scripts/gutenberg_extractor.py --plays hamlet macbeth caesar canterbury malory [--min-quality N] [--max-pairs N]
```

Idempotent — skips any play already present in the dataset's `metadata.sources`. Three parsing modes, dispatched by source:

- **Play mode** (hamlet, macbeth, caesar): regex-matches `SPEAKER. line` cues, pairs consecutive turns from different speakers as `(input, output)`, maps speaker abbreviation → archetype via `ARCHETYPE_MAP`.
- **Poem mode** (canterbury, and any future source added to `POEM_SOURCES`): Chaucer's *Canterbury Tales* has no speaker cues — it's a frame narrative where pilgrims tell tales and quoted dialogue is embedded in verse. The extractor:
  1. Locates each `"THE X'S TALE"` header to build a line-number → tale-teller map.
  2. Extracts all `“...”` quoted spans, skipping anything before the first tale header (title page, preface, table of contents — these produced garbage on the first run and are explicitly excluded).
  3. Assigns archetype from the enclosing tale-teller via `TALE_TELLER_ARCHETYPE` (e.g. Knight → noble, Miller → peasant, Pardoner → clergy).
  4. Pairs consecutive quotes as `(input, output)`, dropping fragments (<3 words), run-ons (>60 words), and mojibake.
- **Tagged mode** (malory, and any future source added to `TAGGED_SOURCES`): this edition of *Le Morte Darthur* has **no quotation marks whatsoever** — dialogue is only marked by an inline `<clause>, said <name>` tag (genuine Early Modern English convention, not an OCR artifact). The extractor:
  1. Bounds extraction between the first real chapter header (`CHAP. I.`) and the glossary/index, skipping the ~4000-line table of contents and front matter.
  2. Splits the body into clauses on `.`/`;` boundaries.
  3. Searches each clause for a dialogue tag via `DIALOGUE_TAG`, whose speaker capture is restricted to titles (`king`, `sir`, `dame`, etc., optionally + a proper name) or a bare capitalized proper name — **not** arbitrary lowercase words. (First version was too permissive and matched idioms like "she said so they departed" as if "so they departed" were a speaker name — silently corrupting ~40% of extracted lines. Restricting to title/proper-noun patterns fixed it; verify with a sample if you touch this regex.)
  4. Splices out the tag, reassembling the clause into a clean utterance, and maps archetype via `MALORY_ARCHETYPE_MAP` (e.g. "sir" → guard, "king"/"queen"/"duke" → noble, "merlin"/"hermit"/"bishop" → scholar/clergy).
  5. Drops clauses containing `_..._` (Gutenberg italic markers — chapter-title fragments, not dialogue).

`quality_score()` is a cheap heuristic (length, dialect-marker presence, word count) — not a substitute for manual review. `--min-quality` gates it, `--max-pairs` caps output (sorted by score, highest first) so a single dense source doesn't blow past the target archetype distribution.

**Gotchas hit so far:**
- **Canterbury:** the initial version had no lower bound on where quote-scanning started, so it pulled 1963 "pairs" out of front-matter and footnotes (single words, TOC fragments). Fixed by bounding extraction to start at the first tale header.
- **Malory:** see tagged-mode point 3 above — the regex over-match issue. Also, only the *first* `said X` tag per clause is stripped, so a handful of multi-speaker clauses leave a stray embedded tag in the output text (not worth chasing further; flagged for `dataset_validator.py`).

If you extend `POEM_SOURCES` or `TAGGED_SOURCES` to a new text, sanity-check a sample of the first ~10-15 extracted pairs before trusting the count — both prior sources produced silently-wrong output on the first attempt.

## `soda_extractor.py` (active — the current modern-city source)

```
python data/scripts/soda_extractor.py --per-archetype 75              # report only
python data/scripts/soda_extractor.py --per-archetype 75 --merge      # also merge into modern_npc_dataset.json
```

Source: `allenai/soda` on HuggingFace (CC BY 4.0), `train.parquet` (~1.19M dialogues, 817MB). Each row's `speakers` array names the actual characters in that narrative-derived dialogue (e.g. `["Veda", "Priest"]`) — `ARCHETYPE_PATTERNS` is a regex per archetype (role vocabulary: "police officer", "cop", "policeman", "policewoman" for police officer, etc.) matched directly against those names, not against a domain tag. For each row, the first turn spoken by a matching-named speaker (that also has a preceding turn from someone else) becomes the `(input, output)` pair — one pair per dialogue, to avoid oversampling one narrative.

Verified real match counts before building anything (`grep`-style regex scan over just the `speakers` column, fast): `executive 50252, professor 18425, shopkeeper 16510, service worker 11681, social worker 9920, police officer 8606, bartender 1758, pharmacist 99` — comfortably enough headroom for 75 per archetype even on the thinnest one.

**Two real bugs caught on manual spot-check** (not assumed correct from the count alone):
1. Bare `officer` in the police-officer pattern matched non-police roles like "Ski Patrol Officer" and "Loan Officer". Narrowed to `police officer|cop|policeman|policewoman`.
2. SODA embeds inline third-person stage directions in dialogue text, e.g. `"(He looks at the prescription.)"`. Added `_clean()` — strips `(...)` spans and collapses whitespace — applied before the word-count quality filter (`MIN_WORDS=3, MAX_WORDS=60`), same bounds `gutenberg_extractor.py` uses.

Not a rewrite pass like `chimbiwide_converter.py`'s `register_rewrite()` — SODA's raw text is already in the right contemporary register when the speaker's own name already matches the target role, so this is extraction + relabeling. Each entry's `metadata.conversion_note` says this explicitly, and `metadata.provenance` records the source row's `original_index` for traceability.

Last run (`--per-archetype 75 --merge`): 600/600 entries merged (`SOD-0001`..`SOD-0600`), 1 negligible duplicate pair (identical generic greeting from two different narratives — not deduplicated further, harmless).

## `chimbiwide_converter.py` (archived — medieval only, not part of the modern pipeline)

```
python data/scripts/chimbiwide_converter.py --limit 300                       # filter + rewrite, report only
python data/scripts/chimbiwide_converter.py --limit 300 --merge --max-entries 150   # also merge into the dataset
```

Source: `chimbiwide/NPC-Dialogue_v2` on HuggingFace, config `dialogue` (not the default — must be passed explicitly or `load_dataset` errors). Requires `pip install datasets`.

Row schema is a `messages` list, not `input`/`output`:
```
messages[0]  = user   — roleplay setup prompt: "You are <Name>. Background: ... Current Location: ..."
messages[1]  = assistant — opening greeting (not paired with a preceding player input)
messages[2:] = alternating user/assistant — real dialogue turns
```
`parse_row()` extracts the character name and `Background:` blurb via regex, then pairs every adjacent `(user, assistant)` message from index 1 onward. `remap_archetype()` keyword-matches the background text (e.g. "bounty hunter" → guard, "smuggler" → merchant) since the source has no explicit archetype tag.

`is_medieval_plausible()` drops rows containing modern-leakage terms (phone, internet, wifi, police, rupees, etc.) as a first-pass filter — most source content is contemporary-adjacent fantasy/noir, not medieval, and needs a register rewrite regardless of passing this filter.

**`register_rewrite()` is a rule-based rewriter, not an LLM call.** Deterministic, no API key, no cost: contraction expansion (`don't` → `do not`, `I'm` → `I am`), `you`/`your`/`yours` → `thou`/`thy`/`thine`, irregular thou-verb fixups both directions (`thou are` → `thou art`, and inverted questions `are thou` → `art thou`), sentence-start capitalization, and a small modern-vocabulary swap list (`okay` → `aye, it is well`, `police` → `watchmen`, etc.). It is intentionally modest — grammar isn't perfect (e.g. `you` collapses to `thou` regardless of subject/object case, so some object-position uses that should be `thee` come out as `thou`), and it does not fix genre/setting mismatches (a few entries still read as noir/frontier rather than medieval fantasy, just with archaic-flavored grammar). Every merged entry is tagged `register_rewritten` with `quality_score: 5` so it's flagged for review, not presented as gold-quality.

`build_entries()` takes only the *first* dialogue pair per conversation (avoids oversampling one character/scene across a long roleplay chat into many near-duplicate entries). `--max-entries` caps the final merge count.

Last run (`--limit 300 --merge --max-entries 150`): 255 medieval-plausible / 45 dropped / 0 unparseable → 150 merged as `CHM-####` entries, `source: chimbiwide`.

## `gpt4o_augmentor.py`

```
python data/scripts/gpt4o_augmentor.py --dry-run                                    # gap report only, no API calls
python data/scripts/gpt4o_augmentor.py --archetype guard --count 20                 # fill one archetype
python data/scripts/gpt4o_augmentor.py --all --limit-per-archetype 10               # gap-fill pass, all archetypes
```

Reads current archetype counts from the processed dataset, diffs against `ARCHETYPE_TARGETS` (from `Specs.md` section 6), and calls GPT-4o to generate schema-conformant entries for the gap. Requires `OPENAI_API_KEY` — costs money per call, hence `--dry-run` exists and `--limit-per-archetype` defaults small. **Superseded in practice** — twelve hand-authored batches (227 entries, `source: synthetic_claude`) closed the same archetype gaps at zero API cost. Kept in the repo in case `OPENAI_API_KEY` becomes available later.

## `dataset_validator.py`

```
python data/scripts/dataset_validator.py            # full report
python data/scripts/dataset_validator.py --strict    # exit 1 on schema errors (CI use)
```

Checks required fields per schema v1.0 (top-level, `persona`, `context`, `linguistic_markers`, `metadata`), valid enum values (archetype, disposition, intent), duplicate ids, and duplicate input/output pairs — these are hard errors. Also flags (non-fatal warnings) known extraction artifacts: mojibake, suspiciously short outputs, and the documented Malory stray-dialogue-tag issue (`STRAY_TAG_PATTERN` on `GUT-*` ids). Prints an archetype-balance report against `Specs.md`'s target table and a source breakdown.

Last run: 1003/1003 entries valid, 0 schema errors, 19 warnings (all the known Malory artifact, ~6% of the 300 Malory entries — nothing new).

## `stress_test_corpus.json`

Not a script output — hand-authored directly, 50 entries, **held out, never used in training**. Structurally different from the main dataset: no pre-written NPC output, since the point is to run each conversation's player-side turns against whichever condition (A/B/C/D) is under test and see where the persona breaks (`Specs.md` section 8: PDM > 0.7 and no archaic markers present = break).

Schema: `{"id": "STRESS-####", "archetype": "...", "stress_test_type": "...", "turns": ["...", "..."]}`.

Type breakdown (matches `Specs.md` exactly): 15 `identity_challenge` ("are you an AI?"), 15 `out_of_world_reference` (WiFi, Netflix, credit cards), 12 `modern_language` (slang the NPC must survive in-voice), 8 `extended_pressure` (10-11 turns each, one per archetype, escalating from normal conversation into direct AI-identity pressure and back).

## Not yet built

- **`persona_features[]` and visibility-set annotation** on the 659 existing modern entries — not started. `persona_features[]` needs PDM v2's feature families first (Week 5); visibility sets need KBD's schema decided (Week 6). Populating these early would repeat the "calibrated before an adapter/metric exists" mistake already flagged for PDM v2 itself.
- **Taskmaster and MultiWOZ extraction** — downloaded, not yet used. Optional at this point since SODA alone reached the 600-pair target; worth doing for register diversity once Weeks 5–6 can actually measure whether the current set is thin anywhere.
- **A modern-city equivalent of `stress_test_corpus.json`** — the existing 50-entry corpus is medieval-archetype-only (`archetype` field uses guard/merchant/etc, not the modern 8). Needed before the modern adapters can be stress-tested the same way the medieval ones were.
- **Planted-forbidden-facts extension** to `stress_test_corpus.json` — 15–20 probes whose correct response is a refusal or expression of ignorance, required for KBD to have signal.

### Dead ends / rejected sources (do not retry)

- `microsoft/crd3` filter pass — dead end. HF dropped script-based dataset loading and CRD3 has no Parquet conversion. (Full reasoning in `Docs/TODO.md`.)
- **SGD** — rejected on licence grounds (CC BY-SA 4.0 share-alike, *not* the Apache 2.0 the brief assumed). Clone deleted. See [rejected sources](#source-datasets--rejected-licence-diligence) above.
- **PersonaChat** — HF mirrors declare no licence; not downloaded. Covered by Synthetic-Persona-Chat. See [rejected sources](#source-datasets--rejected-licence-diligence) above.
- **Cornell Movie-Dialogs** — rejected on licence grounds (no formal open licence). See [rejected sources](#source-datasets--rejected-licence-diligence) above.
- **DailyDialog** — rejected on licence grounds (CC BY-NC-SA 4.0: NC + share-alike). See [rejected sources](#source-datasets--rejected-licence-diligence) above.

> **`load_dataset` caveat:** for CRD3 (and SGD's script-based format) HF no longer supports script loading and no Parquet exists — that path is dead. Taskmaster/MultiWOZ are therefore taken as raw GitHub JSON. SODA and Synthetic-Persona-Chat *do* ship Parquet, so `huggingface_hub.snapshot_download` works for them directly.
