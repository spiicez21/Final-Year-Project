# Data Pipeline

Documents `data/` — how it's structured, what each script does, and how to run the pipeline. Schema reference: `DevFiles/Specs.md` section 6.

## Directory layout

```
data/
├── raw/
│   ├── gutenberg/           # Plain-text public domain sources
│   │   ├── hamlet.txt       # Gutenberg #1524
│   │   ├── macbeth.txt      # Gutenberg #1533
│   │   ├── caesar.txt       # Gutenberg #1785
│   │   ├── canterbury.txt   # Gutenberg #2383 (Purves ed.)
│   │   └── malory.txt       # Gutenberg #46853, Le Morte Darthur (Rhys ed.)
│   └── huggingface/         # HF `datasets` cache (gitignored-worthy, large)
│
├── processed/
│   ├── medieval_npc_dataset.json   # Main training dataset, schema v1.0
│   └── stress_test_corpus.json     # 50 held-out persona-breaking conversations (NOT for training)
│
└── scripts/
    ├── gutenberg_extractor.py      # Play + poem + tagged-dialogue extraction
    ├── chimbiwide_converter.py     # HF chimbiwide/NPC-Dialogue_v2 filter + rewrite
    ├── gpt4o_augmentor.py          # Gap-fill synthesis for underrepresented archetypes (superseded by hand-authored batches)
    └── dataset_validator.py        # Schema conformance, duplicate detection, archetype balance report
```

Python interpreter used for all of the above: `C:\Users\spicez\AppData\Local\Programs\Python\Python310\python.exe` (system 3.13 is broken on this machine — see [TODO.md](TODO.md)).

## Current dataset state

**1,003 entries** in `data/processed/medieval_npc_dataset.json` — 1,000-entry spec target met (2026-07-04):

| Source | Pairs | Method |
|--------|------:|--------|
| Hamlet | 60 | Play speaker-cue extraction |
| Julius Caesar | 45 | Play speaker-cue extraction |
| Macbeth | 21 | Play speaker-cue extraction |
| Canterbury Tales | 200 | Frame-narrative quote extraction |
| Le Morte Darthur (Malory) | 300 | Inline dialogue-tag extraction (no quotation marks in this edition) |
| chimbiwide/NPC-Dialogue_v2 | 150 | HF filter + rule-based archaic rewrite (no LLM) |
| Hand-authored (Claude, in-session, no API cost) | 227 | Direct schema-conformant writing across 12 batches, targeted at worst archetype gaps |

Final archetype distribution: peasant 221, guard 189, noble 182, clergy 123, scholar 100, merchant 87, innkeeper 63, herbalist 38. Merchant/scholar/innkeeper/herbalist are still under the per-archetype target table in `Specs.md` (noble/peasant/clergy are over target) — the 1,000-count milestone is met, but true archetype balance would need several more hundred entries skewed toward those four. Revisit if Phase 4 evaluation shows weak persona consistency for those archetypes specifically.

Archetype distribution: peasant 221, guard 187, noble 182, clergy 123, scholar 45, merchant 28, innkeeper 12, herbalist 5. Merchant/scholar/innkeeper/herbalist are now the only real gaps relative to the target table in `Specs.md` — none of the sources mined so far (Shakespeare, Chaucer, Malory, chimbiwide) produce many of those characters. Closing them needs targeted hand-authored entries or CRD3.

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

## `chimbiwide_converter.py`

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

- `microsoft/crd3` filter pass — dead end, not pursued. See `Docs/TODO.md` Phase 2 for the full reasoning (HF dropped script-based dataset loading, no Parquet conversion exists for CRD3).
