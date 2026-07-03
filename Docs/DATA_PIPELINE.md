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
│   └── medieval_npc_dataset.json   # Main training dataset, schema v1.0
│
└── scripts/
    ├── gutenberg_extractor.py      # Play + poem dialogue extraction
    ├── chimbiwide_converter.py     # HF chimbiwide/NPC-Dialogue_v2 filter + rewrite
    └── gpt4o_augmentor.py          # Gap-fill synthesis for underrepresented archetypes
```

Python interpreter used for all of the above: `C:\Users\spicez\AppData\Local\Programs\Python\Python310\python.exe` (system 3.13 is broken on this machine — see [TODO.md](TODO.md)).

## Current dataset state

653 entries in `data/processed/medieval_npc_dataset.json`:

| Source | Pairs | Method |
|--------|------:|--------|
| Hamlet | 60 | Play speaker-cue extraction |
| Julius Caesar | 45 | Play speaker-cue extraction |
| Macbeth | 21 | Play speaker-cue extraction |
| Canterbury Tales | 200 | Frame-narrative quote extraction |
| Le Morte Darthur (Malory) | 300 | Inline dialogue-tag extraction (no quotation marks in this edition) |
| Hand-authored (Claude, in-session) | 27 | Direct schema-conformant writing, targeted at worst gaps |

Archetype distribution: guard 162, peasant 160, clergy 123, noble 132, scholar 33, merchant 28, innkeeper 10, herbalist 5. Merchant/innkeeper/herbalist are now the worst gaps relative to the target table in `Specs.md` — Gutenberg literary sources structurally don't have many of those characters, so closing them needs hand-authored entries, CRD3, or chimbiwide (once its register-rewrite is implemented).

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
python data/scripts/chimbiwide_converter.py --limit 300
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

**`register_rewrite()` intentionally raises `NotImplementedError`.** It needs an LLM backend (GPT-4o via `gpt4o_augmentor.py`'s client, or local) to convert modern-register dialogue into archaic NPC voice, and `Specs.md` flags this source as medium IP risk requiring a scrub pass before any entries are merged or published. Do not remove the guard without implementing the rewrite.

Last run (`--limit 300`): 255 medieval-plausible / 45 dropped / 0 unparseable. Stops before merge — nothing from chimbiwide is in `medieval_npc_dataset.json` yet.

## `gpt4o_augmentor.py`

```
python data/scripts/gpt4o_augmentor.py --dry-run                                    # gap report only, no API calls
python data/scripts/gpt4o_augmentor.py --archetype guard --count 20                 # fill one archetype
python data/scripts/gpt4o_augmentor.py --all --limit-per-archetype 10               # gap-fill pass, all archetypes
```

Reads current archetype counts from the processed dataset, diffs against `ARCHETYPE_TARGETS` (from `Specs.md` section 6), and calls GPT-4o to generate schema-conformant entries for the gap. Requires `OPENAI_API_KEY` — costs money per call, hence `--dry-run` exists and `--limit-per-archetype` defaults small. Not yet run against the live API in this repo.

## Not yet built

- `dataset_validator.py` — schema conformance, duplicate detection, archetype/intent balance report. Referenced in `Specs.md`'s file tree but doesn't exist yet.
- `microsoft/crd3` filter pass — not started.
- `stress_test_corpus.json` (50 persona-breaking conversations) — not started, and per spec is held out from training entirely.
