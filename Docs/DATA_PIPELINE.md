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
│   │   └── canterbury.txt   # Gutenberg #2383 (Purves ed.)
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

326 entries in `data/processed/medieval_npc_dataset.json`:

| Source | Pairs | Method |
|--------|------:|--------|
| Hamlet | 60 | Play speaker-cue extraction |
| Julius Caesar | 45 | Play speaker-cue extraction |
| Macbeth | 21 | Play speaker-cue extraction |
| Canterbury Tales | 200 | Frame-narrative quote extraction |

Archetype distribution: clergy 123, noble 88, peasant 63, merchant 23, scholar 18, guard 6, innkeeper 5. Guard/scholar/merchant/innkeeper/herbalist are badly underrepresented relative to the target table in `Specs.md` — that's what `gpt4o_augmentor.py` and the CRD3 filter pass are for.

## `gutenberg_extractor.py`

```
python data/scripts/gutenberg_extractor.py --plays hamlet macbeth caesar canterbury [--min-quality N] [--max-pairs N]
```

Idempotent — skips any play already present in the dataset's `metadata.sources`. Two parsing modes, dispatched by source:

- **Play mode** (hamlet, macbeth, caesar): regex-matches `SPEAKER. line` cues, pairs consecutive turns from different speakers as `(input, output)`, maps speaker abbreviation → archetype via `ARCHETYPE_MAP`.
- **Poem mode** (canterbury, and any future source added to `POEM_SOURCES`): Chaucer's *Canterbury Tales* has no speaker cues — it's a frame narrative where pilgrims tell tales and quoted dialogue is embedded in verse. The extractor:
  1. Locates each `"THE X'S TALE"` header to build a line-number → tale-teller map.
  2. Extracts all `“...”` quoted spans, skipping anything before the first tale header (title page, preface, table of contents — these produced garbage on the first run and are explicitly excluded).
  3. Assigns archetype from the enclosing tale-teller via `TALE_TELLER_ARCHETYPE` (e.g. Knight → noble, Miller → peasant, Pardoner → clergy).
  4. Pairs consecutive quotes as `(input, output)`, dropping fragments (<3 words), run-ons (>60 words), and mojibake.

`quality_score()` is a cheap heuristic (length, dialect-marker presence, word count) — not a substitute for manual review. `--min-quality` gates it, `--max-pairs` caps output (sorted by score, highest first) so a single dense source doesn't blow past the target archetype distribution.

**Gotcha hit during the first canterbury run:** the initial version had no lower bound on where quote-scanning started, so it pulled 1963 "pairs" out of front-matter and footnotes (single words, TOC fragments). Fixed by bounding extraction to start at the first tale header. If you extend `POEM_SOURCES` to a new text, sanity-check a sample of the first ~10 extracted pairs before trusting the count.

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
- Malory (*Le Morte d'Arthur*) raw text — spec roadmap mentions Chaucer *and* Malory; only Chaucer is downloaded.
- `microsoft/crd3` filter pass — not started.
- `stress_test_corpus.json` (50 persona-breaking conversations) — not started, and per spec is held out from training entirely.
