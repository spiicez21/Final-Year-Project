# NPC dialogue architecture

How one line a player types in the game becomes one NPC reply, and why it is
built this way. Code: `backend/dialogue/`. Server: `backend/gguf_server.py`
(HTTP, model pool, voices, scoring — nothing about *what* an NPC says).

## Why it was rebuilt

Features had been added one at a time — priming turns, fact demonstrations,
player memory, fact retrieval, two refusal retries — and each put its own
turns into the prompt on every message. Nothing coordinated them and nothing
looked at the reply. Played the way people actually type, it failed in the
ways a player notices first:

| What the player saw | Cause |
|---|---|
| NPC keeps asking your name after you gave it | "myself Yuga, 3rd year cse" saved **nothing**: names had to be capitalised |
| "You told me your name was Emily." | the recall retry fired on *any* memory, not the slot asked for |
| "what do u do" → "I don't do anything." ×3 on a return visit | memory was shown before *every* message and crowded out the NPC's job; "u" matched no demonstration |
| the same line three times in a row | a bad line in the transcript is copied by greedy decoding |
| head of department answers a project question with his salary | retrieval matched on "year"; "i'm" was split into a stray "m" that matched every "I'm …" fact |
| "1. Research your topic… 2. … 5." | nothing shaped the reply for speech |

A second round of play showed the limit of fixing understanding with patterns.
After "my name is yugabharathi", the line **"i am class cse d" renamed the
player "Class"** — and "i am tamil", "i am hosteller", "i am cr of cse d" saved
"Tamil", "Hosteller", "Cr". Every pattern fix needed a word list, and every
word list had a hole. So understanding is now **learned** (see below); rules
remain only as checks on what the models produce.

## The pipeline

```
player line
   │
   ▼
understand   intent.classify  learned heads over a sentence encoder:
                              greeting · farewell · ack · recall · about_npc · question · statement
                              + for recall, which remembered fact is asked about
   │
   ▼
learn        extractor.extract  fine-tuned GLiNER span model, NPC's last line as context
             memory.merge       name, year, department, section, college, hometown,
   │                            project, interests, feeling — with per-slot confidence
   │                            (before generation: "myself Yuga" can be answered by name)
   ▼
retrieve     knowledge.select           the 1–2 NPC facts this *question* needs (questions only)
   │
   ▼
compose      composer.compose
             ┌──────────────────────────────────────────────┐
             │ system: persona, background, guests, memory   │  identical every turn
             │ priming turns + the game's fact demos         │  → stays in llama.cpp's cache
             ├──────────────────────────────────────────────┤
             │ transcript (normalised, repeated lines removed)│
             │ ONE late turn, chosen by intent:              │  the only part that
             │   recall / returning greeting → memory        │  changes per message
             │   question with facts found   → those facts   │
             │ the player's line                             │
             └──────────────────────────────────────────────┘
   │
   ▼
generate     injected Generator          llama.cpp in the server, a scripted fake in tests
   │
   ▼
check        guard.clean   → guard.detect → at most ONE regeneration → guard.finish
```

`turn.run_turn` runs the stages and owns no rules. Every stage is a module
with its own unit tests (`backend/test_dialogue.py`, no model needed).

### Design rules

1. **One late turn.** On a 1.1B model the exchange just before the question
   dominates the reply (memory: 14/40 early → 21/40 late). That is why only
   one thing may go there, chosen by what the player is doing.
2. **Stable prefix.** Everything before the transcript is the same on every
   turn, so llama.cpp reuses it. Putting per-question facts in the system
   prompt instead took median generation from 0.62 s to 2.42 s.
3. **Models read; rules check.** What the player *means* — which facts they
   stated, what they are asking — is decided by small learned models, because
   patterns cannot keep up with how people type. Rules stay where they are
   checks on a model's output (the reply guard, the confidence threshold), never
   the source of a fact. A missed fact means *forgetting*, never inventing:
   every saved value is a span of what the player actually typed.
4. **Show, don't tell.** The persona is taught by demonstration turns, not
   instructions (a long rule list produced "I don't have a job, I'm just a
   machine").
5. **Check every reply; repair once.** One extra generation at most, so a bad
   turn costs ~0.3 s, not seconds. Restarts give a *frame* ("You told me…",
   the job line's first words), never the answer.
6. **What the model can see, it copies.** Loops are removed from the replayed
   transcript, and a repeat is regenerated with the copied exchange hidden. A
   repetition penalty did nothing (llama.cpp penalises only the last 64
   tokens; 0/4 repeats broken at 1.3–2.0, 4/4 by hiding the source).
7. **The evaluation prompt is untouched.** `/chat` without a persona uses
   `persona.eval_messages` and `guard.clean_basic` only — byte-identical to
   `evaluation/run_*.py`, so the paper's numbers stay reproducible.

## Understanding: two learned components

Both run on one model, `training/extractor/player_facts_gliner/` (GLiNER small,
166M parameters, ~600 MB, CPU): the span model extracts facts, and its
DeBERTa encoder also feeds the intent heads, so there is no second model in
memory.

**Fact extractor** (`extractor.py`). Scores every span of the player's line
against plain-English labels — "student name", "class section", "department
or branch". Nothing describes what a name or a section *looks like*; the model
was fine-tuned on 6,000 generated student lines (`training/extractor/`) whose
names, towns, colleges, departments, projects and section letters are
**disjoint from every evaluation line** — the generator refuses to run
otherwise. So "i am class cse d" is read correctly by structure, not because
it has seen "cse".

- **Context.** The NPC's previous line is prepended, so "yuga" after "What's
  your name?" is a name; spans inside the NPC's line are ignored.
- **Confidence.** Each saved slot keeps the model's score. A later reading with
  a different value replaces it only if it is nearly as confident, so a clear
  "my name is yugabharathi" survives a hesitant misreading, while "actually my
  name is bharathi" still corrects it. Slots saved by the old pattern rules have
  no confidence and are replaced by the first model reading.

**Intent** (`intent.py`). Two linear heads over the encoder's mean-pooled
sentence vector, trained on 4,500 generated lines: one picks the intent, one
picks which remembered facts a recall question asks about. A predicted slot
only narrows the answer when that fact is actually remembered — otherwise
the NPC treats it as "what do you know about me", so a slot mistake costs
precision, never a wrong answer. The pattern classifier is kept solely as the
fallback when no trained head is on disk.

## News: reports that spread

Players don't only talk about themselves. "i saw a person with a weapon
roaming around" is news about the world, and it is handled as such
(`events.py`, game side `Systems/NPC/world_event_store.gd`):

1. **Understand.** The intent classifier has a `report` class, and the
   extractor has two event labels, *incident* and *incident location*, trained
   on generated report lines. An incident is never saved as a fact about the
   player: `extractor.read()` returns facts and events separately.
2. **Store.** The server returns `reported_event` (`what`, `where`); the game's
   event log records it, with the NPC who was told as its first listener.
3. **Spread.** Every `news_spread_seconds` (45 s) one NPC who knows an event
   tells one who doesn't. Nobody is special, not even the police officer: news
   travels round the event the way gossip does.
4. **Use.** Each request carries the events that NPC has heard. They join the
   retrieval pool, so "is anything happening?" can retrieve them like campus
   facts, and a greeted NPC with fresh news passes it on (the late turn, as
   always one exchange). The reply that passes news on returns `shared_event_id`
   so the game marks it told.
   On its own the 1.1B model ignored that turn — 0/9 shared on greeting, 0/9
   answered, with the right line in the prompt every time — so the guard checks
   for it (`news_miss`) and restarts on the line's opening words. The model then
   completes the event itself; the authored line is only a last resort.
   Incident spans use the same 0.7 threshold as facts (a lower one made "the
   canteen food was bad today" an incident). When a line is classified as a
   report but no incident span clears it, the player's words are kept and quoted.
5. **Measure.** Each request also carries the events the NPC has *not* heard —
   never in the prompt, only so a reply that mentions one is flagged in
   `event_leaks`. This is Knowledge Boundary Drift for a visibility set that
   grows at runtime, the information-diffusion case in the paper.

Commands for testing: `/news`, `/forget news`. The HUD shows **HEARD NEWS** (with
who each item came from) and a **news leaks** metric.

## The reply guard

| Problem | Seen in play | Repair |
|---|---|---|
| `list` | "1. Research your topic…" | flatten to one sentence, cap at two sentences |
| `asks_known` | "Hi Yuga, what's your name?" | drop that question; if nothing else was said and the name was just given, restart as "Nice to meet you, Yuga." |
| `recall_miss` | "That's not something I can tell you" / "Yes, I remember you." to "u remember my name?" (name saved) | restart as "You told me" — only if the asked slot is saved |
| `invented_name` | "You told me your name was Emily." | saved name: substitute it; nothing saved: restart as "I don't think you've told me" |
| `fact_refusal` | "I'm not sure" with the answer retrieved | restart on the fact's first two words; if it still refuses, say the fact itself (authored, reported as `fact_fallback`) |
| `self_denial`, `wrong_identity` | "I don't do anything." / the police officer: "I'm a librarian." | restart on the job line's first two words |
| `repeat` | the same line as an earlier reply, to a different message | regenerate with that exchange hidden from the transcript |
| `news_miss` | "hi, what's going on?" from an NPC with fresh news; a lecture invented for "is anything happening?" with the heard report retrieved | restart on the news line's opening ("Have you heard? Halvorsen told me a student reported"); if the event is still missing, say the line (`news_fallback`) |
| `report_refusal` | "That's not allowed." to "there's a kid bleeding near the court" | restart as "Thanks for telling me." |

The HUD's metrics panel shows the intent and any repairs for every reply
(`understood as`, `reply guard`), so a tester can see why a line came out as
it did.

## Contract with the game (unchanged)

The Godot client sends the same `ChatRequest` as before (persona fields,
`facts`, `fact_demos`, `history`, `player_memory`) and keeps doing what it did:
one memory record per NPC, saved to `user://npc_memory.json` whenever the
response reports `memory_updates`. New response fields — `intent`, `problems`,
`repairs`, `generations` — are informational. Saved memory gains a
`_confidence` map (hidden in the HUD's text, shown there as a dimmed percentage
per fact) and the slots `studies`/`from` became `department`/`hometown`; old
save files are migrated on the next merge.

## Measured

CONVERSATION_RESULTS

## Changing it

- **A new kind of player line** → an intent in `training/extractor/make_intent_data.py`,
  retrain (`train_intent.py`, about a minute), add labelled lines to
  `evaluation/intent_cases.json` *before* measuring.
- **Something new to remember** → a label in `extractor.LABELS` and the data
  generator (they must match; a test checks), templates and values for it,
  retrain (`train_extractor.py`, ~3 min on the laptop GPU), and `render` /
  `about` / `recall_line` in `memory.py`. Add labelled lines to
  `evaluation/memory_extraction_cases.json` first, including lines where it
  must *not* be learned.
- **A new failure seen in play** → reproduce it in
  `evaluation/run_conversations.py`, add detection and one repair to
  `guard.py`, and a scripted test in `test_dialogue.py`. Keep it to one
  regeneration.
- **Anything that adds to the prompt** → it goes through `composer.py` and must
  not add a second late turn.
