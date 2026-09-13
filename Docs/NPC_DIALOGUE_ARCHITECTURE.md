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

## The pipeline

```
player line
   │
   ▼
understand   text.normalize   "what do u do" → "what do you do"
             intent.classify  greeting · farewell · ack · recall · about_npc · question · statement
   │
   ▼
learn        memory.extract + merge     name, year, studies, from, project, interests, feeling
   │                                    (before generation: "myself Yuga" can be answered by name)
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
3. **Rules, not a second model, for understanding.** Intent, extraction and
   retrieval are regular expressions and word overlap: no extra latency,
   inspectable, unit-tested, and a miss means *forgetting*, never inventing.
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

The HUD's metrics panel shows the intent and any repairs for every reply
(`understood as`, `reply guard`), so a tester can see why a line came out as
it did.

## Contract with the game (unchanged)

The Godot client sends the same `ChatRequest` as before (persona fields,
`facts`, `fact_demos`, `history`, `player_memory`) and keeps doing what it did:
one memory record per NPC, saved to `user://npc_memory.json` whenever the
response reports `memory_updates`. New response fields — `intent`, `problems`,
`repairs`, `generations` — are informational.

## Measured

CONVERSATION_RESULTS

## Changing it

- **A new kind of player line** → `intent.py`, plus a test in `test_intent`.
- **Something new to remember** → a slot in `memory.py` (`extract`, `render`,
  `about`), plus cases in `EXTRACT` *and* `NOT_LEARNED`.
- **A new failure seen in play** → reproduce it in
  `evaluation/run_conversations.py`, add detection and one repair to
  `guard.py`, and a scripted test in `test_dialogue.py`. Keep it to one
  regeneration.
- **Anything that adds to the prompt** → it goes through `composer.py` and must
  not add a second late turn.
