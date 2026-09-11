# NPC dialogue layer

Talkable campus NPCs whose replies are generated live by the project's
per-archetype LoRA models, with the paper's drift metrics shown per turn.

## The scenario

It is the **Computer Science department open day**. Five staff and invited
guests are in the ground-floor corridor to meet students, answer questions
about themselves, and talk about what their job is actually like.

| NPC | Archetype | Who they are |
|---|---|---|
| Officer Reyes | `police officer` | Campus liaison officer with the city police, invited to talk about staying safe on campus |
| Prof. Adeyemi | `professor` | Lecturer in the department, here to talk about the modules they teach |
| Halvorsen | `executive` | Head of department, hosting the open day |
| Ms. Okafor | `social worker` | Student counsellor from the welfare office |
| Nadia | `shopkeeper` | Runs the campus store, has a stall at the event |

Each NPC also carries a **background** (career history, what they care about)
and knows **who else is at the event**, so they can hand a question on rather
than dead-ending it:

> **you:** Who do I talk to about lost property?
> **Prof. Adeyemi:** You should talk to the campus liaison, Reyes.

## Campus knowledge

`campus_facts.gd` holds what the NPCs know about the department: student and
staff numbers, which floor holds what, term dates, the open-day timetable, and
per-NPC role detail — modules taught, caseload, shift pattern, pay.

Pay is included deliberately. It is one of the first things people ask about a
job, and each NPC knows only their own — nobody carries anyone else's figure,
so asking the shopkeeper what the head of department earns should not produce
a number.

> **you:** What do you earn?
> **Prof. Adeyemi:** I'm on grade 8, so about 52,000 a year.

All figures are **invented for the demo**. Plausible for a UK department, not
real institutional data.

### This is not the KBD knowledge base

`data/processed/knowledge_base.json` is the scoring artifact for Contribution
C1. Seven evaluation scripts read it, its 39 signature phrases decide what
counts as a factual reference, and the modern-city dataset froze on 6 Sep 2026.
Adding campus facts to it would change KBD for every banked result — alpha
sweep, Conditions B and C, hybrid BC, calibration, recheck — and quietly
invalidate them.

So the campus facts live in the game instead, and are written to be lexically
disjoint from those signature phrases. Verified rather than assumed: asking
about student numbers, pay, modules and floors returned `kbd: null` with zero
leaked facts every time, i.e. in-game KBD still measures exactly what it
measured before.

The adapters were never trained on any of this, so unlike the evaluation
knowledge base there is no Condition B claim to protect — injecting these is
the only way an NPC could know them.

### Facts alone were not enough

Putting the facts in the system prompt got **6 of 11** test questions answered
correctly. The failures were not random: numeric and pay questions hit the
adapters' refusal training, answering *"How many students are in the
department?"* with *"I don't have that information"* while the number sat in
the prompt a few lines above.

Two worked examples (`fact_demos` — one numeric question, one about pay) took
it to **8 of 11**. The last three came from fixing the authored text rather
than adding more prompt:

- `job_line` said "I teach two modules" while the facts named CS2011 and
  CS3040. The vaguer demonstration beat the specific fact. Making them agree
  fixed both that question and the supervision count.
- One pay phrasing was not enough — "what do you earn" worked, "What is your
  salary?" still returned "I don't have a salary." Same register-coverage
  lesson as the job questions.

Final: **11 of 11**, at a cost of **+55 ms** per warm turn (median 333 ms with
facts vs 278 ms without).

## Voices

Each NPC speaks its replies aloud in its own voice, synthesised locally by
[Piper](https://github.com/rhasspy/piper) (ONNX, CPU). Five distinct speakers,
one per NPC, assigned in `VOICE_BY_NPC` in `backend/tts.py`.

Fetch the models once — they are ~63 MB each, gitignored, and not
pip-installable:

```bash
.venv/Scripts/python.exe backend/fetch_voices.py
```

Use `backend/fetch_voices.py` rather than Piper's own downloader: Piper's does
not verify that the file it wrote is the size the index advertises, and on a
slow link that fails silently. We ended up with a 6.6 MB file where 63.1 MB was
expected, and the only symptom was `InvalidProtobuf: Protobuf parsing failed`
at load time several steps later. Ours checks the size and retries.

Set `voice_enabled = false` on the `NpcDirector` node to run silent. Speech is
also entirely optional at runtime: with no voices on disk the server reports
`speech: false` on `/health`, `/speak` returns 503, and the game logs one
warning and carries on with text only.

### Why speech is a second request, not part of `/chat`

`/chat` returns text; the game then calls `/speak`. Bundling them would hold
the subtitle back until synthesis finished and push every spoken line past the
project's 500 ms target. Split, the line appears at generation latency and the
voice follows:

| Stage | Time |
|---|---|
| Text on screen | 557 ms (generation) |
| Voice starts | 648 ms (+91 ms synthesis) |
| Audio length | 2810 ms |

Report those two separately. Do not add them together and call the sum text
latency — the RQ4 number is unchanged by speech.

Synthesis is far cheaper than playback: median **191 ms of compute for 4841 ms
of audio**, a real-time factor of **0.039**. Nothing here needs streaming at
NPC line lengths.

Two costs worth knowing. A voice takes ~1.3 s to load and its first synthesis
costs ~2 s of ONNX warm-up against a ~190 ms steady state, so the server loads
and warms every voice at startup — which is why boot now takes ~8 s. And audio
is returned as raw 16-bit mono PCM rather than WAV, because Godot builds an
`AudioStreamWAV` straight from samples and a header would only be parsed and
discarded.

Playback is positional (`AudioStreamPlayer3D` on the NPC), so a voice comes
from the character rather than from the HUD, and it stops when you leave the
conversation.

### Pauses and tone

Out of the box Piper made four specific mistakes on the lines these NPCs
produce, all found by probing rather than by ear:

| Input | What Piper did |
|---|---|
| `Prof. Adeyemi` | Two sentences — "prof." *pause* "Adeyemi" — the stop after the abbreviation ended the sentence mid-name |
| `Ms. Okafor` | "M. S." *pause* "Okafor" |
| `4pm... Ms. Okafor` | No pause at the ellipsis at all; the next sentence glued on |
| `CS2011` | Read as a year: "C S two thousand and eleven" |

`backend/tts.py` now normalises text before synthesis (`Professor`, `Miz`,
`C S 20 11`, `C S 10 oh 2`) and speaks one sentence at a time, inserting a
pause chosen by how the sentence ended: longer after a question than a
statement, longest on a trailing-off ellipsis. The on-screen line keeps
reading "Prof." and "CS2011"; only the audio is rewritten. `/speak` returns
the rewritten text as `spoken_text`, so a mispronunciation can be traced to
the text rather than guessed at.

Sentence gaps went from **115–290 ms** (Piper's default, which sounds
rushed) to **~300–670 ms**, and each NPC has its own `SpeechStyle` in
`STYLE_BY_NPC` — pace, pitch liveliness, rhythm variation and pause length.
Halvorsen is the slowest with the longest pauses; Nadia is brisk. These are
tuned by ear against each character, not measured against anything, so
change them freely.

One thing that looked equivalent and was not: the first version refused to
split a sentence after any digit, to protect decimals like `3.5`. That
silently removed the pause after every sentence ending in a number —
"I earn about 52,000. I love it." ran together — and pay and course-code
answers end in numbers constantly. Only a full stop *followed* by a digit is
now protected.

## Memory

Each NPC keeps its own conversation memory for the session. Walking away and
coming back continues the same conversation.

The client keeps 16 turns and the server replays 8 of them. When a
conversation outruns that, the server does **not** simply keep the most recent
8: it keeps the **opening exchange plus the most recent turns**, because the
first exchange is usually the one that established the subject. Ask four
questions about a project idea and plain truncation makes the fifth answer
forget what the project was.

What this buys, in practice:

> **you:** I'm thinking of doing a final-year project on compilers.
> **Adeyemi:** That's a great idea! What kind of compiler would you be working on?
> **you:** Do I need to know assembly?
> **Adeyemi:** No, you don't need to know assembly to do a final-year project on compiler design.

### What memory does *not* do

Context is used **implicitly**, as above. Asking the NPC to **recite** the
conversation back fails — all four phrasings tested ("Remind me what I said I
wanted to work on?", "What was my project idea again?", …) got a refusal or a
non-answer. That is a capability limit of a 1.1B model, not a missing history:
the turns are demonstrably in the prompt, since the same model uses them to
answer the next question. Do not demo the NPCs by asking them to recall.

## How the personas are built — and why it is done this way

Each NPC carries four first-person lines in `SPAWNS`: `occupation`, `intro`
(what they are doing here), `job_line` (what the job involves) and `background`
(how they got here, what they care about). The server seeds every conversation
with these as turns the NPC has **already answered**, then appends the real
question. Who else is at the event is passed too, which is what makes referrals
possible.

That is deliberate, and it replaced an approach that did not work. The first
attempt was the obvious one — a long system prompt full of rules ("always
answer questions about who you are", "never mention being an AI"). On
TinyLlama-1.1B it made things measurably worse:

| Question | Long rule-based prompt | Priming turns |
|---|---|---|
| "Who are you?" | *"I'm not allowed to tell you that."* | *"I'm Halvorsen, the head of the Computer Science department."* |
| "What is your job like?" | *"I don't have a job, I'm just a machine."* | *"I run the department, so timetables, hiring, budgets and a great many meetings."* |

Two reasons. The adapters were fine-tuned against the short evaluation prompt,
so a long structured instruction block is out-of-distribution for them; and a
1.1B model follows demonstrations far more reliably than negative rules, which
mostly serve to put the forbidden words into context.

The third priming pair (`job_line`) exists for a related reason: several
adapters were trained with hand-authored refusal examples, and without a
worked example of a *permitted* question about their own work they generalise
the refusal to it.

`event_line` answers "what is happening here" — a question about the *scene*
rather than the person. Without it each adapter answered from its own training
topic instead: asked what was happening at the open day, the police officer
replied *"we have received reports of a group of people breaking into the
computer science department."* KBD scored `null` on that reply, so it matched
no knowledge-base fact — a hallucination shaped by the archetype's topic prior,
**not** a visibility-set leak. Worth knowing which of the two you are looking at
before treating an odd line as a metric failure.

That one had a second cause worth recording: `job_line` for the officer
originally read "lost property, **the odd break-in**, and getting students home
safely". The demonstration was itself seeding the topic. Authored persona text
is part of the prompt — a stray word in it steers the model as surely as the
adapter does.

`background` had to be promoted from the system prompt to a demonstration for
exactly the same reason. Stated only in the system prompt it was largely
ignored — the professor whose background says "came back to teach after four
years in industry" answered *"Did you work outside academia?"* with *"No, I'm
not allowed to work outside academia."* As a demonstration it answers *"I've
worked in industry for the last four years."* Shown beats told, consistently,
on this model.

**Honest caveat.** Asking one of the three primed questions verbatim returns
something very close to the scripted line, because the model is continuing a
pattern it can see. Unprimed questions are genuinely generated — "how long
have you been doing that?" gets a different, role-appropriate answer from each
NPC, and out-of-scope questions are refused in each NPC's own voice.

## Running it

Two processes. Start the model server first, from the **repo root**
(`E:\Final Year Project`, two levels above this Godot project):

```bash
.venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
```

First start takes a few seconds: it builds the PDM v2 reference features for
all 8 archetypes before serving. Then run the game normally (F5, or
`godot --path .`).

The game does **not** need the server to run. With it down the campus still
loads and is still walkable; the dialogue box reports the connection failure
and the status line under the clock turns red. That is deliberate — an
explicit "model server offline" is much easier to diagnose than an empty
campus or a hang.

## Building a standalone .exe

```bash
tools\build.bat
```

Produces `SampleGame\build\CampusNPC.exe` plus `CampusNPC.pck` (~209MB total).
**Both files must ship together** — the `.pck` holds every asset, and the
`.exe` will not start without it beside it.

The build directory is gitignored: it is large and fully reproducible from
source.

To launch the built game with its server in one step:

```bash
tools\run_demo.bat
```

That is the one-step way to run the demo with voices. In order, it:

1. downloads the NPC voices (~315 MB) if `backend/voices/` has none;
2. **rebuilds the exe if it is older than any script in `Systems/NPC/`**;
3. starts `backend/gguf_server.py` and waits for `/health` (about 10 s, since
   startup loads and warms every voice);
4. prints `Speech: ON (5 voices)`, or says plainly that speech is off;
5. launches the game.

Step 2 exists because this bit once: speech was added to the game code but
the exe was not re-exported, so the demo launched a build that never asked
for audio while the server stood ready to provide it. Nothing looked broken;
the NPCs were simply mute.

### What is NOT in the .exe

The adapter weights. `training/gguf_models/` is roughly 5GB of Q4_K_M models
and they stay on disk, loaded by the Python server — the executable is only
the game client and talks to it over HTTP. Shipping the exe to another machine
therefore means shipping the server, the weights and a Python environment too,
or pointing the client at a server elsewhere:

```bash
CampusNPC.exe -- --server=http://192.168.0.9:8000
```

(The `--` matters: everything after it is passed to the game rather than to
the engine.)

## Playing

Walk into the ground-floor corridor. Five NPCs stand along it. Get within
~3.4m, a `[E] talk` prompt appears over the nearest one, press **E**.
Type, press **Enter**, read the reply. **Esc** leaves the conversation.

Movement is WASD (or arrows), **Shift** to sprint, **Space** to jump.

## What the metrics panel means

| Row | Meaning |
|---|---|
| `model switch` | Time to make this NPC's model resident. `0ms (warm)` means the server's LRU pool already held it. |
| `generation` | Wall-clock time for the model to produce the reply. |
| `total` | What the player actually waits. Green under 500ms (the RQ4 real-time target), amber above. |
| `PDM v2 drift` | Domain-agnostic persona drift (RQ3). Lower = more in-character. |
| `KBD` | Knowledge Boundary Drift (C1) — the fraction of factual references falling outside this NPC's visibility set. |
| `leaked facts` | `knowledge_base.json` ids this NPC should not have known. |

### What the persona layer costs in latency

The persona prefix is byte-identical on every turn with a given NPC, so
llama.cpp's prompt cache absorbs most of its cost after the first line.
Measured across one conversation, model already resident:

| Turn | Generation |
|---|---|
| 1 (cold prefix) | 1318 ms |
| 2–5 | 189–497 ms |

So the richer context is paid **once per conversation**, not once per line.
A first turn with an NPC nobody has spoken to yet also pays the ~1.2–1.9 s
model load on top of that, which is why the HUD total goes amber on the first
thing you say to someone and green afterwards.

Beware of benchmarking this naively: because of that prefix cache, a
measurement's result depends on what ran immediately before it. An A/B of
"persona vs no persona" run back-to-back reported the *richer* prompt as
faster, purely because it inherited a warm prefix.

`KBD: n/a (no factual claim)` is common and is **not** the same as `KBD: 0.00`.
It means the reply asserted nothing checkable, so there was nothing to score.
Reporting it as zero would understate leakage.

A caution on `model switch`: these are **merged** base+LoRA GGUF exports, so
switching persona swaps a whole model handle. It is not the framework's
`set_adapter()` LoRA-delta switch, and should not be quoted as that number.
`backend/main.py` is the server that does the real adapter swap.

## Files

| File | Role |
|---|---|
| `npc_director.gd` | Spawns the NPCs, tracks which one is in range, runs the conversation loop. The only node you place in a scene. |
| `npc_actor.gd` | One NPC: billboard sprite, name plates, interaction trigger, persona fields, positional voice playback, and that NPC's conversation memory. |
| `npc_client.gd` | HTTP client for `/archetypes`, `/chat` and `/speak`. |
| `npc_dialogue_hud.gd` | Dialogue box and metrics panel. |
| `../../tools/npc_smoke_test.gd` | Headless check that the game and server still agree on the wire format. |
| `../../tools/npc_persona_check.gd` | Holds a real multi-turn conversation with one NPC and prints every reply. Use it after editing a persona. |

## Adding or moving an NPC

Edit `SPAWNS` in `npc_director.gd`. Give every NPC an `occupation`, an `intro`,
a `job_line` and a `background`, all written as natural first-person sentences — they go
verbatim into that NPC's mouth, so anything stilted will be echoed back at the
player. The `archetype` string is the contract
with the server — it is POSTed verbatim and must match a key in
`gguf_server.ARCHETYPE_GGUF` exactly (`"police officer"`, not
`"police_officer"`). An archetype the server does not serve is skipped rather
than spawned, so you never get an NPC who errors when spoken to.

Positions only need to be roughly right horizontally: each NPC is raycast onto
the floor beneath it at spawn. Keep them in the corridor band — room blocks on
the ground floor occupy `z=0..8` and `z=16..24` (see `tools/layout_plan.json`),
so `z=8..16` is the circulation corridor.

After changing the spawn table, re-run the smoke test — it checks every entry
against what the server actually serves:

```bash
godot --headless --path . --script res://tools/npc_smoke_test.gd
```

To hear a persona rather than just check the wire format, run a real
conversation against it (index into `SPAWNS`):

```bash
set NPC_INDEX=2
godot --path . --script res://tools/npc_persona_check.gd
```

## Note on project.godot

The InputMap was missing from this project: the movement and camera scripts
have always referenced `move_left` / `move_forward` / `sprint` / `jump`, but
no `[input]` section was ever committed, so the player could not move. It was
written (along with `interact` for talking, and `run/main_scene`) by
`tools/_setup_input.gd`, which is kept so the bindings can be regenerated if
`project.godot` is ever reset.


## Known rough edges

Honest list, all observed while testing rather than guessed at:

- **Explicit recall fails.** See the Memory section — context is used, but not
  recited on request.
- **Background consistency is partial.** "How did you end up here?" is answered
  from the background reliably; "Have you always worked here?" still sometimes
  invents a length of service that contradicts it (2 of 4 phrasings tested were
  consistent, up from 1 of 4 before the background demonstration was added).
- **Referrals are inconsistent.** Adeyemi correctly sends you to Reyes for lost
  property, but Halvorsen answered "I'm really stressed, who can help?" with
  "I'm not sure what you're talking about" rather than naming the counsellor.
- **Small transcription corruptions** — "hiiring", "liasion", "Moestly". These
  are the 1.1B model's own rendering of the primed lines, not typos in `SPAWNS`.
- **Floor questions get mixed up.** The three floor facts are near-identical
  in shape, and NPCs sometimes attribute the ground floor's contents to the
  first. "What's on the second floor?" answered correctly in testing while
  "What's on the first floor?" did not.
- **Reason-form questions still stall.** "What is happening here" and "What's
  going on today?" are answered correctly by all five NPCs, but "Why is
  everyone here?" still tends to begin "I don't know" — the host of the event
  answered it with "I've never seen this before". A second demonstration in
  that phrasing improved the content for 2 of 5 NPCs but did not remove the
  "I don't know" prefix, so further demonstrations were not added: the returns
  had clearly flattened and each one costs prompt tokens.

All four are model-capacity limits rather than wiring bugs, and all four would
be worth re-checking against the Qwen3-0.6B-Instruct adapters as a comparison.
