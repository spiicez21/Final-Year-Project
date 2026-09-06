# NPC dialogue layer

Talkable campus NPCs whose replies are generated live by the project's
per-archetype LoRA models, with the paper's drift metrics shown per turn.

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

That starts `backend/gguf_server.py`, waits for `/health` to answer, then
launches the exe.

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
| `npc_actor.gd` | One NPC: billboard sprite + name plates + interaction trigger. |
| `npc_client.gd` | HTTP client for `/archetypes` and `/chat`. |
| `npc_dialogue_hud.gd` | Dialogue box and metrics panel. |
| `../../tools/npc_smoke_test.gd` | Headless check that the game and server still agree on the wire format. |

## Adding or moving an NPC

Edit `SPAWNS` in `npc_director.gd`. The `archetype` string is the contract
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

## Note on project.godot

The InputMap was missing from this project: the movement and camera scripts
have always referenced `move_left` / `move_forward` / `sprint` / `jump`, but
no `[input]` section was ever committed, so the player could not move. It was
written (along with `interact` for talking, and `run/main_scene`) by
`tools/_setup_input.gd`, which is kept so the bindings can be regenerated if
`project.godot` is ever reset.
