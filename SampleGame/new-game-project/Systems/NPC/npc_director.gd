extends Node3D
class_name NpcDirector

## Spawns the talkable NPCs and drives the conversation loop.
##
## Drop this node into Scene/Main.tscn and point `player_path` at the Player.
## Everything else is self-contained: it creates its own NpcClient and HUD.
##
## Replies come from the per-archetype LoRA models served by
## backend/gguf_server.py in the repo root. Start it first:
##   .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
## With the server down the NPCs still spawn and are still walkable — the
## dialogue box just reports the connection failure, which is far easier to
## diagnose than an empty campus.

@export var player_path: NodePath = ^"../Player"
@export var server_url: String = "http://127.0.0.1:8000"

## Higher than the server's default of 40, which regularly clipped replies
## mid-clause. The server trims any leftover partial sentence, so the cost of
## the extra headroom is latency, not rambling — expect a few hundred ms more
## per turn than the evaluation configuration.
const REPLY_TOKENS := 64

## Let the player talk: hold push-to-talk (Tab) in a conversation, release to
## send. The line is transcribed by the server (backend/stt.py) and then sent
## exactly like a typed one.
@export var speech_input_enabled: bool = true
const PUSH_TO_TALK := &"push_to_talk"

## Speak replies aloud. Turn off to run silent — everything else is unaffected,
## and the server is never asked to synthesise.
@export var voice_enabled: bool = true

## The scenario every NPC is framed with. Sent to the server as `situation`.
const EVENT := "the Computer Science department open day"

## Everyone's answer to "what is happening here" — the event, not their job.
##
## Without this demonstration each adapter answered from its own training
## topic instead of from the scene: asked "what is happening here" at the open
## day, the police officer replied "we have received reports of a group of
## people breaking into the computer science department". Shared rather than
## per-NPC because every guest is standing at the same event and would
## describe it the same way.
const EVENT_LINE := "It's the department open day — stalls down the corridor and a lot of students asking questions."

## Ground-floor placements and personas.
##
## Room blocks on this floor occupy z=0..8 and z=16..24 (see
## tools/layout_plan.json), so z=8..16 is the circulation corridor and z=12 is
## its centre line. `y` is only a starting height — each NPC is dropped onto
## whatever floor is actually beneath it (see _snap_to_floor).
##
## `occupation`, `intro` and `job_line` are what give an NPC a personality
## rather than a job title. The server seeds each conversation with them as
## already-answered turns, so "who are you", "what are you doing here" and
## "what is your job like" are answered by continuing a demonstrated pattern
## rather than by obeying an instruction the 1.1B model would ignore.
##
## `intro` and `job_line` must be natural first-person sentences: they are put
## verbatim into the NPC's mouth as answers it has already given.
##
## `background` is history rather than script — it goes into the system prompt,
## not into a demonstration, so it colours answers ("I came back after four
## years in industry") without being recited verbatim.
const SPAWNS := [
	{"archetype": "police officer", "name": "Officer Reyes", "role": "campus liaison",
		"occupation": "the campus liaison officer with the city police",
		"intro": "I was invited along to the open day to talk to students about staying safe on campus.",
		"job_line": "Mostly I'm walking the campus and talking to people. The rest is lost property, bike thefts, and making sure students get home safely at night.",
		"background": "I've been the campus liaison for six years, after eight on regular patrol in the north of the city. I know most of the porters by name and I still get lost in the science block.",
		"pos": Vector3(40.0, 1.0, 12.0), "tint": Color(0.62, 0.72, 1.00)},
	{"archetype": "professor", "name": "Prof. Adeyemi", "role": "faculty",
		"occupation": "a lecturer in the Computer Science department",
		"intro": "I came along for the open day, meeting students and talking about the modules I teach.",
		"job_line": "I teach CS2011 Algorithms and CS3040 Compilers, supervise six final-year projects, and spend whatever's left on my own research.",
		"background": "I did my doctorate in this department and came back to teach after four years in industry. I care more about students finishing than about publication counts.",
		"pos": Vector3(47.0, 1.0, 12.0), "tint": Color(0.86, 0.72, 1.00)},
	{"archetype": "executive", "name": "Halvorsen", "role": "head of department",
		"occupation": "the head of the Computer Science department",
		"intro": "I'm hosting the open day, so I'm here to meet students and answer questions about the department.",
		"job_line": "I run the department, so timetables, hiring, budgets and a great many meetings. I still teach one module because I'd miss it otherwise.",
		"background": "I've been head for three years and taught here for eleven before that. I took the job thinking I could fix the timetable, and I was wrong about that.",
		"pos": Vector3(53.0, 1.0, 12.0), "tint": Color(0.90, 0.90, 0.95)},
	{"archetype": "social worker", "name": "Ms. Okafor", "role": "student counsellor",
		"occupation": "a student counsellor in the college welfare office",
		"intro": "I'm at the open day so students know the counselling service exists and how to reach us.",
		"job_line": "I see students one to one, mostly about stress, money worries and homesickness, and I point them towards the right support.",
		"background": "I trained as a social worker in the city before moving into student welfare nine years ago. Most of the job is listening to people who assume nobody wants to hear it.",
		"pos": Vector3(59.0, 1.0, 12.0), "tint": Color(0.70, 0.94, 1.00)},
	{"archetype": "shopkeeper", "name": "Nadia", "role": "campus store",
		"occupation": "the person who runs the campus store",
		"intro": "I've got a stall at the open day, so I'm chatting to students between customers.",
		"job_line": "I open at seven, keep the shelves stocked, and serve a few hundred students a day between lectures.",
		"background": "I took the campus shop over from my aunt and I've run it ever since. I know what most of the regulars want before they reach the till.",
		"pos": Vector3(65.0, 1.0, 12.0), "tint": Color(1.00, 0.88, 0.55)},
]

var _client: NpcClient
var _hud: NpcDialogueHud
var _player: Node3D
var _camera_rig: Node

var _npcs: Array[NpcActor] = []
var _active: NpcActor = null
var _nearest: NpcActor = null
var _awaiting := false

## Speech failures are logged once, not once per line — a missing voice model
## would otherwise fill the output with the same warning every turn.
var _voice_warning_shown := false

## What each NPC has learned about the player; persisted across sessions.
var _memory := PlayerMemoryStore.new()

## News players have reported, and which NPCs have heard it (world_event_store.gd).
var _news := WorldEventStore.new()

var _voice_input: VoiceInput
var _transcribing := false
var _settings: NpcSettings

## How often one piece of news passes from an NPC who knows it to one who does
## not. Slow enough that a player can see news travel, fast enough that it does
## within a few minutes of play.
@export var news_spread_seconds := 45.0

## News reaches every NPC the moment it is reported, rather than travelling one
## NPC at a time. The gossip timer above still runs as a fallback -- and note
## that it only passes news between NPCs of the same archetype, which never
## fires here, because every NPC in SPAWNS is a different archetype.
@export var news_spreads_instantly := true


func _ready() -> void:
	_player = get_node_or_null(player_path)
	if _player == null:
		push_error("NpcDirector: player_path '%s' does not resolve; NPCs will not be interactive."
			% player_path)
	else:
		# third_person_camera.gd grabs the mouse on click. While a dialogue is
		# open we hand the cursor back, so its input handler is muted rather
		# than left fighting the text box for the pointer.
		_camera_rig = _player.get_node_or_null(^"CameraPivot")

	_client = NpcClient.new()
	_client.base_url = _resolve_server_url()
	_client.connection_changed.connect(_on_connection_changed)
	add_child(_client)

	_hud = NpcDialogueHud.new()
	_hud.message_submitted.connect(_on_message_submitted)

	if speech_input_enabled:
		# Tab, not a letter: the chat box has focus during a conversation, so a
		# letter key would be typed into it. Rebindable in Project Settings.
		# Rebindable in the settings panel (F2); defaults in npc_settings.gd.
		_voice_input = VoiceInput.new()
		_voice_input.auto_stopped.connect(_finish_talking)
		add_child(_voice_input)

	# Saved microphone and key bindings, before anyone can be talked to.
	NpcSettings.apply_saved()
	_settings = NpcSettings.new()
	_settings.voice_input = _voice_input
	_settings.closed.connect(_on_settings_closed)
	_settings.bindings_changed.connect(_on_bindings_changed)
	add_child(_settings)
	_on_bindings_changed()

	var gossip := Timer.new()
	gossip.wait_time = news_spread_seconds
	gossip.autostart = true
	gossip.timeout.connect(_spread_news)
	add_child(gossip)
	add_child(_hud)

	await _spawn_npcs()


## Lets an exported build point at a server on another machine without a
## rebuild:
##   CampusNPC.exe -- --server=http://192.168.0.9:8000
## The bare `--` is what separates engine arguments from game arguments.
func _resolve_server_url() -> String:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--server="):
			var url := arg.trim_prefix("--server=")
			print("NpcDirector: using server from command line: ", url)
			return url
	return server_url


func _spawn_npcs() -> void:
	var served: Array = await _client.fetch_archetypes()

	for entry in SPAWNS:
		# With the server up, only spawn NPCs it can actually answer as —
		# otherwise the campus has someone standing in it who errors the
		# moment you talk to them. With the server down we spawn everyone,
		# so the world still looks right.
		if not served.is_empty() and not served.has(entry["archetype"]):
			continue
		var npc := NpcActor.create(entry)
		add_child(npc)
		npc.set_prompt_key(NpcSettings.key_name(&"interact"))
		_npcs.append(npc)
		_snap_to_floor(npc)

	if _npcs.is_empty():
		push_warning("NpcDirector: no NPCs spawned — server served none of the known archetypes.")


## Drops an NPC onto the floor under its spawn point so it never hovers or
## sinks if the corridor height changes.
func _snap_to_floor(npc: NpcActor) -> void:
	# Cast from just above the spawn point, not from high overhead: storeys
	# are 3m apart, so a ray starting 3m up begins ON the next floor and
	# snaps the NPC one level too high.
	var space := get_world_3d().direct_space_state
	var from := npc.global_position + Vector3.UP * 0.8
	var to := npc.global_position + Vector3.DOWN * 2.5
	var query := PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = [npc.get_rid()]
	var hit := space.intersect_ray(query)
	if hit.has("position"):
		npc.global_position = hit["position"]


# --- interaction -----------------------------------------------------------

func _process(_delta: float) -> void:
	if _active != null or _player == null or (_settings and _settings.is_open()):
		return

	# Nearest in range wins, so standing between two NPCs is never ambiguous
	# about which one E will talk to.
	var best: NpcActor = null
	var best_distance := INF
	for npc in _npcs:
		var d := npc.global_position.distance_to(_player.global_position)
		if d <= NpcActor.INTERACT_RANGE and d < best_distance:
			best_distance = d
			best = npc

	if best != _nearest:
		if _nearest != null:
			_nearest.set_prompt_visible(false)
		if best != null:
			best.set_prompt_visible(true)
		_nearest = best


## Push-to-talk. In _input, ahead of the GUI, so Tab never reaches the chat
## box (where it would move focus).
func _input(event: InputEvent) -> void:
	if _settings.is_open():
		return   # the panel handles its own input, including rebinding
	if event.is_action_pressed(NpcSettings.OPEN_ACTION):
		_open_settings()
		get_viewport().set_input_as_handled()
		return
	if _voice_input == null or _active == null or not event.is_action(PUSH_TO_TALK):
		return
	get_viewport().set_input_as_handled()
	if event.is_echo():
		return
	if event.is_pressed():
		if not _awaiting and not _transcribing:
			_voice_input.start()
			_hud.show_listening()
	elif _voice_input.is_recording():
		_finish_talking()


func _finish_talking() -> void:
	if not _voice_input.is_recording():
		return
	var npc := _active
	var clip: Dictionary = _voice_input.stop()
	if npc == null:
		return
	_transcribing = true
	_hud.show_transcribing()
	var heard: Dictionary = await _client.transcribe(clip["pcm"], clip["sample_rate"], clip["channels"])
	_transcribing = false
	if _active != npc:
		return   # walked away while it was being transcribed
	if heard.has("error"):
		_hud.show_error("(speaking)", "Could not hear you: %s" % heard["error"])
		return
	var text := str(heard.get("text", "")).strip_edges()
	_hud.set_metric_heard(float(heard.get("transcribe_ms", 0.0)), str(heard.get("skipped", "")))
	if text.is_empty():
		var key := NpcSettings.key_name(PUSH_TO_TALK)
		_hud.show_note("(speaking)", ("Didn't catch that. Hold %s and speak, then let go. "
			+ "Check the microphone in settings (F2).") % key
			if heard.get("skipped", "") != "short" else "Hold %s while you speak, then let go." % key)
		return
	_on_message_submitted(text)


func _open_settings() -> void:
	if _voice_input and _voice_input.is_recording():
		_voice_input.stop()
	if _nearest:
		_nearest.set_prompt_visible(false)
		_nearest = null   # re-shown by _process once the panel closes
	_settings.open()
	_set_player_active(false)


func _on_settings_closed() -> void:
	if _active == null:
		_set_player_active(true)
	else:
		_hud.focus_input()


func _on_bindings_changed() -> void:
	_hud.set_key_hints(NpcSettings.key_name(PUSH_TO_TALK) if _voice_input else "")
	for npc in _npcs:
		npc.set_prompt_key(NpcSettings.key_name(&"interact"))


func _unhandled_input(event: InputEvent) -> void:
	# LineEdit marks typed keys as handled, so this never fires mid-sentence,
	# which is why "e" can be typed into the chat box without reopening a
	# conversation.
	if event.is_action_pressed(&"interact") and _active == null and _nearest != null:
		_open(_nearest)
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed(&"ui_cancel") and _active != null:
		_close()
		get_viewport().set_input_as_handled()


func _open(npc: NpcActor) -> void:
	_active = npc
	npc.set_prompt_visible(false)
	_hud.open_with(npc.display_name, npc.archetype, npc.tint, npc.role_label_text, npc.tint)
	_hud.show_memory(_memory.get_for(npc.display_name))
	_hud.show_news(_news.known_by(npc.display_name))
	_set_player_active(false)


func _close() -> void:
	# Deliberately allowed mid-request: the pending chat() call still resolves
	# and _deliver() drops the result because _active is null. Blocking Esc
	# until a slow reply lands would feel like a hang.
	if _voice_input and _voice_input.is_recording():
		_voice_input.stop()
	if _active:
		_active.stop_voice()
	_active = null
	_awaiting = false
	_hud.set_dialogue_visible(false)
	_set_player_active(true)


func _set_player_active(active: bool) -> void:
	if _player:
		if not active:
			_player.velocity = Vector3.ZERO
		_player.set_physics_process(active)
	if _camera_rig:
		_camera_rig.set_process_unhandled_input(active)
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if active else Input.MOUSE_MODE_VISIBLE


func _on_message_submitted(text: String) -> void:
	if _awaiting or _transcribing or _active == null:
		return
	if text.begins_with("/"):
		_run_command(_active, text)
		return
	_awaiting = true
	_hud.show_pending(text)

	var npc := _active
	var reply: Dictionary = await _client.chat(
		npc.archetype, text, _persona_for(npc), npc.history)
	_deliver(npc, text, reply)


## The framing sent with every line: who this NPC is, and what they are doing
## at the event. Without it the server falls back to the evaluation prompt,
## which has no name, no job and no idea it is in a college.
## The other guests, so an NPC can hand a question on to the right person
## instead of inventing an answer or dead-ending the player. Built from the
## live NPC list rather than hardcoded, so it stays correct when the server
## serves only some archetypes and only some NPCs actually spawn.
func _others_at_event(self_npc: NpcActor) -> String:
	var names: Array[String] = []
	for npc in _npcs:
		if npc != self_npc:
			# The occupation, not the short label: "campus liaison" alone never
			# told another NPC that Reyes is the police officer.
			names.append("%s (%s)" % [npc.display_name,
				npc.occupation if not npc.occupation.is_empty() else npc.role_label_text])
	return ", ".join(names)


func _persona_for(npc: NpcActor) -> Dictionary:
	return {
		"name": npc.display_name,
		"occupation": npc.occupation,
		"intro": npc.intro,
		"job_line": npc.job_line,
		"background": npc.background,
		"situation": EVENT,
		"event_line": EVENT_LINE,
		"facts": CampusFacts.for_npc(npc.display_name),
		"fact_demos": CampusFacts.demos_for(npc.display_name),
		"others": _others_at_event(npc),
		"max_tokens": REPLY_TOKENS,
		"player_memory": _memory.get_for(npc.display_name),
		"known_events": _news.known_by(npc.display_name),
		"unheard_events": _news.unheard_by(npc.display_name),
	}


## Everyone hears what was just reported (news_spreads_instantly).
func _spread_news_now() -> void:
	var names: Array = []
	for npc in _npcs:
		names.append(npc.display_name)
	var told: Array = _news.spread_all(names)
	if told.is_empty():
		return
	print("NpcDirector: news reached %d other NPCs at once" % told.size())
	if _active != null:
		_hud.show_news(_news.known_by(_active.display_name))


## One step of gossip between the NPCs that are actually in the world.
func _spread_news() -> void:
	var names: Array = []
	var types: Dictionary = {}
	for npc in _npcs:
		names.append(npc.display_name)
		types[npc.display_name] = npc.archetype
	var step := _news.spread(names, types)
	if step.is_empty():
		return
	print("NpcDirector: news %s passed from %s to %s" % [step["id"], step["from"], step["to"]])
	if _active != null and _active.display_name == step["to"]:
		_hud.show_news(_news.known_by(step["to"]))


## Testing commands, typed into the chat box instead of a line of dialogue.
## They exist so a personalisation test can be repeated from a known state
## without quitting the game or hand-editing the save file.
##   /forget       this NPC forgets you (memory and conversation)
##   /forget all   everyone forgets you
##   /memory       show what this NPC remembers
##   /news         show what news this NPC has heard
##   /forget news  clear all reported news
func _run_command(npc: NpcActor, text: String) -> void:
	var command := text.strip_edges().to_lower()
	match command:
		"/forget":
			_memory.forget(npc.display_name)
			npc.history.clear()
			_hud.show_note(text, "%s no longer remembers you." % npc.display_name)
		"/forget all":
			_memory.forget_all()
			for other in _npcs:
				other.history.clear()
			_hud.show_note(text, "Nobody remembers you now.")
		"/memory":
			var memory := _memory.get_for(npc.display_name)
			_hud.show_note(text, JSON.stringify(memory) if not memory.is_empty()
				else "%s knows nothing about you yet." % npc.display_name)
		"/news":
			var heard := _news.known_by(npc.display_name)
			_hud.show_note(text, JSON.stringify(heard) if not heard.is_empty()
				else "%s has not heard any news." % npc.display_name)
		"/forget news":
			_news.forget_all()
			_hud.show_note(text, "All reported news cleared.")
		"/settings":
			_hud.show_note(text, "Settings open (F2 or Esc to close).")
			_open_settings()
		_:
			_hud.show_note(text, "Commands: /memory, /news, /forget, /forget all, /forget news, /settings")
	_hud.show_memory(_memory.get_for(npc.display_name))
	_hud.show_news(_news.known_by(npc.display_name))


func _deliver(npc: NpcActor, message: String, reply: Dictionary) -> void:
	_awaiting = false

	# Learned before the walked-away check below: the NPC heard what the player
	# said even if the player left before the answer came back.
	var updates: Dictionary = reply.get("memory_updates", {})
	if not reply.has("error") and not updates.is_empty():
		_memory.set_for(npc.display_name, reply.get("player_memory", {}))
	# Same for news: a report was made to this NPC even if the player walked off.
	var reported = reply.get("reported_event")
	if not reply.has("error") and reported is Dictionary and not str(reported.get("what", "")).is_empty():
		_news.add_report(npc.display_name, str(reported["what"]), str(reported.get("where", "")),
			str(reported.get("kind", "incident")))
		if news_spreads_instantly:
			_spread_news_now()
	var shared := str(reply.get("shared_event_id", ""))
	if not shared.is_empty():
		_news.mark_shared(npc.display_name, shared)

	# The player may have walked away, or opened a different NPC, while this
	# was in flight. Showing the reply now would put one NPC's line in
	# another's mouth, so it is dropped instead.
	if _active == null or _active != npc:
		return

	if reply.has("error"):
		_hud.show_error(message, str(reply["error"]))
		return

	var text := str(reply.get("response", ""))
	npc.remember(message, text)
	_hud.show_reply(message, text)
	_hud.update_metrics(reply)
	_hud.show_memory(_memory.get_for(npc.display_name), updates)
	_hud.show_news(_news.known_by(npc.display_name))
	# Fire and forget. The line is already on screen; speech catches up a few
	# hundred milliseconds later and must never hold the subtitle back.
	if voice_enabled:
		_speak(npc, text)


## Synthesises and plays one reply.
##
## Deliberately tolerant: if the server has no voices installed, or synthesis
## fails, the NPC simply does not speak. Speech is an enhancement, not a
## dependency, so nothing here is allowed to interrupt the conversation.
func _speak(npc: NpcActor, text: String) -> void:
	var audio: Dictionary = await _client.speak(text, npc.display_name)
	if audio.has("error"):
		if not _voice_warning_shown:
			_voice_warning_shown = true
			push_warning("NpcDirector: speech unavailable (%s). Continuing silently."
				% audio["error"])
		return
	# The player may have walked off or started a new line while this was being
	# synthesised; do not make an NPC speak over its own next answer.
	if _active != npc:
		return
	npc.play_voice(audio["pcm"], audio["sample_rate"])


func _on_connection_changed(online: bool, detail: String) -> void:
	_hud.set_status(online, detail)
