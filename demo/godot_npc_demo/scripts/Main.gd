extends Node2D

## Walkable demo for the domain-adaptive LoRA NPC framework.
##
## Walk up to an NPC, press E, type. Every reply is generated live by the
## per-archetype Q4_K_M GGUF behind backend/gguf_server.py, and the HUD shows
## the three numbers the project actually cares about: model-switch latency,
## generation latency, and the two drift metrics (PDM v2 and KBD).
##
## The whole world is constructed here in code rather than authored as a
## .tscn. For a four-NPC demo that is less to keep in sync than a scene tree,
## and it keeps the repo free of binary/.import files.

const WORLD_SIZE := Vector2(1280, 720)
const MAX_NPCS := 5

## Street geometry. NPCs stand on the upper sidewalk and the player starts on
## the lower one, so talking to someone means crossing the road — which is
## what makes the model-switch latency visible as a thing you walk into.
const SIDEWALK_TOP_Y := 350.0
const SIDEWALK_BOTTOM_Y := 530.0

## NPC row sits below the metrics panel (which ends around y=178) so the
## rightmost NPC is never hidden behind the HUD.
const NPC_ROW_Y := 250.0

## Display name and colour per archetype. Keys must match the server's
## ARCHETYPE_GGUF keys exactly — the key is what gets POSTed to /chat.
## Names are drawn from the personas in data/processed/modern_npc_dataset.json
## so the demo does not invent characters the adapters were never trained on.
const ARCHETYPE_INFO := {
	"police officer": {"name": "Officer Reyes", "color": Color(0.36, 0.52, 0.86)},
	"bartender": {"name": "Sam, the bartender", "color": Color(0.85, 0.55, 0.30)},
	"pharmacist": {"name": "Dr. Wu, pharmacist", "color": Color(0.42, 0.78, 0.62)},
	"professor": {"name": "Prof. Adeyemi", "color": Color(0.72, 0.55, 0.85)},
	"shopkeeper": {"name": "Nadia, the clerk", "color": Color(0.86, 0.72, 0.35)},
	"social worker": {"name": "Ms. Okafor", "color": Color(0.55, 0.80, 0.85)},
	"executive": {"name": "Halvorsen, CEO", "color": Color(0.70, 0.70, 0.76)},
	"service worker": {"name": "Devi, attendant", "color": Color(0.80, 0.45, 0.55)},
}

## Fallback roster when the server cannot be reached at startup, so the world
## is still walkable and the failure is visible in-world rather than an empty
## scene that looks like a broken build.
const FALLBACK_ARCHETYPES := ["police officer", "bartender", "pharmacist", "professor"]

var _client: NpcClient
var _player: Player
var _npcs: Array[Npc] = []
var _active_npc: Npc = null
var _nearest_npc: Npc = null
var _awaiting_reply := false

var _dialogue_panel: PanelContainer
var _dialogue_name: Label
var _dialogue_text: RichTextLabel
var _dialogue_input: LineEdit
var _status_label: Label
var _metric_labels := {}


func _ready() -> void:
	_client = NpcClient.new()
	_client.connection_changed.connect(_on_connection_changed)
	add_child(_client)

	# Let the base URL be overridden without editing the project, e.g.
	#   Godot_v4.7-stable_win64.exe --path . -- --server=http://192.168.0.9:8000
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--server="):
			_client.base_url = arg.trim_prefix("--server=")

	_build_world()
	_build_ui()
	_spawn_player()
	await _spawn_npcs()


# --- world -----------------------------------------------------------------

## Asphalt, a pale sidewalk band top and bottom, and a dashed centre line.
## Purely cosmetic. Drawn by Main itself rather than a child node because a
## parent's _draw() runs before its children's, which puts the road under the
## player and NPCs without any z_index juggling.
func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, WORLD_SIZE), Color(0.13, 0.14, 0.17))
	draw_rect(Rect2(0, 0, WORLD_SIZE.x, SIDEWALK_TOP_Y), Color(0.21, 0.22, 0.26))
	draw_rect(Rect2(0, SIDEWALK_BOTTOM_Y, WORLD_SIZE.x, WORLD_SIZE.y - SIDEWALK_BOTTOM_Y),
		Color(0.21, 0.22, 0.26))
	draw_line(Vector2(0, SIDEWALK_TOP_Y), Vector2(WORLD_SIZE.x, SIDEWALK_TOP_Y),
		Color(0.30, 0.31, 0.36), 3.0)
	draw_line(Vector2(0, SIDEWALK_BOTTOM_Y), Vector2(WORLD_SIZE.x, SIDEWALK_BOTTOM_Y),
		Color(0.30, 0.31, 0.36), 3.0)
	var lane_y := (SIDEWALK_TOP_Y + SIDEWALK_BOTTOM_Y) * 0.5 - 2.5
	for x in range(40, int(WORLD_SIZE.x), 90):
		draw_rect(Rect2(x, lane_y, 46, 5), Color(0.42, 0.42, 0.30))


func _build_world() -> void:
	queue_redraw()

	var title := Label.new()
	title.text = "Modern City — NPC persona demo"
	title.add_theme_font_size_override("font_size", 15)
	title.add_theme_color_override("font_color", Color(0.55, 0.58, 0.66))
	title.position = Vector2(24, 18)
	add_child(title)


func _spawn_player() -> void:
	_player = Player.new()
	_player.position = Vector2(WORLD_SIZE.x * 0.5, SIDEWALK_BOTTOM_Y + 95.0)
	add_child(_player)


func _spawn_npcs() -> void:
	var archetypes: Array = await _client.fetch_archetypes()
	if archetypes.is_empty():
		archetypes = FALLBACK_ARCHETYPES.duplicate()

	# Keep only archetypes we have a character for, and cap the count so the
	# street stays readable. The server may legitimately offer all 8.
	var usable: Array = []
	for a in archetypes:
		if ARCHETYPE_INFO.has(a):
			usable.append(a)
		if usable.size() >= MAX_NPCS:
			break

	var count := usable.size()
	for i in range(count):
		var archetype: String = usable[i]
		var info: Dictionary = ARCHETYPE_INFO[archetype]
		# Evenly spaced along the upper sidewalk, alternating slightly in y so
		# the name labels of adjacent NPCs do not overlap.
		var x: float = WORLD_SIZE.x * (float(i) + 1.0) / (float(count) + 1.0)
		var y: float = NPC_ROW_Y + (46.0 if i % 2 == 1 else 0.0)
		var npc := Npc.create(archetype, info["name"], info["color"], Vector2(x, y))
		_npcs.append(npc)
		add_child(npc)


# --- ui --------------------------------------------------------------------

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	_status_label = Label.new()
	_status_label.text = "connecting…"
	_status_label.add_theme_font_size_override("font_size", 12)
	_status_label.position = Vector2(24, 44)
	layer.add_child(_status_label)
	_set_status(false, "connecting…")

	layer.add_child(_build_metrics_panel())
	_dialogue_panel = _build_dialogue_panel()
	layer.add_child(_dialogue_panel)

	var hint := Label.new()
	hint.text = "WASD / arrows to move   ·   E to talk   ·   Esc to leave"
	hint.add_theme_font_size_override("font_size", 12)
	hint.add_theme_color_override("font_color", Color(0.55, 0.58, 0.66))
	hint.position = Vector2(24, WORLD_SIZE.y - 32)
	layer.add_child(hint)


func _build_metrics_panel() -> PanelContainer:
	var panel := PanelContainer.new()
	panel.position = Vector2(WORLD_SIZE.x - 330, 18)
	panel.custom_minimum_size = Vector2(306, 0)
	panel.add_theme_stylebox_override("panel", _panel_style())

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 3)
	panel.add_child(box)

	var heading := Label.new()
	heading.text = "LIVE METRICS"
	heading.add_theme_font_size_override("font_size", 11)
	heading.add_theme_color_override("font_color", Color(0.55, 0.58, 0.66))
	box.add_child(heading)

	# Label text is set in _update_metrics(); the tooltip is where the actual
	# definition lives, since these are the paper's metrics and "0.41" means
	# nothing to someone seeing the demo for the first time.
	var rows := [
		["switch", "model switch", "Time to make this NPC's model resident.\n0ms = already warm in the LRU pool.\nNOTE: these are merged GGUFs, so this is a\nwhole-model swap, not a LoRA set_adapter() call."],
		["gen", "generation", "Wall-clock time for the model to produce the reply."],
		["total", "total", "What the player actually waits: switch + generation."],
		["pdm", "PDM v2 drift", "Domain-agnostic persona drift (RQ3).\nLower = more consistent with the archetype."],
		["kbd", "KBD", "Knowledge Boundary Drift (C1): fraction of factual\nreferences falling outside this NPC's visibility set.\n'n/a' = the reply stated no checkable fact."],
		["leak", "leaked facts", "knowledge_base.json ids this NPC should not know."],
	]
	for row in rows:
		var line := Label.new()
		line.add_theme_font_size_override("font_size", 12)
		line.tooltip_text = row[2]
		line.mouse_filter = Control.MOUSE_FILTER_STOP
		line.text = "%s: —" % row[1]
		line.set_meta("caption", row[1])
		box.add_child(line)
		_metric_labels[row[0]] = line

	return panel


func _build_dialogue_panel() -> PanelContainer:
	var panel := PanelContainer.new()
	panel.position = Vector2(140, WORLD_SIZE.y - 232)
	panel.custom_minimum_size = Vector2(1000, 170)
	panel.add_theme_stylebox_override("panel", _panel_style())
	panel.visible = false

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	panel.add_child(box)

	_dialogue_name = Label.new()
	_dialogue_name.add_theme_font_size_override("font_size", 15)
	box.add_child(_dialogue_name)

	_dialogue_text = RichTextLabel.new()
	_dialogue_text.bbcode_enabled = true
	_dialogue_text.fit_content = false
	_dialogue_text.custom_minimum_size = Vector2(980, 76)
	_dialogue_text.add_theme_font_size_override("normal_font_size", 14)
	box.add_child(_dialogue_text)

	_dialogue_input = LineEdit.new()
	_dialogue_input.placeholder_text = "Say something, then press Enter…"
	_dialogue_input.custom_minimum_size = Vector2(980, 32)
	_dialogue_input.text_submitted.connect(_on_message_submitted)
	box.add_child(_dialogue_input)

	return panel


func _panel_style() -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.09, 0.10, 0.13, 0.94)
	style.border_color = Color(0.28, 0.30, 0.36)
	style.set_border_width_all(1)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(12)
	return style


# --- interaction -----------------------------------------------------------

func _process(_delta: float) -> void:
	if _active_npc != null or _player == null:
		return

	# Nearest NPC in range wins, so standing between two NPCs is never
	# ambiguous about which one E will talk to.
	var best: Npc = null
	var best_distance := INF
	for npc in _npcs:
		if npc.is_player_in_range(_player.global_position):
			var d := npc.global_position.distance_to(_player.global_position)
			if d < best_distance:
				best_distance = d
				best = npc

	if best != _nearest_npc:
		if _nearest_npc != null:
			_nearest_npc.set_prompt_visible(false)
		if best != null:
			best.set_prompt_visible(true)
		_nearest_npc = best


func _unhandled_input(event: InputEvent) -> void:
	# LineEdit marks typed keys as handled, so this never fires mid-sentence —
	# which is why "e" can be typed into the chat box without reopening a
	# conversation.
	if not (event is InputEventKey and event.pressed and not event.echo):
		return

	if event.keycode == KEY_E and _active_npc == null and _nearest_npc != null:
		_open_dialogue(_nearest_npc)
		get_viewport().set_input_as_handled()
	elif event.keycode == KEY_ESCAPE and _active_npc != null:
		_close_dialogue()
		get_viewport().set_input_as_handled()


func _open_dialogue(npc: Npc) -> void:
	_active_npc = npc
	npc.set_prompt_visible(false)
	_player.movement_enabled = false

	_dialogue_name.text = "%s  ·  %s" % [npc.display_name, npc.archetype]
	_dialogue_name.add_theme_color_override("font_color", npc.tint)
	_dialogue_text.text = "[color=#7f8694]The %s turns to face you.[/color]" % npc.archetype
	_dialogue_input.text = ""
	_dialogue_panel.visible = true
	_dialogue_input.editable = true
	_dialogue_input.grab_focus()
	_reset_metrics()


func _close_dialogue() -> void:
	# Deliberately allowed mid-request: the pending chat() call still resolves,
	# and _on_reply() drops the result because _active_npc is null. Blocking
	# Esc until a slow reply lands would feel like a hang.
	_active_npc = null
	_awaiting_reply = false
	_dialogue_panel.visible = false
	_dialogue_input.release_focus()
	_player.movement_enabled = true


func _on_message_submitted(text: String) -> void:
	var message := text.strip_edges()
	if message.is_empty() or _awaiting_reply or _active_npc == null:
		return

	_awaiting_reply = true
	_dialogue_input.editable = false
	_dialogue_input.text = ""
	_dialogue_text.text = "[color=#7f8694]you:[/color] %s\n\n[color=#7f8694]…thinking[/color]" % message

	var npc := _active_npc
	var reply: Dictionary = await _client.chat(npc.archetype, message)
	_on_reply(npc, message, reply)


func _on_reply(npc: Npc, message: String, reply: Dictionary) -> void:
	_awaiting_reply = false

	# The player may have walked away, or switched NPCs, while this was in
	# flight. Showing the reply now would put one NPC's line in another's
	# mouth, so it is dropped instead.
	if _active_npc == null or _active_npc != npc:
		return

	_dialogue_input.editable = true
	_dialogue_input.grab_focus()

	if reply.has("error"):
		_dialogue_text.text = "[color=#7f8694]you:[/color] %s\n\n[color=#ff8080]%s[/color]" % [
			message, reply["error"]
		]
		return

	_dialogue_text.text = "[color=#7f8694]you:[/color] %s\n\n[color=#e8ecf5]%s[/color]" % [
		message, reply.get("response", "")
	]
	_update_metrics(reply)


# --- metrics ---------------------------------------------------------------

func _reset_metrics() -> void:
	for key in _metric_labels:
		var label: Label = _metric_labels[key]
		label.text = "%s: —" % label.get_meta("caption")
		label.remove_theme_color_override("font_color")


func _update_metrics(reply: Dictionary) -> void:
	var switch_ms: float = reply.get("adapter_switch_ms", 0.0)
	var gen_ms: float = reply.get("generation_ms", 0.0)

	_set_metric("switch", "%.0f ms%s" % [switch_ms, "  (warm)" if switch_ms == 0.0 else "  (cold load)"])
	_set_metric("gen", "%.0f ms" % gen_ms)
	# RQ4's real-time target is <500ms, so the total is colour-coded against it
	# rather than left as a bare number.
	_set_metric("total", "%.0f ms" % (switch_ms + gen_ms),
		Color(0.45, 0.85, 0.55) if (switch_ms + gen_ms) < 500.0 else Color(0.95, 0.72, 0.35))

	var pdm = reply.get("drift_score")
	_set_metric("pdm", "n/a" if pdm == null else "%.3f" % float(pdm))

	# KBD is null whenever the reply made no checkable factual claim, which is
	# common for short in-character barks. That is "nothing to score", not
	# "scored zero" — conflating the two would understate leakage.
	var kbd = reply.get("kbd")
	if kbd == null:
		_set_metric("kbd", "n/a (no factual claim)")
	else:
		var value := float(kbd)
		_set_metric("kbd", "%.2f" % value,
			Color(0.45, 0.85, 0.55) if value == 0.0 else Color(1.0, 0.45, 0.45))

	var leaks: Array = reply.get("leaked_fact_ids", [])
	_set_metric("leak", "none" if leaks.is_empty() else ", ".join(leaks),
		Color(1.0, 0.45, 0.45) if not leaks.is_empty() else Color(0.55, 0.58, 0.66))


func _set_metric(key: String, value: String, color: Color = Color(0.85, 0.88, 0.94)) -> void:
	var label: Label = _metric_labels[key]
	label.text = "%s: %s" % [label.get_meta("caption"), value]
	label.add_theme_color_override("font_color", color)


func _on_connection_changed(online: bool, detail: String) -> void:
	_set_status(online, detail)


func _set_status(online: bool, detail: String) -> void:
	_status_label.text = ("● %s" if online else "○ offline — %s") % detail
	_status_label.add_theme_color_override(
		"font_color", Color(0.45, 0.85, 0.55) if online else Color(0.95, 0.55, 0.45)
	)
