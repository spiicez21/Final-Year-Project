extends CanvasLayer
class_name NpcDialogueHud

## Dialogue box + live metrics readout.
##
## Built in code and anchored rather than authored as a .tscn with fixed
## coordinates, so it survives the window being resized (the project uses
## stretch mode "canvas_items" with an "expand" aspect, so the viewport really
## does change shape).
##
## The metrics are the point of the demo as much as the dialogue is: they are
## the paper's own numbers, computed per turn on the reply you just read.

signal message_submitted(text: String)

var _panel: PanelContainer
var _speaker: Label
var _body: RichTextLabel
var _input: LineEdit
var _status: Label
var _metrics := {}


func _ready() -> void:
	layer = 10
	_build_status()
	_build_metrics()
	_build_dialogue()
	set_dialogue_visible(false)


func _panel_style() -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.07, 0.08, 0.11, 0.93)
	style.border_color = Color(0.32, 0.35, 0.42)
	style.set_border_width_all(1)
	style.set_corner_radius_all(6)
	style.set_content_margin_all(14)
	return style


func _build_status() -> void:
	_status = Label.new()
	_status.add_theme_font_size_override("font_size", 12)
	_status.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.9))
	_status.add_theme_constant_override("shadow_offset_y", 1)
	# Below the existing time-of-day HUD, which occupies the top-left corner.
	_status.offset_left = 16.0
	_status.offset_top = 62.0
	add_child(_status)
	set_status(false, "connecting…")


func _build_metrics() -> void:
	var panel := PanelContainer.new()
	# Anchored under the minimap, which occupies offset_top 16..212 in the
	# top-right corner (see the Minimap node in Scene/Main.tscn).
	#
	# All offsets here are in the project's 1152x648 UI space, NOT window
	# pixels: the project stretches "canvas_items", so on a 1600x900 window
	# these coordinates are scaled by ~1.39 on the way to the screen.
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	panel.offset_left = -300.0
	panel.offset_right = -16.0
	panel.offset_top = 224.0
	panel.add_theme_stylebox_override("panel", _panel_style())
	add_child(panel)

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 3)
	panel.add_child(box)

	var heading := Label.new()
	heading.text = "LIVE METRICS"
	heading.add_theme_font_size_override("font_size", 11)
	heading.add_theme_color_override("font_color", Color(0.55, 0.58, 0.66))
	box.add_child(heading)

	# The tooltip carries the definition: "0.41" means nothing to someone
	# seeing the demo for the first time.
	var rows := [
		["switch", "model switch", "Time to make this NPC's model resident.\n0ms = already warm in the server's LRU pool.\nNOTE: these are merged GGUFs, so this is a\nwhole-model swap, not a LoRA set_adapter() call."],
		["gen", "generation", "Wall-clock time for the model to produce the reply."],
		["total", "total", "What the player actually waits: switch + generation.\nGreen when under the 500ms real-time target (RQ4)."],
		["pdm", "PDM v2 drift", "Domain-agnostic persona drift (RQ3).\nLower = more consistent with the archetype."],
		["kbd", "KBD", "Knowledge Boundary Drift (C1): fraction of factual\nreferences falling outside this NPC's visibility set.\n'n/a' = the reply stated no checkable fact."],
		["leak", "leaked facts", "knowledge_base.json ids this NPC should not know."],
	]
	for row in rows:
		var line := Label.new()
		line.add_theme_font_size_override("font_size", 12)
		line.tooltip_text = row[2]
		line.mouse_filter = Control.MOUSE_FILTER_STOP
		line.set_meta("caption", row[1])
		box.add_child(line)
		_metrics[row[0]] = line
	reset_metrics()


func _build_dialogue() -> void:
	_panel = PanelContainer.new()
	# Bottom-centre strip, again in 1152x648 UI space.
	_panel.anchor_top = 1.0
	_panel.anchor_bottom = 1.0
	_panel.anchor_right = 1.0
	_panel.grow_vertical = Control.GROW_DIRECTION_BEGIN
	_panel.offset_left = 72.0
	_panel.offset_right = -72.0
	_panel.offset_top = -190.0
	_panel.offset_bottom = -24.0
	_panel.add_theme_stylebox_override("panel", _panel_style())
	add_child(_panel)

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	_panel.add_child(box)

	_speaker = Label.new()
	_speaker.add_theme_font_size_override("font_size", 16)
	box.add_child(_speaker)

	_body = RichTextLabel.new()
	_body.bbcode_enabled = true
	_body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_body.add_theme_font_size_override("normal_font_size", 14)
	box.add_child(_body)

	_input = LineEdit.new()
	_input.placeholder_text = "Say something, then press Enter…"
	_input.custom_minimum_size = Vector2(0, 34)
	_input.text_submitted.connect(_on_submit)
	box.add_child(_input)


func _on_submit(text: String) -> void:
	var trimmed := text.strip_edges()
	if not trimmed.is_empty():
		message_submitted.emit(trimmed)


# --- public API used by npc_director.gd ------------------------------------

func set_dialogue_visible(value: bool) -> void:
	_panel.visible = value
	if value:
		_input.text = ""
		_input.editable = true
		_input.grab_focus()
	else:
		_input.release_focus()


func is_dialogue_visible() -> bool:
	return _panel.visible


func open_with(speaker: String, archetype: String, color: Color) -> void:
	_speaker.text = "%s   ·   %s" % [speaker, archetype]
	_speaker.add_theme_color_override("font_color", color)
	_body.text = "[color=#7f8694]The %s turns to face you.[/color]" % archetype
	reset_metrics()
	set_dialogue_visible(true)


func show_pending(message: String) -> void:
	_input.editable = false
	_input.text = ""
	_body.text = "[color=#7f8694]you:[/color] %s\n\n[color=#7f8694]…thinking[/color]" % message


func show_reply(message: String, reply: String) -> void:
	_input.editable = true
	_input.grab_focus()
	_body.text = "[color=#7f8694]you:[/color] %s\n\n[color=#e8ecf5]%s[/color]" % [message, reply]


func show_error(message: String, error: String) -> void:
	_input.editable = true
	_input.grab_focus()
	_body.text = "[color=#7f8694]you:[/color] %s\n\n[color=#ff8080]%s[/color]" % [message, error]


func reset_metrics() -> void:
	for key in _metrics:
		var label: Label = _metrics[key]
		label.text = "%s: —" % label.get_meta("caption")
		label.remove_theme_color_override("font_color")


func update_metrics(reply: Dictionary) -> void:
	var switch_ms: float = reply.get("adapter_switch_ms", 0.0)
	var gen_ms: float = reply.get("generation_ms", 0.0)

	_set_metric("switch", "%.0f ms%s" % [switch_ms, "  (warm)" if switch_ms == 0.0 else "  (cold load)"])
	_set_metric("gen", "%.0f ms" % gen_ms)
	# RQ4's real-time target is <500ms, so the total is colour-coded against
	# it rather than left as a bare number.
	var total := switch_ms + gen_ms
	_set_metric("total", "%.0f ms" % total,
		Color(0.45, 0.85, 0.55) if total < 500.0 else Color(0.95, 0.72, 0.35))

	var pdm = reply.get("drift_score")
	_set_metric("pdm", "n/a" if pdm == null else "%.3f" % float(pdm))

	# KBD is null when the reply made no checkable factual claim, which is
	# common for short in-character lines. That is "nothing to score", not
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
		Color(0.55, 0.58, 0.66) if leaks.is_empty() else Color(1.0, 0.45, 0.45))


func _set_metric(key: String, value: String, color: Color = Color(0.85, 0.88, 0.94)) -> void:
	var label: Label = _metrics[key]
	label.text = "%s: %s" % [label.get_meta("caption"), value]
	label.add_theme_color_override("font_color", color)


func set_status(online: bool, detail: String) -> void:
	_status.text = ("● model server: %s" if online else "○ model server offline — %s") % detail
	_status.add_theme_color_override(
		"font_color", Color(0.45, 0.85, 0.55) if online else Color(0.95, 0.55, 0.45)
	)
