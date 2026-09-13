extends CanvasLayer
class_name NpcDialogueHud

## Dialogue box + live metrics readout.
##
## Built in code and anchored rather than authored as a .tscn with fixed
## coordinates, so it survives the window being resized (the project uses
## stretch mode "canvas_items" with an "expand" aspect, so the viewport really
## does change shape).
##
## All offsets here are in the project's 1152x648 UI space, NOT window pixels:
## the project stretches "canvas_items", so on a 1600x900 window these are
## scaled by ~1.39 on the way to the screen. A number that looks too small here
## is probably right.
##
## Panels are frosted glass (see glass_panel.gdshader) rather than solid fills.
## Straight alpha over this game does not work -- the concrete corridor reads
## straight through and fights the text -- so what is behind a panel is blurred
## to give the letterforms a calm ground while keeping the scene visible.
##
## The metrics are the point of the demo as much as the dialogue is: they are
## the paper's own numbers, computed per turn on the reply you just read, so
## they get their own panel rather than being tucked away as debug text.

signal message_submitted(text: String)

const HERO_FRAMES := "res://Resources/Characters/hero_frames.tres"
const GLASS_SHADER := "res://Systems/NPC/glass_panel.gdshader"

# One palette, used everywhere, so panels and text cannot drift apart as the
# UI is edited. Text runs bright because it sits on blurred, unpredictable
# background.
const TEXT := Color(0.96, 0.97, 0.99)
const TEXT_DIM := Color(0.74, 0.78, 0.85)
const TEXT_MUTED := Color(0.56, 0.60, 0.68)
const GOOD := Color(0.52, 0.91, 0.63)
const WARN := Color(1.00, 0.78, 0.42)
const BAD := Color(1.00, 0.52, 0.52)

var _panel: Control
var _portrait: TextureRect
var _speaker: Label
var _role: Label
var _said: Label
var _body: RichTextLabel
var _input: LineEdit
var _status: Label
var _metrics := {}
var _memory: RichTextLabel

## Display order and captions for player_memory slots (backend/dialogue/memory.py).
const MEMORY_ROWS := [
	["name", "name"], ["year", "year"], ["studies", "studies"], ["from", "from"],
	["project", "project"], ["interests", "into"], ["feeling", "felt"],
]

# "…thinking" animates rather than sitting still, so a slow first turn reads as
# working rather than as frozen.
var _pending := false
var _pending_t := 0.0

## Glass panels needing their size pushed into the shader, as
## {holder, material} pairs. See _sync_glass().
var _glass_panels: Array = []


func _ready() -> void:
	layer = 10
	_build_status()
	_build_metrics()
	_build_dialogue()
	set_dialogue_visible(false)


func _process(delta: float) -> void:
	_sync_glass()
	if not _pending:
		return
	_pending_t += delta
	_body.text = "[color=#8b93a1]%s[/color]" % ("thinking" + ".".repeat(1 + int(_pending_t * 2.5) % 3))


## Keeps each glass shader's `panel_size` in step with its Control. Only
## writes when the size actually changed.
func _sync_glass() -> void:
	for entry in _glass_panels:
		var holder: Control = entry["holder"]
		# Panels marked auto-height grow to fit their content. A plain Control
		# does not size itself the way a PanelContainer would, so without this
		# the last metrics row was simply clipped off the bottom.
		if entry.get("auto_height", false):
			var content: Control = entry["content"]
			var wanted: float = content.get_combined_minimum_size().y
			if wanted > 0.0 and absf(holder.size.y - wanted) > 0.5:
				holder.offset_bottom = holder.offset_top + wanted
		if entry.get("last") != holder.size:
			entry["last"] = holder.size
			entry["mat"].set_shader_parameter("panel_size", holder.size)


# --- glass ------------------------------------------------------------------

## Builds an anchored frosted panel; returns the holder to position and the
## container to put content into.
##
## The glass has to be a SIBLING of the content, not a child of a
## PanelContainer. A PanelContainer insets its children by the stylebox's
## content margins, so a full-rect ColorRect placed inside one comes out
## smaller than the panel itself and the text ends up sitting on the blurred
## edge — which is exactly what the first version did. Holder holding
## {glass, MarginContainer} keeps the backing at full size while the margins
## apply only to the content.
func _glass(radius: float, tint: Color, margin: int, blur: float = 2.6) -> Dictionary:
	var holder := Control.new()
	holder.mouse_filter = Control.MOUSE_FILTER_IGNORE

	var shader := load(GLASS_SHADER) as Shader
	if shader == null:
		# No shader means no glass, but the UI must still work; fall back to a
		# plain translucent fill rather than an invisible panel.
		var flat := StyleBoxFlat.new()
		flat.bg_color = Color(0.05, 0.06, 0.09, 0.85)
		flat.set_corner_radius_all(int(radius))
		var fallback := Panel.new()
		fallback.set_anchors_preset(Control.PRESET_FULL_RECT)
		fallback.add_theme_stylebox_override("panel", flat)
		holder.add_child(fallback)
	else:
		var mat := ShaderMaterial.new()
		mat.shader = shader
		mat.set_shader_parameter("radius", radius)
		mat.set_shader_parameter("tint", tint)
		mat.set_shader_parameter("blur", blur)

		var glass := ColorRect.new()
		glass.material = mat
		glass.set_anchors_preset(Control.PRESET_FULL_RECT)
		glass.mouse_filter = Control.MOUSE_FILTER_IGNORE
		holder.add_child(glass)

		# A shader cannot ask its Control how big it is, so the size is pushed
		# in. Relying on the `resized` signal alone is not enough: it did not
		# fire for the metrics panel, leaving panel_size at (0,0) — at which
		# size the rounded-box test discards every pixel and the panel simply
		# vanishes. Checking each frame is a couple of comparisons and cannot
		# miss a layout pass.
		_glass_panels.append({"holder": holder, "mat": mat, "content": null})

	var content := MarginContainer.new()
	content.set_anchors_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		content.add_theme_constant_override("margin_" + side, margin)
	holder.add_child(content)
	for entry in _glass_panels:
		if entry["holder"] == holder:
			entry["content"] = content

	return {"root": holder, "content": content}


func _label(text: String, size: int, color: Color) -> Label:
	var l := Label.new()
	l.text = text
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	# Everything sits on a blurred scene, so a soft shadow keeps text legible
	# regardless of what the player happens to be standing in front of.
	l.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.55))
	l.add_theme_constant_override("shadow_offset_x", 0)
	l.add_theme_constant_override("shadow_offset_y", 1)
	return l


# --- status line -----------------------------------------------------------

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


# --- metrics ---------------------------------------------------------------

func _build_metrics() -> void:
	var g := _glass(16.0, Color(0.05, 0.06, 0.09, 0.55), 18)
	var panel: Control = g["root"]
	# Anchored under the minimap, which occupies offset_top 16..212 in the
	# top-right corner (see the Minimap node in Scene/Main.tscn).
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.offset_left = -300.0
	panel.offset_right = -20.0
	panel.offset_top = 226.0
	panel.offset_bottom = 226.0 + 200.0   # replaced on the first sync
	add_child(panel)
	for entry in _glass_panels:
		if entry["holder"] == panel:
			entry["auto_height"] = true

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 9)
	g["content"].add_child(box)

	box.add_child(_label("LIVE METRICS", 10, TEXT_MUTED))

	# The tooltip carries the definition: "0.41" means nothing to someone
	# seeing the demo for the first time.
	var rows := [
		["switch", "model switch", "Time to make this NPC's model resident.\n0ms = already warm in the server's LRU pool.\nNOTE: these are merged GGUFs, so this is a\nwhole-model swap, not a LoRA set_adapter() call."],
		["gen", "generation", "Wall-clock time for the model to produce the reply."],
		["total", "total", "What the player actually waits: switch + generation.\nGreen when under the 500ms real-time target (RQ4)."],
		["pdm", "PDM v2 drift", "Domain-agnostic persona drift (RQ3).\nLower = more consistent with the archetype."],
		["kbd", "KBD", "Knowledge Boundary Drift (C1): fraction of factual\nreferences falling outside this NPC's visibility set.\n'n/a' = the reply stated no checkable fact."],
		["leak", "leaked facts", "knowledge_base.json ids this NPC should not know."],
		["turn", "understood as", "What the server took your line to be (greeting, recall,\na question about the NPC, ...). It decides what context\nthe reply gets. See backend/dialogue/intent.py."],
		["guard", "reply guard", "What the server fixed in this reply before you saw it:\nrepeat, recall, fact, self, introduced, name, ...\n'clean' = the first reply passed every check.\nSee backend/dialogue/guard.py."],
	]
	for row in rows:
		# Label left, value hard right. A two-column row keeps the numbers on
		# a common edge, which a single "label: value" string cannot do.
		var line := HBoxContainer.new()
		line.tooltip_text = row[2]
		line.mouse_filter = Control.MOUSE_FILTER_STOP
		line.add_theme_constant_override("separation", 10)

		var caption := _label(row[1], 12, TEXT_DIM)
		caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		line.add_child(caption)

		var value := _label("—", 12, TEXT)
		value.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		line.add_child(value)

		box.add_child(line)
		_metrics[row[0]] = value
	reset_metrics()

	# What this NPC has learned about the player. In the metrics column rather
	# than the dialogue box because it is the same kind of thing: state the
	# model is conditioned on, shown so a tester can see *why* an NPC said
	# "Priya" rather than taking it on faith.
	var spacer := Control.new()
	spacer.custom_minimum_size = Vector2(0, 4)
	box.add_child(spacer)
	var heading := _label("REMEMBERS YOU", 10, TEXT_MUTED)
	heading.tooltip_text = ("Facts this NPC picked up from what you told it, and only\n"
		+ "this NPC. Saved between sessions. Type /forget to reset.")
	heading.mouse_filter = Control.MOUSE_FILTER_STOP
	box.add_child(heading)
	_memory = RichTextLabel.new()
	_memory.bbcode_enabled = true
	_memory.fit_content = true
	_memory.scroll_active = false
	_memory.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_memory.add_theme_font_size_override("normal_font_size", 12)
	_memory.add_theme_color_override("default_color", TEXT)
	box.add_child(_memory)
	show_memory({})


# --- dialogue --------------------------------------------------------------

func _build_dialogue() -> void:
	var g := _glass(18.0, Color(0.04, 0.05, 0.08, 0.58), 22)
	_panel = g["root"]
	_panel.anchor_top = 1.0
	_panel.anchor_bottom = 1.0
	_panel.anchor_right = 1.0
	# Stops short of the metrics column on the right. Full width looked better
	# in isolation but the taller panel then drew straight over the metrics
	# captions, leaving a column of values with nothing naming them.
	_panel.offset_left = 72.0
	_panel.offset_right = -336.0
	# Sized to the content rather than to taste: header, the player's line,
	# a wrapped reply and the input row come to roughly 270px, and anything
	# less pushed the input field out through the bottom of the glass.
	_panel.offset_top = -304.0
	_panel.offset_bottom = -32.0
	add_child(_panel)

	var col := VBoxContainer.new()
	col.add_theme_constant_override("separation", 14)
	g["content"].add_child(col)

	col.add_child(_build_header())
	col.add_child(_build_body())
	col.add_child(_build_input())


func _build_header() -> HBoxContainer:
	var header := HBoxContainer.new()
	header.add_theme_constant_override("separation", 14)

	# Portrait, reusing the same sprite sheet the NPC is drawn from, so the
	# figure in the box is literally the character in the corridor.
	_portrait = TextureRect.new()
	_portrait.custom_minimum_size = Vector2(40, 56)
	_portrait.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_portrait.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	header.add_child(_portrait)

	var names := VBoxContainer.new()
	names.alignment = BoxContainer.ALIGNMENT_CENTER
	names.add_theme_constant_override("separation", 3)
	names.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_speaker = _label("", 20, TEXT)
	_role = _label("", 12, TEXT_MUTED)
	names.add_child(_speaker)
	names.add_child(_role)
	header.add_child(names)

	return header


func _build_body() -> VBoxContainer:
	var body := VBoxContainer.new()
	body.add_theme_constant_override("separation", 10)
	body.size_flags_vertical = Control.SIZE_EXPAND_FILL

	# The player's own line, small and dim: context for the reply, not the
	# thing you are meant to read.
	_said = _label("", 12, TEXT_MUTED)
	_said.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	body.add_child(_said)

	_body = RichTextLabel.new()
	_body.bbcode_enabled = true
	_body.fit_content = false
	_body.scroll_following = true
	_body.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_body.add_theme_font_size_override("normal_font_size", 17)
	_body.add_theme_color_override("default_color", TEXT)
	_body.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.55))
	_body.add_theme_constant_override("shadow_offset_y", 1)
	_body.custom_minimum_size = Vector2(0, 84)
	body.add_child(_body)

	return body


func _build_input() -> HBoxContainer:
	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 14)

	_input = LineEdit.new()
	_input.placeholder_text = "Say something…"
	_input.custom_minimum_size = Vector2(0, 36)
	_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_input.add_theme_font_size_override("font_size", 14)
	_input.add_theme_color_override("font_color", TEXT)
	_input.add_theme_color_override("font_placeholder_color", TEXT_MUTED)

	# An underline rather than a boxed field: fewer edges competing with the
	# panel's own, and the focus state is still obvious.
	var rest := StyleBoxFlat.new()
	rest.bg_color = Color(1, 1, 1, 0.05)
	rest.set_corner_radius_all(8)
	rest.set_content_margin_all(10)
	rest.border_color = Color(1, 1, 1, 0.10)
	rest.border_width_bottom = 1

	var focused := rest.duplicate() as StyleBoxFlat
	focused.bg_color = Color(1, 1, 1, 0.08)
	focused.border_color = Color(0.55, 0.72, 1.0, 0.75)
	focused.border_width_bottom = 2

	_input.add_theme_stylebox_override("normal", rest)
	_input.add_theme_stylebox_override("focus", focused)
	_input.text_submitted.connect(_on_submit)
	row.add_child(_input)

	var hint := _label("Enter  ·  Esc", 11, TEXT_MUTED)
	hint.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	row.add_child(hint)

	return row


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
		_pending = false
		_input.release_focus()


func is_dialogue_visible() -> bool:
	return _panel.visible


func open_with(speaker: String, archetype: String, color: Color,
		role: String = "", portrait_tint: Color = Color.WHITE) -> void:
	_speaker.text = speaker
	_speaker.add_theme_color_override("font_color", color)
	_role.text = role if not role.is_empty() else archetype
	_set_portrait(portrait_tint)
	_pending = false
	_said.text = ""
	_body.text = "[color=#8b93a1]The %s turns to face you.[/color]" % archetype
	reset_metrics()
	set_dialogue_visible(true)


## Uses the first idle frame of the shared hero sheet, tinted to match the NPC
## in the world. Silently leaves the portrait blank if the sheet or animation
## is missing, rather than taking the dialogue box down with it.
func _set_portrait(tint: Color) -> void:
	var frames := load(HERO_FRAMES) as SpriteFrames
	if frames == null or not frames.has_animation(&"idle_s"):
		return
	if frames.get_frame_count(&"idle_s") <= 0:
		return
	_portrait.texture = frames.get_frame_texture(&"idle_s", 0)
	_portrait.modulate = tint


func show_pending(message: String) -> void:
	_input.editable = false
	_input.text = ""
	_said.text = "You:  %s" % message
	_pending = true
	_pending_t = 0.0


func show_reply(message: String, reply: String) -> void:
	_pending = false
	_input.editable = true
	_input.grab_focus()
	_said.text = "You:  %s" % message
	_body.text = reply


func show_error(message: String, error: String) -> void:
	_pending = false
	_input.editable = true
	_input.grab_focus()
	_said.text = "You:  %s" % message
	_body.text = "[color=#ff8585]%s[/color]" % error


## A system line in the reply area (command results), styled apart from
## anything an NPC says so it is never mistaken for dialogue.
func show_note(message: String, note: String) -> void:
	_pending = false
	_input.editable = true
	_input.text = ""
	_input.grab_focus()
	_said.text = "You:  %s" % message
	_body.text = "[color=#8b93a1][i]%s[/i][/color]" % note.replace("[", "[lb]")


## `updates` are the slots learned from the latest line; they are marked so
## the moment of learning is visible, which is most of what a tester wants
## to check.
func show_memory(memory: Dictionary, updates: Dictionary = {}) -> void:
	if _memory == null:
		return
	if memory.is_empty():
		_memory.text = "[color=#8e96a3]nothing yet[/color]"
		return
	var lines: Array[String] = []
	for row in MEMORY_ROWS:
		if not memory.has(row[0]):
			continue
		var value = memory[row[0]]
		# Values are the player's own words, so "[" is escaped rather than
		# trusted as markup.
		var shown := (", ".join(value) if value is Array else str(value)).replace("[", "[lb]")
		var fresh := "  [color=#85e8a1]new[/color]" if updates.has(row[0]) else ""
		lines.append("[color=#bdc6d8]%s[/color]  %s%s" % [row[1], shown, fresh])
	_memory.text = "\n".join(lines)


func reset_metrics() -> void:
	for key in _metrics:
		var label: Label = _metrics[key]
		label.text = "—"
		label.add_theme_color_override("font_color", TEXT_MUTED)


func update_metrics(reply: Dictionary) -> void:
	var switch_ms: float = reply.get("adapter_switch_ms", 0.0)
	var gen_ms: float = reply.get("generation_ms", 0.0)

	_set_metric("switch", "%.0f ms  %s" % [switch_ms, "warm" if switch_ms == 0.0 else "cold"])
	_set_metric("gen", "%.0f ms" % gen_ms)
	# RQ4's real-time target is <500ms, so the total is colour-coded against
	# it rather than left as a bare number.
	var total := switch_ms + gen_ms
	_set_metric("total", "%.0f ms" % total, GOOD if total < 500.0 else WARN)

	var pdm = reply.get("drift_score")
	_set_metric("pdm", "n/a" if pdm == null else "%.3f" % float(pdm))

	# KBD is null when the reply made no checkable factual claim, which is
	# common for short in-character lines. That is "nothing to score", not
	# "scored zero" — conflating the two would understate leakage.
	var kbd = reply.get("kbd")
	if kbd == null:
		_set_metric("kbd", "n/a", TEXT_MUTED)
	else:
		var value := float(kbd)
		_set_metric("kbd", "%.2f" % value, GOOD if value == 0.0 else BAD)

	var leaks: Array = reply.get("leaked_fact_ids", [])
	_set_metric("leak", "none" if leaks.is_empty() else ", ".join(leaks),
		TEXT_MUTED if leaks.is_empty() else BAD)

	# Older servers do not send these; show a dash rather than a wrong "clean".
	if reply.has("intent"):
		_set_metric("turn", str(reply.get("intent", "")).replace("_", " "), TEXT_DIM)
		var repairs: Array = reply.get("repairs", [])
		_set_metric("guard", "clean" if repairs.is_empty() else ", ".join(repairs),
			TEXT_MUTED if repairs.is_empty() else WARN)


func _set_metric(key: String, value: String, color: Color = TEXT) -> void:
	var label: Label = _metrics[key]
	label.text = value
	label.add_theme_color_override("font_color", color)


func set_status(online: bool, detail: String) -> void:
	_status.text = ("● model server: %s" if online else "○ model server offline — %s") % detail
	_status.add_theme_color_override("font_color", GOOD if online else Color(0.95, 0.55, 0.45))
