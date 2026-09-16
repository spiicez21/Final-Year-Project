extends CanvasLayer
class_name NpcSettings

## Settings panel: which microphone to talk with, and which keys do what.
##
## Opened with F2 (anywhere) or "/settings" in a conversation. Saved to
## user://settings.cfg and applied at startup by apply_saved(), before any
## NPC can be talked to.
##
## The microphone list is the operating system's input devices
## (AudioServer.get_input_device_list); "Default" follows whatever Windows
## calls the default recording device. A level bar shows the chosen mic live
## while the panel is open, so "is it hearing me?" can be answered before
## trying to talk to an NPC.
##
## Keys are stored as {"type": "key", "physical": int} or
## {"type": "mouse", "button": int}: physical keys, so a binding means the
## same place on the keyboard whatever the layout.

signal closed
signal bindings_changed

const PATH := "user://settings.cfg"
const OPEN_ACTION := &"open_settings"

## [action, caption, default event]
const REBINDABLE := [
	[&"push_to_talk", "Push to talk (hold)", {"type": "key", "physical": KEY_TAB}],
	[&"interact", "Talk to NPC", {"type": "key", "physical": KEY_E}],
]

const TEXT := Color(0.96, 0.97, 0.99)
const TEXT_DIM := Color(0.74, 0.78, 0.85)
const TEXT_MUTED := Color(0.56, 0.60, 0.68)
const WARN := Color(1.00, 0.78, 0.42)

## Set by the director so the level bar can listen to the chosen microphone.
var voice_input: VoiceInput

var _panel: PanelContainer
var _devices: OptionButton
var _level: ProgressBar
var _note: Label
var _key_buttons := {}
var _waiting_for: StringName = &""


# --- saved settings, usable without the panel --------------------------------

static func ensure_actions() -> void:
	if not InputMap.has_action(OPEN_ACTION):
		InputMap.add_action(OPEN_ACTION)
		InputMap.action_add_event(OPEN_ACTION, event_from({"type": "key", "physical": KEY_F2}))
	for row in REBINDABLE:
		if not InputMap.has_action(row[0]):
			InputMap.add_action(row[0])
			InputMap.action_add_event(row[0], event_from(row[2]))


static func apply_saved() -> void:
	ensure_actions()
	var cfg := ConfigFile.new()
	if cfg.load(PATH) != OK:
		return
	var device := str(cfg.get_value("audio", "input_device", "Default"))
	if device in AudioServer.get_input_device_list():
		AudioServer.input_device = device
	for row in REBINDABLE:
		var saved = cfg.get_value("keys", String(row[0]), null)
		if saved is Dictionary and event_from(saved) != null:
			_bind(row[0], event_from(saved))


static func event_from(d: Dictionary) -> InputEvent:
	match str(d.get("type", "")):
		"key":
			var key := InputEventKey.new()
			key.physical_keycode = int(d.get("physical", 0)) as Key
			return key
		"mouse":
			var button := InputEventMouseButton.new()
			button.button_index = int(d.get("button", 0)) as MouseButton
			return button
	return null


static func describe(event: InputEvent) -> Dictionary:
	if event is InputEventKey:
		var physical: int = event.physical_keycode if event.physical_keycode != 0 else event.keycode
		return {"type": "key", "physical": physical}
	if event is InputEventMouseButton:
		return {"type": "mouse", "button": event.button_index}
	return {}


## "Tab", "E", "Mouse 4" -- for hints like "hold Tab to talk".
static func key_name(action: StringName) -> String:
	if not InputMap.has_action(action):
		return "?"
	for event in InputMap.action_get_events(action):
		if event is InputEventKey:
			var physical: Key = event.physical_keycode if event.physical_keycode != 0 else event.keycode
			return OS.get_keycode_string(DisplayServer.keyboard_get_keycode_from_physical(physical))
		if event is InputEventMouseButton:
			return "Mouse %d" % event.button_index
	return "unbound"


static func _bind(action: StringName, event: InputEvent) -> void:
	InputMap.action_erase_events(action)
	InputMap.action_add_event(action, event)


static func _save() -> void:
	var cfg := ConfigFile.new()
	cfg.set_value("audio", "input_device", AudioServer.input_device)
	for row in REBINDABLE:
		var events := InputMap.action_get_events(row[0])
		if not events.is_empty():
			cfg.set_value("keys", String(row[0]), describe(events[0]))
	var err := cfg.save(PATH)
	if err != OK:
		push_warning("NpcSettings: cannot write %s (%s)" % [PATH, error_string(err)])


# --- panel -------------------------------------------------------------------

func _ready() -> void:
	layer = 20
	visible = false
	ensure_actions()
	_build()


func is_open() -> bool:
	return visible


func open() -> void:
	_refresh_devices()
	_refresh_keys()
	_note.text = ""
	visible = true
	if voice_input:
		voice_input.set_monitoring(true)
	_devices.grab_focus()


func close() -> void:
	if not visible:
		return
	_waiting_for = &""
	visible = false
	if voice_input:
		voice_input.set_monitoring(false)
	closed.emit()


func _process(_delta: float) -> void:
	if visible and voice_input:
		# Peak is 0..1; a square root spreads quiet speech across the bar.
		_level.value = sqrt(voice_input.level()) * 100.0


func _input(event: InputEvent) -> void:
	if not visible:
		return
	if _waiting_for != &"":
		_capture_binding(event)
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed(&"ui_cancel") or event.is_action_pressed(OPEN_ACTION):
		close()
		get_viewport().set_input_as_handled()


func _capture_binding(event: InputEvent) -> void:
	var candidate: InputEvent = null
	if event is InputEventKey and event.pressed and not event.echo:
		if event.physical_keycode == KEY_ESCAPE or event.keycode == KEY_ESCAPE:
			_waiting_for = &""
			_note.text = "Unchanged."
			_refresh_keys()
			return
		candidate = event_from(describe(event))
	elif event is InputEventMouseButton and event.pressed:
		# Left and right click run the menus and the camera; the wheel has no
		# "held" state. Side buttons and middle click are fine.
		if event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_WHEEL_UP,
				MOUSE_BUTTON_WHEEL_DOWN, MOUSE_BUTTON_WHEEL_LEFT, MOUSE_BUTTON_WHEEL_RIGHT]:
			return
		candidate = event_from(describe(event))
	if candidate == null:
		return

	var clash := _clashing_action(candidate, _waiting_for)
	if not clash.is_empty():
		_note.text = "%s is already used for %s. Press another key, or Esc." % [
			_event_name(candidate), clash]
		return
	_bind(_waiting_for, candidate)
	var action := _waiting_for
	_waiting_for = &""
	_save()
	_refresh_keys()
	_note.text = "Saved."
	if action == &"push_to_talk" and candidate is InputEventKey and _is_typing_key(candidate):
		_note.text = "Saved. While you hold it in a conversation, this key is not typed into the chat box."
	bindings_changed.emit()


## Another action this event already triggers, as a readable name, or "".
## ui_* actions are skipped except the ones a player actually uses (Esc,
## Enter): Godot binds Tab and the arrows to focus navigation by default, and
## a conversation never needs those.
func _clashing_action(event: InputEvent, action: StringName) -> String:
	for other in InputMap.get_actions():
		if other == action:
			continue
		var name := String(other)
		if name.begins_with("ui_") and not name in ["ui_cancel", "ui_accept"]:
			continue
		if InputMap.event_is_action(event, other, true):
			for row in REBINDABLE:
				if row[0] == other:
					return "\"%s\"" % row[1]
			return {"ui_cancel": "closing menus (Esc)", "ui_accept": "sending a line (Enter)",
				"open_settings": "opening settings"}.get(name, name.replace("_", " "))
	return ""


func _is_typing_key(event: InputEventKey) -> bool:
	var keycode := DisplayServer.keyboard_get_keycode_from_physical(event.physical_keycode)
	return (keycode >= KEY_A and keycode <= KEY_Z) or (keycode >= KEY_0 and keycode <= KEY_9) \
		or keycode == KEY_SPACE


func _event_name(event: InputEvent) -> String:
	if event is InputEventKey:
		return OS.get_keycode_string(DisplayServer.keyboard_get_keycode_from_physical(event.physical_keycode))
	if event is InputEventMouseButton:
		return "Mouse %d" % event.button_index
	return "That"


func _build() -> void:
	var shade := ColorRect.new()
	shade.color = Color(0, 0, 0, 0.45)
	shade.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(shade)

	var center := CenterContainer.new()
	center.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(center)

	_panel = PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.07, 0.08, 0.11, 0.96)
	style.set_corner_radius_all(14)
	style.set_content_margin_all(22)
	style.border_color = Color(1, 1, 1, 0.08)
	style.set_border_width_all(1)
	_panel.add_theme_stylebox_override("panel", style)
	_panel.custom_minimum_size = Vector2(440, 0)
	center.add_child(_panel)

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 10)
	_panel.add_child(box)

	var title := HBoxContainer.new()
	title.add_child(_text("Settings", 18, TEXT))
	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	title.add_child(spacer)
	title.add_child(_text("F2 / Esc to close", 11, TEXT_MUTED))
	box.add_child(title)

	box.add_child(_text("MICROPHONE", 10, TEXT_MUTED))
	_devices = OptionButton.new()
	_devices.item_selected.connect(_on_device_selected)
	box.add_child(_devices)
	_level = ProgressBar.new()
	_level.show_percentage = false
	_level.custom_minimum_size = Vector2(0, 8)
	var fill := StyleBoxFlat.new()
	fill.bg_color = Color(0.52, 0.91, 0.63)
	fill.set_corner_radius_all(4)
	var track := StyleBoxFlat.new()
	track.bg_color = Color(1, 1, 1, 0.08)
	track.set_corner_radius_all(4)
	_level.add_theme_stylebox_override("fill", fill)
	_level.add_theme_stylebox_override("background", track)
	box.add_child(_level)
	box.add_child(_text("Say something: the bar should move. If it doesn't, pick another microphone,\n"
		+ "or allow desktop apps to use the microphone in Windows privacy settings.", 11, TEXT_DIM))

	box.add_child(_text("KEYS", 10, TEXT_MUTED))
	for row in REBINDABLE:
		var line := HBoxContainer.new()
		var caption := _text(row[1], 13, TEXT)
		caption.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		line.add_child(caption)
		var button := Button.new()
		button.custom_minimum_size = Vector2(150, 0)
		button.pressed.connect(_start_rebind.bind(row[0]))
		line.add_child(button)
		_key_buttons[row[0]] = button
		box.add_child(line)

	_note = _text("", 11, WARN)
	_note.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(_note)

	var buttons := HBoxContainer.new()
	buttons.add_theme_constant_override("separation", 10)
	var reset := Button.new()
	reset.text = "Reset to defaults"
	reset.pressed.connect(_reset)
	buttons.add_child(reset)
	var gap := Control.new()
	gap.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	buttons.add_child(gap)
	var done := Button.new()
	done.text = "Done"
	done.pressed.connect(close)
	buttons.add_child(done)
	box.add_child(buttons)


func _text(value: String, size: int, color: Color) -> Label:
	var l := Label.new()
	l.text = value
	l.add_theme_font_size_override("font_size", size)
	l.add_theme_color_override("font_color", color)
	return l


func _refresh_devices() -> void:
	_devices.clear()
	var current := AudioServer.input_device
	var list := AudioServer.get_input_device_list()
	for i in list.size():
		_devices.add_item(list[i])
		if list[i] == current:
			_devices.select(i)
	if list.size() <= 1:
		_note.text = "Only the default microphone was found."


func _on_device_selected(index: int) -> void:
	AudioServer.input_device = _devices.get_item_text(index)
	if voice_input:
		voice_input.restart_monitoring()
	_save()
	_note.text = "Microphone: %s" % AudioServer.input_device


func _refresh_keys() -> void:
	for action in _key_buttons:
		var button: Button = _key_buttons[action]
		button.text = "press a key…" if action == _waiting_for else key_name(action)


func _start_rebind(action: StringName) -> void:
	_waiting_for = action
	_note.text = "Press the new key or mouse button (Esc to cancel)."
	_refresh_keys()
	# Keep the button from grabbing the key press as a click.
	(_key_buttons[action] as Button).release_focus()


func _reset() -> void:
	_waiting_for = &""
	for row in REBINDABLE:
		_bind(row[0], event_from(row[2]))
	AudioServer.input_device = "Default"
	if voice_input:
		voice_input.restart_monitoring()
	_save()
	_refresh_devices()
	_refresh_keys()
	_note.text = "Defaults restored."
	bindings_changed.emit()
