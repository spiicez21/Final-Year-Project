extends SceneTree

## One-off: writes the InputMap and main scene into project.godot.
##
## The movement/camera scripts have always referenced these actions, but the
## InputMap section was never committed, so the player could not move. Written
## through ProjectSettings rather than by hand-editing project.godot because
## the InputEventKey serialisation format is easy to get subtly wrong.
##
##   godot --headless --path . --script res://tools/_setup_input.gd

const ACTIONS := {
	"move_left": [KEY_A, KEY_LEFT],
	"move_right": [KEY_D, KEY_RIGHT],
	"move_forward": [KEY_W, KEY_UP],
	"move_back": [KEY_S, KEY_DOWN],
	"sprint": [KEY_SHIFT],
	"jump": [KEY_SPACE],
	"interact": [KEY_E],
}


func _initialize() -> void:
	for action_name in ACTIONS:
		var events: Array = []
		for keycode in ACTIONS[action_name]:
			var event := InputEventKey.new()
			# physical_keycode keeps WASD in the same physical place on
			# AZERTY/QWERTZ layouts.
			event.physical_keycode = keycode
			events.append(event)
		ProjectSettings.set_setting("input/" + action_name,
			{"deadzone": 0.2, "events": events})
		print("  + %s" % action_name)

	ProjectSettings.set_setting("application/run/main_scene", "res://Scene/Main.tscn")
	print("  + main_scene = res://Scene/Main.tscn")

	var err := ProjectSettings.save()
	print("save: ", "OK" if err == OK else "FAILED %d" % err)
	quit(0 if err == OK else 1)
