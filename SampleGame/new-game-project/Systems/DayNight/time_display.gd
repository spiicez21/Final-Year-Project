extends Label
## Shows the current time and lets you scrub it while testing.
##   [  and  ]  step the clock back / forward an hour
##   P          pause or resume the cycle

@export var cycle_path: NodePath
@export var show_hint: bool = true

var _cycle: Node = null


func _ready() -> void:
	_cycle = get_node_or_null(cycle_path)
	if _cycle == null:
		# Fall back to finding it anywhere under the current scene.
		for node in get_tree().get_nodes_in_group("day_night"):
			_cycle = node
			break
	set_process(_cycle != null)
	visible = _cycle != null


func _process(_delta: float) -> void:
	var text_value: String = _cycle.time_string()
	if not _cycle.running:
		text_value += "  (paused)"
	if show_hint:
		text_value += "\n[ / ]  scrub    P  pause"
	text = text_value


func _unhandled_key_input(event: InputEvent) -> void:
	if _cycle == null:
		return
	var key := event as InputEventKey
	if key == null or not key.pressed or key.echo:
		return
	# _cycle is a plain Node, so its properties come back as Variant -- pull them
	# into typed locals before doing float maths with them.
	var now: float = _cycle.time_of_day
	var first: float = _cycle.start_hour
	var last: float = _cycle.end_hour
	match key.keycode:
		KEY_BRACKETLEFT:
			_cycle.time_of_day = maxf(first, now - 1.0)
		KEY_BRACKETRIGHT:
			_cycle.time_of_day = minf(last, now + 1.0)
		KEY_P:
			_cycle.running = not _cycle.running
