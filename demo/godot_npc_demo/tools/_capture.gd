extends SceneTree

# Temporary dev utility: render Main.tscn and save a PNG. Must run WITHOUT
# --headless (headless has no rendering device to read back from).

func _initialize() -> void:
	_run.call_deferred()

func _run() -> void:
	var scene: Node = load("res://scenes/Main.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	# Long enough for the /archetypes round trip to land and NPCs to spawn.
	for i in range(240):
		await process_frame
	var img: Image = root.get_texture().get_image()
	var out: String = OS.get_environment("CAPTURE_OUT")
	print("saving to ", out, " -> ", img.save_png(out))
	quit(0)
