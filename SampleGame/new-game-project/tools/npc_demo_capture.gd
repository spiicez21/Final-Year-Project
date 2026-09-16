extends SceneTree

# Dev-only: plays a short scripted conversation and saves a screenshot with the
# HUD filled in (reply, live metrics, what the NPC remembers). Used for slides
# and documentation. Needs the model server running.
#
#   set CAPTURE_DIR=<dir> & godot --path . --script res://tools/npc_demo_capture.gd --resolution 1600x900
#
# Starts from a clean memory so the capture shows learning, not an old save.

const NPC := "Prof. Adeyemi"
const LINES := ["my name is yugabharathi", "i am class cse d", "what is my name"]


func _initialize() -> void:
	_run.call_deferred()


func _settle(frames: int) -> void:
	for i in range(frames):
		await process_frame


func _run() -> void:
	var scene: Node = load("res://Scene/Main.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	await _settle(150)

	var director = scene.get_node("NpcDirector")
	var player = scene.get_node("Player")
	var npc = null
	for candidate in director._npcs:
		if candidate.display_name == NPC:
			npc = candidate
	if npc == null:
		print("NPC not found -- is the model server running?")
		quit(1)
		return

	director._memory.forget_all()
	player.global_position = npc.global_position + Vector3(1.8, 0.0, -2.0)
	await _settle(20)
	director._open(npc)
	await _settle(10)
	for line in LINES:
		director._on_message_submitted(line)
		var waited := 0
		while director._awaiting and waited < 3600:
			await process_frame
			waited += 1
		await _settle(10)
		print("  you: %s\n  %s: %s" % [line, NPC, npc.history[-1]["content"] if not npc.history.is_empty() else "<none>"])
	print("  memory: %s" % director._memory.get_for(NPC))

	await _settle(40)
	var out_dir := OS.get_environment("CAPTURE_DIR")
	if not out_dir.is_empty():
		DirAccess.make_dir_recursive_absolute(out_dir)
		var path := "%s/demo_capture.png" % out_dir
		root.get_texture().get_image().save_png(path)
		print("  saved %s" % path)
	director._memory.forget_all()   # leave no test memory behind in the player's save
	quit(0)
