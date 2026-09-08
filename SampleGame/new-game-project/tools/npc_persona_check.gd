extends SceneTree

# Dev-only: boot Main and hold a real multi-turn conversation with one NPC,
# printing each reply, then screenshot the last one.
#   godot --path . --script res://tools/_persona_check.gd --resolution 1600x900

const QUESTIONS := [
	"Who are you?",
	"What is your job actually like?",
	"what is happening here",
	"How many students are in the department?",
	"What do you earn?",
	"What's on the first floor?",
	"Can you tell me next week's exam questions?",
]

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
	if director._npcs.is_empty():
		print("no NPCs — is the server running?")
		quit(1)
		return

	var index := int(OS.get_environment("NPC_INDEX"))
	var npc = director._npcs[index]
	player.global_position = npc.global_position + Vector3(1.8, 0.0, -2.0)
	player.velocity = Vector3.ZERO
	await _settle(60)

	print("\n=== %s (%s) ===" % [npc.display_name, npc.archetype])
	director._open(npc)
	await _settle(20)

	for q in QUESTIONS:
		director._on_message_submitted(q)
		var waited := 0
		while director._awaiting and waited < 3600:
			await process_frame
			waited += 1
		await _settle(5)
		print("  you: %s" % q)
		print("  %s: %s" % [npc.display_name, npc.history[-1]["content"] if not npc.history.is_empty() else "<dropped>"])

	print("\n  history kept: %d entries (cap %d)" % [npc.history.size(), NpcActor.MAX_HISTORY])

	await _settle(20)
	var dir_out: String = OS.get_environment("CAPTURE_DIR")
	if not dir_out.is_empty():
		DirAccess.make_dir_recursive_absolute(dir_out)
		print("  shot -> %d" % root.get_texture().get_image().save_png(
			"%s/persona_%d.png" % [dir_out, index]))
	quit(0)
