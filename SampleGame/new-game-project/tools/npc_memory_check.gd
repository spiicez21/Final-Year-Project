extends SceneTree

# Dev-only: checks player memory end to end, across a real restart.
# Needs the model server running. Run twice, as two separate processes:
#
#   set MEMORY_PHASE=teach  & godot --path . --script res://tools/npc_memory_check.gd
#   set MEMORY_PHASE=recall & godot --path . --script res://tools/npc_memory_check.gd
#
# teach   wipes all memories, introduces the player to Prof. Adeyemi only,
#         and checks the fact was learned, saved, and not given to anyone else
# recall  a fresh process: checks the save was loaded, then asks Adeyemi (who
#         was told) and Halvorsen (who was not) to recall the player
#
# Optional CAPTURE_DIR=<dir> saves a screenshot of each phase.

const TOLD := "Prof. Adeyemi"
const NOT_TOLD := "Halvorsen"
const INTRO := "Hi, I'm Priya, a first-year. I'm thinking of doing a final-year project on robotics."
const RECALL := "who am I again?"

var _failures := 0


func _initialize() -> void:
	_run.call_deferred()


func _settle(frames: int) -> void:
	for i in range(frames):
		await process_frame


func _check(ok: bool, what: String) -> void:
	print("  %s  %s" % ["PASS" if ok else "FAIL", what])
	if not ok:
		_failures += 1


func _find(director, npc_name: String):
	for npc in director._npcs:
		if npc.display_name == npc_name:
			return npc
	return null


func _say(director, player, npc, text: String) -> String:
	player.global_position = npc.global_position + Vector3(1.8, 0.0, -2.0)
	await _settle(10)
	director._open(npc)
	await _settle(5)
	director._on_message_submitted(text)
	var waited := 0
	while director._awaiting and waited < 3600:
		await process_frame
		waited += 1
	await _settle(5)
	var reply: String = npc.history[-1]["content"] if not npc.history.is_empty() else "<none>"
	print("    you: %s\n    %s: %s" % [text, npc.display_name, reply])
	return reply


func _run() -> void:
	var phase := OS.get_environment("MEMORY_PHASE")
	var scene: Node = load("res://Scene/Main.tscn").instantiate()
	root.add_child(scene)
	current_scene = scene
	await _settle(150)

	var director = scene.get_node("NpcDirector")
	var player = scene.get_node("Player")
	var told = _find(director, TOLD)
	var not_told = _find(director, NOT_TOLD)
	if told == null or not_told == null:
		print("NPCs missing -- is the model server running?")
		quit(1)
		return

	if phase == "teach":
		print("\n=== teach ===")
		director._memory.forget_all()
		await _say(director, player, told, INTRO)
		var learned: Dictionary = director._memory.get_for(TOLD)
		_check(learned.get("name", "") == "Priya", "%s learned the name (%s)" % [TOLD, learned])
		_check(learned.get("project", "") == "robotics", "%s learned the project" % TOLD)
		_check(director._memory.get_for(NOT_TOLD).is_empty(), "%s learned nothing" % NOT_TOLD)
		var saved = JSON.parse_string(FileAccess.get_file_as_string(PlayerMemoryStore.PATH))
		_check(saved is Dictionary and saved.has(TOLD) and not saved.has(NOT_TOLD),
			"saved to %s" % ProjectSettings.globalize_path(PlayerMemoryStore.PATH))
	elif phase == "recall":
		print("\n=== recall (new process) ===")
		var loaded: Dictionary = director._memory.get_for(TOLD)
		_check(loaded.get("name", "") == "Priya", "memory loaded from disk (%s)" % loaded)
		_check(told.history.is_empty(), "no transcript carried over -- memory is all it has")
		var reply := await _say(director, player, told, RECALL)
		_check("priya" in reply.to_lower(), "%s recalls the player" % TOLD)
		director._close()
		var other := await _say(director, player, not_told, "can you recall my name?")
		_check(not ("priya" in other.to_lower()), "%s does not know the name" % NOT_TOLD)
		# Leave the told NPC open so the screenshot shows its memory panel.
		director._close()
		player.global_position = told.global_position + Vector3(1.8, 0.0, -2.0)
		await _settle(10)
		director._open(told)
		director._hud.show_reply(RECALL, reply)
	else:
		print("set MEMORY_PHASE to teach or recall")
		quit(2)
		return

	await _settle(30)
	var dir_out: String = OS.get_environment("CAPTURE_DIR")
	if not dir_out.is_empty():
		DirAccess.make_dir_recursive_absolute(dir_out)
		root.get_texture().get_image().save_png("%s/memory_%s.png" % [dir_out, phase])
	print("\n%s" % ("ALL PASSED" if _failures == 0 else "%d FAILED" % _failures))
	quit(0 if _failures == 0 else 1)
