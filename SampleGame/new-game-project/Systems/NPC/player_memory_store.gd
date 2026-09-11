extends RefCounted
class_name PlayerMemoryStore

## What each NPC remembers about the player, kept between play sessions.
##
## Per NPC, not global. Each character knows only what the player told *that*
## character: tell Prof. Adeyemi your name and Halvorsen still has to ask. That
## is how people work, and it keeps the memory honest to test -- an NPC
## knowing something it was never told is a leak, the same failure KBD
## measures for world facts.
##
## The server does the learning (backend/player_memory.py extracts facts from
## what the player says and returns the merged memory); this class only keeps
## the result and writes it to disk after every change. The file is plain JSON
## so a tester can read, edit or seed it by hand:
##
##   %APPDATA%/Godot/app_userdata/<project>/npc_memory.json
##   {"Prof. Adeyemi": {"name": "Priya", "project": "robotics"}}
##
## Start the game with `-- --forget` to begin with nobody remembering you.

const PATH := "user://npc_memory.json"

var _by_npc: Dictionary = {}


func _init() -> void:
	if "--forget" in OS.get_cmdline_user_args():
		print("PlayerMemoryStore: --forget given, starting with no memories.")
		save()
		return
	_load()


func get_for(npc_name: String) -> Dictionary:
	return _by_npc.get(npc_name, {})


## Replaces an NPC's memory with the server's merged copy. Saves only when
## something actually changed, so ordinary small talk does not touch the disk.
func set_for(npc_name: String, memory: Dictionary) -> bool:
	if memory == get_for(npc_name):
		return false
	_by_npc[npc_name] = memory
	save()
	return true


func forget(npc_name: String) -> void:
	_by_npc.erase(npc_name)
	save()


func forget_all() -> void:
	_by_npc.clear()
	save()


func save() -> void:
	var file := FileAccess.open(PATH, FileAccess.WRITE)
	if file == null:
		push_warning("PlayerMemoryStore: cannot write %s (%s)"
			% [PATH, error_string(FileAccess.get_open_error())])
		return
	file.store_string(JSON.stringify(_by_npc, "  "))


func _load() -> void:
	if not FileAccess.file_exists(PATH):
		return
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(PATH))
	# A hand-edited file with a typo should cost the memories, not the game.
	if parsed is Dictionary:
		_by_npc = parsed
	else:
		push_warning("PlayerMemoryStore: %s is not a JSON object; ignoring it." % PATH)
