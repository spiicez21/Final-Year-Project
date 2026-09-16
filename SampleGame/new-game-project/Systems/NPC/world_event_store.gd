extends RefCounted
class_name WorldEventStore

## News in the world: what players report, and which NPCs have heard it.
##
## When the player tells an NPC about something happening ("i saw a man
## carrying a knife near the gym"), the server marks the incident and its
## place (backend/dialogue/events.py) and returns it; this store keeps it. The
## NPC that was told knows it at once. Afterwards spread() passes news from an NPC
## who knows it to another NPC of the same archetype, a step at a time. Rumours
## stay within a character type, so police news cannot reach teaching staff.
##
## Every NPC only ever gets the events it has heard. The ones it has not are
## also sent to the server -- never into the prompt, only so a reply that
## mentions them can be flagged as a leak (knowledge that should not be there).
##
## Saved as plain JSON beside the player memory:
##   %APPDATA%/Godot/app_userdata/<project>/world_events.json
## Start the game with `-- --forget` to clear it along with the memories.

const PATH := "user://world_events.json"

var _events: Array = []   # [{id, what, where, at, reported_to, heard_by: {npc: {from, at, told_player}}}]
var _next_id := 1


func _init() -> void:
	if "--forget" in OS.get_cmdline_user_args():
		save()
		return
	_load()


## Records a report made to `npc_name`. The same incident reported twice is one
## event, now known to both NPCs. Returns the event id.
func add_report(npc_name: String, what: String, where: String) -> String:
	var key := what.strip_edges().to_lower() + "|" + where.strip_edges().to_lower()
	for e in _events:
		if (str(e["what"]).to_lower() + "|" + str(e["where"]).to_lower()) == key:
			_hear(e, npc_name, "player", true)
			save()
			return e["id"]
	var event := {
		"id": "e%d" % _next_id,
		"what": what.strip_edges(),
		"where": where.strip_edges(),
		"at": Time.get_unix_time_from_system(),
		"reported_to": npc_name,
		"heard_by": {},
	}
	_next_id += 1
	_hear(event, npc_name, "player", true)
	_events.append(event)
	save()
	return event["id"]


## Events this NPC has heard, oldest first, in the shape the server expects.
## "fresh": the NPC has not yet passed it on to the player.
func known_by(npc_name: String) -> Array:
	var out: Array = []
	for e in _events:
		if e["heard_by"].has(npc_name):
			var h: Dictionary = e["heard_by"][npc_name]
			out.append({"id": e["id"], "what": e["what"], "where": e["where"],
				"heard_from": h["from"], "fresh": not h["told_player"]})
	return out


## Events this NPC has NOT heard -- sent only so the server can flag leaks.
func unheard_by(npc_name: String) -> Array:
	var out: Array = []
	for e in _events:
		if not e["heard_by"].has(npc_name):
			out.append({"id": e["id"], "what": e["what"], "where": e["where"]})
	return out


## The NPC passed this news on to the player; it is no longer fresh for them.
func mark_shared(npc_name: String, event_id: String) -> void:
	for e in _events:
		if e["id"] == event_id and e["heard_by"].has(npc_name):
			e["heard_by"][npc_name]["told_player"] = true
			save()
			return


## One step of gossip among `npc_names`, restricted by `npc_types`: a random NPC
## who knows some event tells a random NPC of the same archetype who does not.
## Returns {id, from, to}, or {} when nobody of that type can hear it.
func spread(npc_names: Array, npc_types: Dictionary) -> Dictionary:
	var options: Array = []
	for e in _events:
		var type_knowers: Dictionary = {}
		var type_listeners: Dictionary = {}
		for n in npc_names:
			if e["heard_by"].has(n):
				var known_type: String = str(npc_types.get(n, ""))
				if not type_knowers.has(known_type):
					type_knowers[known_type] = []
				type_knowers[known_type].append(n)
			else:
				var listener_type: String = str(npc_types.get(n, ""))
				if not type_listeners.has(listener_type):
					type_listeners[listener_type] = []
					type_listeners[listener_type].append(n)
		for archetype in type_knowers:
			if type_listeners.has(archetype):
				options.append([e, type_knowers[archetype], type_listeners[archetype]])
	if options.is_empty():
		return {}
	var pick: Array = options[randi() % options.size()]
	var teller: String = pick[1][randi() % pick[1].size()]
	var listener: String = pick[2][randi() % pick[2].size()]
	_hear(pick[0], listener, teller, false)
	save()
	return {"id": pick[0]["id"], "from": teller, "to": listener}


func count() -> int:
	return _events.size()


func forget_all() -> void:
	_events.clear()
	_next_id = 1
	save()


func _hear(event: Dictionary, npc_name: String, source: String, told_player: bool) -> void:
	if event["heard_by"].has(npc_name):
		return
	event["heard_by"][npc_name] = {"from": source, "at": Time.get_unix_time_from_system(),
		"told_player": told_player}


func save() -> void:
	var file := FileAccess.open(PATH, FileAccess.WRITE)
	if file == null:
		push_warning("WorldEventStore: cannot write %s (%s)"
			% [PATH, error_string(FileAccess.get_open_error())])
		return
	file.store_string(JSON.stringify({"next_id": _next_id, "events": _events}, "  "))


func _load() -> void:
	if not FileAccess.file_exists(PATH):
		return
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(PATH))
	if parsed is Dictionary and parsed.get("events") is Array:
		_events = parsed["events"]
		_next_id = int(parsed.get("next_id", _events.size() + 1))
	else:
		push_warning("WorldEventStore: %s is not a valid event log; ignoring it." % PATH)
