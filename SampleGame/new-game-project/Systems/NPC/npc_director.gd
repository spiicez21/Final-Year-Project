extends Node3D
class_name NpcDirector

## Spawns the talkable NPCs and drives the conversation loop.
##
## Drop this node into Scene/Main.tscn and point `player_path` at the Player.
## Everything else is self-contained: it creates its own NpcClient and HUD.
##
## Replies come from the per-archetype LoRA models served by
## backend/gguf_server.py in the repo root. Start it first:
##   .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
## With the server down the NPCs still spawn and are still walkable — the
## dialogue box just reports the connection failure, which is far easier to
## diagnose than an empty campus.

@export var player_path: NodePath = ^"../Player"
@export var server_url: String = "http://127.0.0.1:8000"

## Ground-floor placements, all inside the corridor band.
##
## Room blocks on this floor occupy z=0..8 and z=16..24 (see
## tools/layout_plan.json), so z=8..16 is the circulation corridor and z=12 is
## its centre line. The entrance hall spans x=32..48 and opens onto the same
## band, so this row runs from the hall eastward past the classrooms towards
## the HOD suite — the route a player walks on entering.
##
## `y` is only a starting height: each NPC is dropped onto whatever floor is
## actually beneath it (see _snap_to_floor), so these need no vertical
## precision.
const SPAWNS := [
	{"archetype": "police officer", "name": "Officer Reyes", "role": "campus security",
		"pos": Vector3(40.0, 1.0, 12.0), "tint": Color(0.62, 0.72, 1.00)},
	{"archetype": "professor", "name": "Prof. Adeyemi", "role": "faculty",
		"pos": Vector3(47.0, 1.0, 12.0), "tint": Color(0.86, 0.72, 1.00)},
	{"archetype": "executive", "name": "Halvorsen", "role": "head of department",
		"pos": Vector3(53.0, 1.0, 12.0), "tint": Color(0.90, 0.90, 0.95)},
	{"archetype": "social worker", "name": "Ms. Okafor", "role": "student counsellor",
		"pos": Vector3(59.0, 1.0, 12.0), "tint": Color(0.70, 0.94, 1.00)},
	{"archetype": "shopkeeper", "name": "Nadia", "role": "campus store",
		"pos": Vector3(65.0, 1.0, 12.0), "tint": Color(1.00, 0.88, 0.55)},
]

var _client: NpcClient
var _hud: NpcDialogueHud
var _player: Node3D
var _camera_rig: Node

var _npcs: Array[NpcActor] = []
var _active: NpcActor = null
var _nearest: NpcActor = null
var _awaiting := false


func _ready() -> void:
	_player = get_node_or_null(player_path)
	if _player == null:
		push_error("NpcDirector: player_path '%s' does not resolve; NPCs will not be interactive."
			% player_path)
	else:
		# third_person_camera.gd grabs the mouse on click. While a dialogue is
		# open we hand the cursor back, so its input handler is muted rather
		# than left fighting the text box for the pointer.
		_camera_rig = _player.get_node_or_null(^"CameraPivot")

	_client = NpcClient.new()
	_client.base_url = _resolve_server_url()
	_client.connection_changed.connect(_on_connection_changed)
	add_child(_client)

	_hud = NpcDialogueHud.new()
	_hud.message_submitted.connect(_on_message_submitted)
	add_child(_hud)

	await _spawn_npcs()


## Lets an exported build point at a server on another machine without a
## rebuild:
##   CampusNPC.exe -- --server=http://192.168.0.9:8000
## The bare `--` is what separates engine arguments from game arguments.
func _resolve_server_url() -> String:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--server="):
			var url := arg.trim_prefix("--server=")
			print("NpcDirector: using server from command line: ", url)
			return url
	return server_url


func _spawn_npcs() -> void:
	var served: Array = await _client.fetch_archetypes()

	for entry in SPAWNS:
		# With the server up, only spawn NPCs it can actually answer as —
		# otherwise the campus has someone standing in it who errors the
		# moment you talk to them. With the server down we spawn everyone,
		# so the world still looks right.
		if not served.is_empty() and not served.has(entry["archetype"]):
			continue
		var npc := NpcActor.create(entry["archetype"], entry["name"], entry["role"],
			entry["tint"], entry["pos"])
		add_child(npc)
		_npcs.append(npc)
		_snap_to_floor(npc)

	if _npcs.is_empty():
		push_warning("NpcDirector: no NPCs spawned — server served none of the known archetypes.")


## Drops an NPC onto the floor under its spawn point so it never hovers or
## sinks if the corridor height changes.
func _snap_to_floor(npc: NpcActor) -> void:
	# Cast from just above the spawn point, not from high overhead: storeys
	# are 3m apart, so a ray starting 3m up begins ON the next floor and
	# snaps the NPC one level too high.
	var space := get_world_3d().direct_space_state
	var from := npc.global_position + Vector3.UP * 0.8
	var to := npc.global_position + Vector3.DOWN * 2.5
	var query := PhysicsRayQueryParameters3D.create(from, to)
	query.exclude = [npc.get_rid()]
	var hit := space.intersect_ray(query)
	if hit.has("position"):
		npc.global_position = hit["position"]


# --- interaction -----------------------------------------------------------

func _process(_delta: float) -> void:
	if _active != null or _player == null:
		return

	# Nearest in range wins, so standing between two NPCs is never ambiguous
	# about which one E will talk to.
	var best: NpcActor = null
	var best_distance := INF
	for npc in _npcs:
		var d := npc.global_position.distance_to(_player.global_position)
		if d <= NpcActor.INTERACT_RANGE and d < best_distance:
			best_distance = d
			best = npc

	if best != _nearest:
		if _nearest != null:
			_nearest.set_prompt_visible(false)
		if best != null:
			best.set_prompt_visible(true)
		_nearest = best


func _unhandled_input(event: InputEvent) -> void:
	# LineEdit marks typed keys as handled, so this never fires mid-sentence,
	# which is why "e" can be typed into the chat box without reopening a
	# conversation.
	if event.is_action_pressed(&"interact") and _active == null and _nearest != null:
		_open(_nearest)
		get_viewport().set_input_as_handled()
	elif event.is_action_pressed(&"ui_cancel") and _active != null:
		_close()
		get_viewport().set_input_as_handled()


func _open(npc: NpcActor) -> void:
	_active = npc
	npc.set_prompt_visible(false)
	_hud.open_with(npc.display_name, npc.archetype, npc.tint)
	_set_player_active(false)


func _close() -> void:
	# Deliberately allowed mid-request: the pending chat() call still resolves
	# and _deliver() drops the result because _active is null. Blocking Esc
	# until a slow reply lands would feel like a hang.
	_active = null
	_awaiting = false
	_hud.set_dialogue_visible(false)
	_set_player_active(true)


func _set_player_active(active: bool) -> void:
	if _player:
		if not active:
			_player.velocity = Vector3.ZERO
		_player.set_physics_process(active)
	if _camera_rig:
		_camera_rig.set_process_unhandled_input(active)
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if active else Input.MOUSE_MODE_VISIBLE


func _on_message_submitted(text: String) -> void:
	if _awaiting or _active == null:
		return
	_awaiting = true
	_hud.show_pending(text)

	var npc := _active
	var reply: Dictionary = await _client.chat(npc.archetype, text)
	_deliver(npc, text, reply)


func _deliver(npc: NpcActor, message: String, reply: Dictionary) -> void:
	_awaiting = false

	# The player may have walked away, or opened a different NPC, while this
	# was in flight. Showing the reply now would put one NPC's line in
	# another's mouth, so it is dropped instead.
	if _active == null or _active != npc:
		return

	if reply.has("error"):
		_hud.show_error(message, str(reply["error"]))
		return

	_hud.show_reply(message, str(reply.get("response", "")))
	_hud.update_metrics(reply)


func _on_connection_changed(online: bool, detail: String) -> void:
	_hud.set_status(online, detail)
