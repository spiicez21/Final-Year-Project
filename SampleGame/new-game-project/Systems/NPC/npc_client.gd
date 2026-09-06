extends Node
class_name NpcClient

## HTTP client for the project's inference server (backend/gguf_server.py in
## the repo root, two levels above this Godot project).
##
## One HTTPRequest node is spawned per call and freed after, rather than
## reusing a single node: Godot's HTTPRequest refuses a second request while
## one is in flight, and the game can legitimately have a /chat pending while
## something else refreshes the archetype list.

signal connection_changed(online: bool, detail: String)

const DEFAULT_BASE_URL := "http://127.0.0.1:8000"

## Generation is ~500-950ms warm, but a cold archetype also pays a ~1.2s model
## load, and the very first request additionally waits on the server building
## its scoring references. 60s is loose enough that a slow first call reads as
## "slow" rather than as a failure.
const REQUEST_TIMEOUT := 60.0
const DISCOVERY_TIMEOUT := 5.0

var base_url: String = DEFAULT_BASE_URL
var _online := false


func _make_request(timeout: float) -> HTTPRequest:
	var http := HTTPRequest.new()
	http.timeout = timeout
	add_child(http)
	return http


func _set_online(online: bool, detail: String) -> void:
	# Only emit on an actual transition, or the HUD flickers on every poll.
	if online != _online:
		_online = online
		connection_changed.emit(online, detail)


func is_online() -> bool:
	return _online


## Archetypes the server can actually serve, or [] on failure. The server
## filters this to archetypes whose GGUF is present on disk, so it is safe to
## spawn one NPC per returned entry.
func fetch_archetypes() -> Array:
	var http := _make_request(DISCOVERY_TIMEOUT)
	var err := http.request(base_url + "/archetypes")
	if err != OK:
		http.queue_free()
		_set_online(false, "cannot reach %s" % base_url)
		return []

	var result: Array = await http.request_completed
	http.queue_free()

	if result[0] != HTTPRequest.RESULT_SUCCESS or result[1] != 200:
		_set_online(false, "no response from %s" % base_url)
		return []

	var parsed = JSON.parse_string(result[3].get_string_from_utf8())
	if not (parsed is Dictionary):
		_set_online(false, "malformed archetype list")
		return []

	_set_online(true, base_url)
	return parsed.get("available", [])


## Sends one player line to one NPC archetype.
##
## Always resolves to a Dictionary. On success it is the server's ChatResponse;
## on failure it is {"error": <human readable>}, which the caller shows in the
## dialogue box rather than crashing the game.
func chat(archetype: String, message: String) -> Dictionary:
	var http := _make_request(REQUEST_TIMEOUT)
	var payload := JSON.stringify({"archetype": archetype, "message": message})
	var err := http.request(
		base_url + "/chat",
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		payload
	)
	if err != OK:
		http.queue_free()
		_set_online(false, "request failed to start")
		return {"error": "Could not send the request. Is the model server running?"}

	var result: Array = await http.request_completed
	http.queue_free()

	if result[0] != HTTPRequest.RESULT_SUCCESS:
		_set_online(false, "transport error %d" % result[0])
		return {"error": "No reply from the model server (%s). Is it running?" % base_url}

	var parsed = JSON.parse_string(result[3].get_string_from_utf8())

	if result[1] != 200:
		# FastAPI puts the useful message in "detail"; a 503 here specifically
		# means that archetype's GGUF is missing, which is worth showing
		# verbatim rather than flattening to "server error".
		var detail := "HTTP %d" % result[1]
		if parsed is Dictionary and parsed.has("detail"):
			detail = str(parsed["detail"])
		_set_online(true, base_url)
		return {"error": detail}

	if not (parsed is Dictionary):
		_set_online(true, base_url)
		return {"error": "Server sent a malformed response."}

	_set_online(true, base_url)
	return parsed
