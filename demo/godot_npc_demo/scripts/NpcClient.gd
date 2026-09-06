extends Node
class_name NpcClient

## HTTP client for backend/gguf_server.py.
##
## One HTTPRequest node is spawned per call and freed after, rather than
## reusing a single node. Godot's HTTPRequest refuses a second request while
## one is in flight, and the demo can legitimately have a /chat pending while
## the player walks into a different NPC's radius and triggers a /health
## refresh — sharing one node would drop whichever call arrived second.

## Emitted for every state change worth showing in the HUD, so Main.gd never
## has to poll this node.
signal connection_changed(online: bool, detail: String)

const DEFAULT_BASE_URL := "http://127.0.0.1:8000"

## Generation is ~500-950ms warm, but a cold archetype also pays a ~1.2s
## model load, and the very first request additionally waits on the server
## building its scoring references. 60s is loose enough that a slow first
## call reads as "slow", not as a failure.
const REQUEST_TIMEOUT := 60.0
const HEALTH_TIMEOUT := 5.0

var base_url: String = DEFAULT_BASE_URL
var _online := false


func _make_request(timeout: float) -> HTTPRequest:
	var http := HTTPRequest.new()
	http.timeout = timeout
	add_child(http)
	return http


func _set_online(online: bool, detail: String) -> void:
	# Only emit on an actual transition — the HUD would otherwise flicker on
	# every poll.
	if online != _online:
		_online = online
		connection_changed.emit(online, detail)


func is_online() -> bool:
	return _online


## Returns the archetype list the server can actually serve, or [] on failure.
## The server filters this to archetypes whose GGUF is present on disk, so it
## is safe to spawn one NPC per returned entry.
func fetch_archetypes() -> Array:
	var http := _make_request(HEALTH_TIMEOUT)
	var err := http.request(base_url + "/archetypes")
	if err != OK:
		http.queue_free()
		_set_online(false, "could not reach %s" % base_url)
		return []

	var result: Array = await http.request_completed
	http.queue_free()

	var parsed := _parse_json_response(result)
	if parsed.is_empty():
		_set_online(false, "no response from %s" % base_url)
		return []

	_set_online(true, "connected to %s" % base_url)
	return parsed.get("available", [])


## Sends one player line to one NPC archetype.
##
## Always resolves to a Dictionary. On success it is the server's ChatResponse;
## on failure it is {"error": <human readable>}, which the caller shows in the
## dialogue box rather than crashing the demo.
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
		return {"error": "Could not send the request. Is the server running?"}

	var result: Array = await http.request_completed
	http.queue_free()

	var http_result: int = result[0]
	var status_code: int = result[1]
	var body: PackedByteArray = result[3]

	if http_result != HTTPRequest.RESULT_SUCCESS:
		_set_online(false, "transport error %d" % http_result)
		return {"error": "No reply from the model server (%s). Is it running?" % base_url}

	var text := body.get_string_from_utf8()
	var parsed = JSON.parse_string(text)

	if status_code != 200:
		# FastAPI puts the useful message in "detail" for both 400 and 503,
		# and 503 here specifically means the GGUF for that archetype is
		# missing — worth surfacing verbatim rather than flattening to
		# "server error".
		var detail := "HTTP %d" % status_code
		if parsed is Dictionary and parsed.has("detail"):
			detail = str(parsed["detail"])
		_set_online(true, "server returned %d" % status_code)
		return {"error": detail}

	if not (parsed is Dictionary):
		_set_online(true, "unparseable response")
		return {"error": "Server sent a malformed response."}

	_set_online(true, "connected to %s" % base_url)
	return parsed


func _parse_json_response(result: Array) -> Dictionary:
	if result[0] != HTTPRequest.RESULT_SUCCESS or result[1] != 200:
		return {}
	var parsed = JSON.parse_string(result[3].get_string_from_utf8())
	return parsed if parsed is Dictionary else {}
