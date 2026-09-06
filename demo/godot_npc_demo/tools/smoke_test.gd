extends SceneTree

## Headless check that the Godot client and backend/gguf_server.py agree.
##
## Verifies the network path without launching the game window, which is the
## part most likely to break silently: a renamed response field shows up here
## as an explicit FAIL instead of as an empty HUD nobody notices during a demo.
##
## Usage (server must already be running):
##   E:\Godot\Godot_v4.7-stable_win64_console.exe --headless --path . \
##       --script res://tools/smoke_test.gd
##
## Exits non-zero on failure so it can gate a commit.

const REQUIRED_CHAT_FIELDS := [
	"response", "archetype", "adapter_switch_ms", "generation_ms",
	"drift_score", "kbd", "leaked_fact_ids",
]

var _failures := 0


func _initialize() -> void:
	_run.call_deferred()


func _check(condition: bool, description: String) -> void:
	if condition:
		print("  PASS  %s" % description)
	else:
		_failures += 1
		print("  FAIL  %s" % description)


func _run() -> void:
	var client := NpcClient.new()
	root.add_child(client)

	print("\n=== /archetypes ===")
	var archetypes: Array = await client.fetch_archetypes()
	_check(not archetypes.is_empty(), "server returned at least one archetype")
	if archetypes.is_empty():
		print("\nIs the server running?  python -m uvicorn backend.gguf_server:app --port 8000")
		_finish()
		return
	print("  archetypes: %s" % ", ".join(archetypes))
	_check(client.is_online(), "client reports online after a successful call")

	print("\n=== /chat ===")
	var archetype: String = archetypes[0]
	var reply: Dictionary = await client.chat(archetype, "Good evening. Everything alright here?")
	_check(not reply.has("error"), "chat returned a reply, not an error")
	if reply.has("error"):
		print("  error: %s" % reply["error"])
		_finish()
		return

	for field in REQUIRED_CHAT_FIELDS:
		_check(reply.has(field), "response contains '%s'" % field)
	_check(str(reply.get("response", "")).strip_edges() != "", "reply text is non-empty")
	_check(reply.get("archetype") == archetype, "reply echoes the requested archetype")

	print("\n  %s said: %s" % [archetype, reply.get("response")])
	print("  switch %.0fms · generation %.0fms · PDM v2 %s · KBD %s" % [
		reply.get("adapter_switch_ms", 0.0),
		reply.get("generation_ms", 0.0),
		reply.get("drift_score"),
		reply.get("kbd"),
	])

	print("\n=== unknown archetype is rejected cleanly ===")
	var bad: Dictionary = await client.chat("dragon", "Hello?")
	_check(bad.has("error"), "unknown archetype surfaces an error instead of crashing")

	_finish()


func _finish() -> void:
	if _failures == 0:
		print("\nALL CHECKS PASSED\n")
		quit(0)
	else:
		print("\n%d CHECK(S) FAILED\n" % _failures)
		quit(1)
