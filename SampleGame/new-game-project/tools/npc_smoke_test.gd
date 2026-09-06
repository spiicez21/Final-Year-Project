extends SceneTree

## Headless check that the game's NPC layer and backend/gguf_server.py agree.
##
## Verifies the network contract without launching the game, which is the part
## most likely to break silently: a renamed response field shows up here as an
## explicit FAIL instead of as an empty metrics panel nobody notices during a
## demo.
##
## Start the server first, from the repo root (two levels above this project):
##   .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000
##
## Then:
##   godot --headless --path . --script res://tools/npc_smoke_test.gd
##
## Exits non-zero on failure, so it can gate a commit.

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
		print("\n  Is the server running?")
		print("  .venv/Scripts/python.exe -m uvicorn backend.gguf_server:app --port 8000")
		_finish()
		return
	print("  serving: %s" % ", ".join(archetypes))

	# Every archetype the director wants to spawn must be one the server can
	# actually answer as, or that NPC silently never appears on the campus.
	print("\n=== spawn table vs server ===")
	for entry in NpcDirector.SPAWNS:
		_check(archetypes.has(entry["archetype"]),
			"'%s' (%s) is served" % [entry["archetype"], entry["name"]])

	print("\n=== /chat ===")
	var archetype: String = NpcDirector.SPAWNS[0]["archetype"]
	var reply: Dictionary = await client.chat(archetype, "Morning. Anything I should know today?")
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
		reply.get("adapter_switch_ms", 0.0), reply.get("generation_ms", 0.0),
		reply.get("drift_score"), reply.get("kbd"),
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
