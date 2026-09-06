extends Node3D

# Dev-only check: drops rays down the seminar hall centre aisle and the stage
# steps to confirm the raked floor has a continuous walking surface.
# Run with:  godot --headless --path . tools/ProbeHall.tscn

func _ready() -> void:
	# GFLevel instances the furniture scene on its own, so probe that in place:
	# the hall sits at (0, 0, 40) inside the level.
	var packed: PackedScene = load("res://Scene/GFLevel.tscn")
	var lvl: Node3D = packed.instantiate()
	lvl.position = Vector3(0, 0, -40)
	add_child(lvl)
	await get_tree().physics_frame
	await get_tree().physics_frame
	var space := get_viewport().find_world_3d().direct_space_state

	print("--- centre aisle, z = 1 .. 23 ---")
	var z := 1.0
	while z <= 23.0:
		_drop(space, 12.0, z)
		z += 1.0
	print("--- stage step flight at x = 5 ---")
	for sz in [15.0, 15.8, 16.4, 17.0, 17.6, 18.2, 20.0]:
		_drop(space, 5.0, sz)
	get_tree().quit()


func _drop(space: PhysicsDirectSpaceState3D, x: float, z: float) -> void:
	var q := PhysicsRayQueryParameters3D.create(
		Vector3(x, 3.0, z), Vector3(x, -3.0, z))
	var hit := space.intersect_ray(q)
	if hit.is_empty():
		print("  (%.1f, %5.1f)  NO FLOOR" % [x, z])
	else:
		print("  (%.1f, %5.1f)  y = %6.3f" % [x, z, hit.position.y])
