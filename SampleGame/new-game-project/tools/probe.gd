extends Node3D

# Dev-only check: walks the physics space and reports whether the lunch hall
# tables/benches and the roof deck actually have solid bodies.
# Run with:  godot --headless --path . tools/Probe.tscn

const HALL_ORIGIN := Vector3(0, 3, 24)   # LunchHall instance inside Floor2 (1F at y=3)


func _ready() -> void:
	await get_tree().physics_frame
	await get_tree().physics_frame
	var space := get_viewport().find_world_3d().direct_space_state

	var tables := []
	var z := 4
	while z < 46:
		for x in [5, 12, 19]:
			tables.append(Vector3(x, 0, z))
		z += 6

	var miss_table := 0
	var miss_bench := 0
	for t in tables:
		var p: Vector3 = HALL_ORIGIN + t
		if not _hits(space, p + Vector3(0, 1.2, 0), p + Vector3(0, 0.2, 0)):
			miss_table += 1
			print("no table collision at ", p)
		for dz in [-1.1, 1.1]:
			var b: Vector3 = p + Vector3(0, 0, dz)
			if not _hits(space, b + Vector3(0, 1.0, 0), b + Vector3(0, 0.15, 0)):
				miss_bench += 1
	print("tables=%d missing=%d | benches=%d missing=%d" % [tables.size(), miss_table, tables.size() * 2, miss_bench])

	# Roof deck over the lunch hall bay (2F at y=6, deck top at y=9.4).
	for p in [Vector3(12, 0, 40), Vector3(12, 0, 60), Vector3(56, 0, 8), Vector3(100, 0, 96)]:
		var hit := _hits(space, p + Vector3(0, 20, 0), p + Vector3(0, 9.2, 0))
		print("roof above (%.0f, %.0f): %s" % [p.x, p.z, "solid" if hit else "OPEN"])

	get_tree().quit()


func _hits(space: PhysicsDirectSpaceState3D, from: Vector3, to: Vector3) -> bool:
	var q := PhysicsRayQueryParameters3D.create(from, to)
	return not space.intersect_ray(q).is_empty()
