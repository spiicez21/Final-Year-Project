# Headless sanity pass over the generated rooms.
#   godot --headless --path . --script res://tools/verify_scenes.gd
# Loads each scene, counts nodes and colliders, and prints the world-space size
# of a few kit props so the root_scale=2.0 import setting can be eyeballed.
extends SceneTree

const SCENES := [
	"res://Scene/Props/LabFurniture_16x16.tscn",
	"res://Scene/Rooms/Lab_16x16.tscn",
	"res://Scene/Props/SeminarFurniture_24x24.tscn",
	"res://Scene/Rooms/SeminarHall_24x24.tscn",
	"res://Scene/Rooms/LunchHall_24x48.tscn",
	"res://Scene/Rooms/Lab_20x16.tscn",
	"res://Scene/Rooms/Lab_24x16.tscn",
	"res://Scene/Rooms/Faculty_12x16.tscn",
	"res://Scene/Rooms/Faculty_16x8.tscn",
	"res://Scene/Rooms/Faculty_20x8.tscn",
	"res://Scene/Rooms/Faculty_24x8.tscn",
	"res://Scene/Rooms/Faculty_24x16.tscn",
	"res://Scene/GFLevel.tscn",
	"res://Scene/1FLevel.tscn",
	"res://Scene/2FLevel.tscn",
]

const PROBES := [
	["res://Assets/lab_bench_long.glb", "2.40 x 0.87 x 0.80"],
	["res://Assets/lab_chair.glb", "0.50 x 0.86 x 0.49"],
	["res://Assets/seminar_seat_row4.glb", "2.36 x 0.93 x 0.56"],
	["res://Assets/canteen_table_4.glb", "0.90 x 0.75 x 0.90"],
	["res://Assets/serving_counter.glb", "2.48 x 1.61 x 0.90"],
]


func _count(node: Node, acc: Dictionary) -> void:
	acc["nodes"] += 1
	if node is CollisionShape3D:
		acc["colliders"] += 1
		if node.shape == null:
			acc["bad_shapes"] += 1
	if node is MeshInstance3D:
		acc["meshes"] += 1
	for c in node.get_children():
		_count(c, acc)


func _aabb(node: Node, box: AABB, first: Array) -> AABB:
	# NOTE: the glTF root is not reported as inside the tree yet, so it logs one
	# harmless get_global_transform error per probe. The child meshes measure fine.
	if node is MeshInstance3D:
		var m: AABB = node.get_aabb()
		var t: Transform3D = node.global_transform
		var world: AABB = t * m
		if first[0]:
			box = world
			first[0] = false
		else:
			box = box.merge(world)
	for c in node.get_children():
		box = _aabb(c, box, first)
	return box


func _initialize() -> void:
	var failed := 0

	print("--- scenes ---")
	for path in SCENES:
		if not ResourceLoader.exists(path):
			print("MISSING  ", path)
			failed += 1
			continue
		var packed: PackedScene = load(path)
		if packed == null:
			print("LOADFAIL ", path)
			failed += 1
			continue
		var root: Node = packed.instantiate()
		root_ready(root)
		var acc := {"nodes": 0, "colliders": 0, "meshes": 0, "bad_shapes": 0}
		_count(root, acc)
		print("%-46s nodes=%-5d meshes=%-5d colliders=%-4d empty_shapes=%d"
			% [path.get_file(), acc["nodes"], acc["meshes"], acc["colliders"], acc["bad_shapes"]])
		if acc["bad_shapes"] > 0:
			failed += 1
		root.free()

	print("")
	print("--- prop world size (want ~= expected, proves root_scale=2.0) ---")
	for probe in PROBES:
		var path: String = probe[0]
		if not ResourceLoader.exists(path):
			print("MISSING  ", path)
			failed += 1
			continue
		var inst: Node = (load(path) as PackedScene).instantiate()
		root_ready(inst)
		var box := _aabb(inst, AABB(), [true])
		print("%-34s %5.2f x %5.2f x %5.2f   expected %s"
			% [path.get_file(), box.size.x, box.size.y, box.size.z, probe[1]])
		inst.free()

	print("")
	print("FAILURES: ", failed)
	quit(1 if failed > 0 else 0)


func root_ready(node: Node) -> void:
	# Parent into the tree so global_transform is valid for the AABB walk.
	get_root().add_child(node)
