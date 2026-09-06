extends Node3D

# Dev-only helper: shots of the CSE/IT department conversion - a faculty room,
# the wide lab variants, and the seminar hall in place on the ground floor.
# Run with:  godot --path . tools/ShotDept.tscn

const ROOMS := [
	{
		"scene": "res://Scene/Rooms/Faculty_24x16.tscn",
		"origin": Vector3(0, 0, 0),
		"views": [
			{"name": "faculty_desks", "pos": Vector3(12.0, 1.8, 14.0), "look": Vector3(11.0, 1.2, 4.0)},
			{"name": "faculty_lounge", "pos": Vector3(20.0, 1.8, 8.0), "look": Vector3(9.0, 1.0, 14.0)},
			{"name": "faculty_aerial", "pos": Vector3(12.0, 2.85, 15.4), "look": Vector3(12.0, 0.9, 2.0)},
			{"name": "faculty_senior", "pos": Vector3(15.0, 1.7, 9.5), "look": Vector3(20.5, 1.1, 3.5)},
		],
	},
	{
		"scene": "res://Scene/Rooms/Faculty_20x8.tscn",
		"origin": Vector3(60, 0, 0),
		"views": [
			{"name": "faculty_small", "pos": Vector3(78.0, 1.8, 6.5), "look": Vector3(63.0, 1.2, 3.0)},
			{"name": "faculty_small_pods", "pos": Vector3(61.5, 1.7, 6.8), "look": Vector3(72.0, 1.1, 2.6)},
		],
	},
	{
		"scene": "res://Scene/Rooms/Lab_24x16.tscn",
		"origin": Vector3(0, 0, 60),
		"views": [
			{"name": "lab24_board", "pos": Vector3(12.0, 1.7, 62.0), "look": Vector3(13.5, 1.5, 75.5)},
			{"name": "lab24_aerial", "pos": Vector3(12.0, 22.0, 50.0), "look": Vector3(12.0, 0.0, 69.0)},
		],
	},
	{
		"scene": "res://Scene/GFLevel.tscn",
		"origin": Vector3(0, 0, 200),
		"views": [
			{"name": "gf_seminar_house", "pos": Vector3(12.0, 2.0, 243.0), "look": Vector3(12.0, 1.6, 270.0)},
			{"name": "gf_seminar_up", "pos": Vector3(12.0, 1.5, 250.0), "look": Vector3(12.0, 5.0, 250.5)},
			{"name": "gf_seminar_stage", "pos": Vector3(12.0, 1.8, 264.0), "look": Vector3(10.0, 1.4, 271.0)},
			{"name": "gf_seminar_aerial", "pos": Vector3(12.0, 30.0, 228.0), "look": Vector3(12.0, 0.0, 260.0)},
			{"name": "gf_corridor", "pos": Vector3(30.0, 1.7, 222.0), "look": Vector3(80.0, 1.5, 222.0)},
			{"name": "gf_connector", "pos": Vector3(68.0, 1.7, 228.0), "look": Vector3(68.0, 1.5, 272.0)},
			{"name": "gf_connector_up", "pos": Vector3(68.0, 1.7, 228.0), "look": Vector3(68.0, 6.0, 240.0)},
			{"name": "gf_corridor_up", "pos": Vector3(30.0, 1.7, 222.0), "look": Vector3(50.0, 6.0, 222.0)},
		],
	},
]

var out_dir := "res://tools/shots"


func _ready() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.12, 0.13, 0.15)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.62, 0.65, 0.7)
	env.ambient_light_energy = 0.55
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	for room in ROOMS:
		var inst: Node3D = (load(room["scene"]) as PackedScene).instantiate()
		inst.position = room["origin"]
		add_child(inst)

	var cam := Camera3D.new()
	cam.fov = 68.0
	cam.far = 800.0
	add_child(cam)
	cam.make_current()

	var dir := ProjectSettings.globalize_path(out_dir)
	DirAccess.make_dir_recursive_absolute(dir)
	for room in ROOMS:
		for view in room["views"]:
			cam.global_position = view["pos"]
			cam.look_at(view["look"], Vector3.UP)
			for i in 20:
				await get_tree().process_frame
			var img: Image = get_viewport().get_texture().get_image()
			img.save_png("%s/%s.png" % [dir, view["name"]])
			print("wrote ", view["name"])
	get_tree().quit()
