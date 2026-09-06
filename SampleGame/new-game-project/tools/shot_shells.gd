extends Node3D

# Dev-only helper: eye-level looks at the freshly cut room shells.
# Run with:  godot --path . tools/ShotShells.tscn

const ROOMS := [
	{"scene": "res://Scene/Rooms/Classroom_12x8.tscn", "origin": Vector3(0, 0, 0),
	 "views": [{"name": "shell_class", "pos": Vector3(11.0, 1.8, 6.6), "look": Vector3(1.0, 1.2, 3.6)}]},
	{"scene": "res://Scene/Rooms/Lab_16x12.tscn", "origin": Vector3(60, 0, 0),
	 "views": [{"name": "shell_lab", "pos": Vector3(62.0, 1.8, 2.0), "look": Vector3(69.0, 1.3, 11.0)}]},
	{"scene": "res://Scene/Rooms/Faculty_12x8.tscn", "origin": Vector3(120, 0, 0),
	 "views": [{"name": "shell_faculty", "pos": Vector3(131.0, 1.8, 6.8), "look": Vector3(121.0, 1.2, 2.6)}]},
	{"scene": "res://Scene/Rooms/Faculty_8x8.tscn", "origin": Vector3(180, 0, 0),
	 "views": [{"name": "shell_hod", "pos": Vector3(187.0, 1.8, 6.8), "look": Vector3(181.5, 1.2, 2.4)}]},
	{"scene": "res://Scene/Rooms/Toilet_12x8.tscn", "origin": Vector3(240, 0, 0),
	 "views": [{"name": "shell_toilet", "pos": Vector3(251.0, 1.8, 6.6), "look": Vector3(243.0, 1.2, 1.2)}]},
	{"scene": "res://Scene/Rooms/SeminarHall_24x24.tscn", "origin": Vector3(300, 0, 0),
	 "views": [{"name": "shell_seminar", "pos": Vector3(312.0, 1.9, 1.5), "look": Vector3(312.0, 1.2, 20.0)}]},
]

var out_dir := "res://tools/shots"


func _ready() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.12, 0.13, 0.15)
	env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	env.ambient_light_color = Color(0.7, 0.73, 0.78)
	env.ambient_light_energy = 0.8
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	for room in ROOMS:
		var inst: Node3D = (load(room["scene"]) as PackedScene).instantiate()
		inst.position = room["origin"]
		add_child(inst)

	var cam := Camera3D.new()
	cam.fov = 70.0
	cam.far = 500.0
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
			get_viewport().get_texture().get_image().save_png(
				"%s/%s.png" % [dir, view["name"]])
			print("wrote ", view["name"])
	get_tree().quit()
