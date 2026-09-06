extends Node3D

# Dev-only helper: instances the individual room scenes (not the whole school)
# and writes eye-level PNGs so the new lab / seminar / cafeteria dressing can be
# checked without opening the editor.
# Run with:  godot --path . tools/ShotRooms.tscn

const ROOMS := [
	{
		"scene": "res://Scene/Rooms/Lab_16x16.tscn",
		"origin": Vector3(0, 0, 0),
		"views": [
			{"name": "lab_entry", "pos": Vector3(2.0, 1.7, 14.0), "look": Vector3(9.0, 1.2, 6.0)},
			{"name": "lab_board", "pos": Vector3(8.0, 1.7, 3.0), "look": Vector3(9.5, 1.5, 15.5)},
			{"name": "lab_aerial", "pos": Vector3(8.0, 16.0, -6.0), "look": Vector3(8.0, 0.0, 9.0)},
		],
	},
	{
		"scene": "res://Scene/Rooms/SeminarHall_24x24.tscn",
		"origin": Vector3(100, 0, 0),
		"views": [
			{"name": "seminar_house", "pos": Vector3(112.0, 2.0, 2.5), "look": Vector3(112.0, 1.6, 30.0)},
			{"name": "seminar_stage", "pos": Vector3(112.0, 1.8, 24.0), "look": Vector3(110.0, 1.4, 31.0)},
			{"name": "seminar_aerial", "pos": Vector3(112.0, 2.9, 0.6), "look": Vector3(112.0, 0.7, 26.0)},
			{"name": "seminar_dais", "pos": Vector3(110.0, 1.4, 22.5), "look": Vector3(116.0, 1.0, 28.5)},
			{"name": "seminar_rake", "pos": Vector3(100.9, 1.5, 21.5), "look": Vector3(101.6, 0.2, 6.0)},
			{"name": "seminar_steps", "pos": Vector3(112.0, 0.6, 20.0), "look": Vector3(105.0, 0.2, 24.5)},
			{"name": "seminar_rear", "pos": Vector3(112.0, 1.8, 12.0), "look": Vector3(112.5, 1.4, 0.5)},
		],
	},
	{
		"scene": "res://Scene/Rooms/LunchHall_24x48.tscn",
		"origin": Vector3(200, 0, 0),
		"views": [
			{"name": "cafe_dining", "pos": Vector3(212.0, 2.0, 6.0), "look": Vector3(212.0, 1.4, 44.0)},
			{"name": "cafe_counter", "pos": Vector3(207.0, 1.7, 40.0), "look": Vector3(210.0, 1.2, 47.0)},
			{"name": "cafe_aerial", "pos": Vector3(212.0, 30.0, -8.0), "look": Vector3(212.0, 0.0, 26.0)},
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
		var packed: PackedScene = load(room["scene"])
		var inst: Node3D = packed.instantiate()
		inst.position = room["origin"]
		add_child(inst)

	var cam := Camera3D.new()
	cam.fov = 68.0
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
			var img: Image = get_viewport().get_texture().get_image()
			img.save_png("%s/%s.png" % [dir, view["name"]])
			print("wrote ", view["name"])
	get_tree().quit()
