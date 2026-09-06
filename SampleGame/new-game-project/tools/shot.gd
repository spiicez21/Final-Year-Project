extends Node3D

# Dev-only helper: loads Main, flies a camera to a few fixed viewpoints and
# writes PNGs next to the project so the levels can be eyeballed after a build.
# Run with:  godot --path . tools/Shot.tscn

const VIEWS := [
	{"name": "entrance", "pos": Vector3(52, 14, -46), "look": Vector3(52, 3, 8)},
	{"name": "aerial", "pos": Vector3(56, 120, -60), "look": Vector3(56, 0, 52)},
	{"name": "roof", "pos": Vector3(-30, 34, -20), "look": Vector3(56, 9, 52)},
	{"name": "lunchhall", "pos": Vector3(12, 5.5, 26), "look": Vector3(12, 3.5, 60)},
	{"name": "lobby", "pos": Vector3(52, 1.7, -3), "look": Vector3(52, 1.6, 20)},
	{"name": "lobby_back", "pos": Vector3(66, 1.7, 14), "look": Vector3(40, 1.6, 4)},
]

var out_dir := "res://tools/shots"


func _ready() -> void:
	var cam := Camera3D.new()
	cam.fov = 70.0
	cam.far = 800.0
	add_child(cam)
	cam.make_current()
	DirAccess.make_dir_recursive_absolute(ProjectSettings.globalize_path(out_dir))
	for view in VIEWS:
		cam.global_position = view["pos"]
		cam.look_at(view["look"], Vector3.UP)
		for i in 30:
			await get_tree().process_frame
		var img := get_viewport().get_texture().get_image()
		img.save_png("%s/%s.png" % [ProjectSettings.globalize_path(out_dir), view["name"]])
	get_tree().quit()
