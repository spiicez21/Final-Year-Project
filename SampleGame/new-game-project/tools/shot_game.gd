extends Node3D

# Dev-only helper: runs Main and shoots through the PLAYER's own camera, so the
# HD-2D orthogonal rig is what ends up in the PNGs.
# Run with:  godot --path . tools/ShotGame.tscn

const SPOTS := [
	{"name": "hd2d_seminar", "at": Vector3(12, 0.2, 50), "yaw": 0.0},
	{"name": "hd2d_seminar_back", "at": Vector3(12, 0.2, 44), "yaw": 0.0, "size": 30.0},
	{"name": "hd2d_topfloor", "at": Vector3(30, 6.4, 8), "yaw": 0.0},
	{"name": "hd2d_topfloor_wide", "at": Vector3(30, 6.4, 8), "yaw": 0.0, "size": 26.0},
]

var out_dir := "res://tools/shots"


func _ready() -> void:
	var player: Node3D = get_node("Main/Player")
	var rig: Node3D = player.get_node("CameraPivot")
	var cycle := get_node_or_null("Main/DayNightCycle")
	if cycle != null:
		cycle.set("time_of_day", 13.0)
	var dir := ProjectSettings.globalize_path(out_dir)
	DirAccess.make_dir_recursive_absolute(dir)

	for spot in SPOTS:
		player.global_position = spot["at"]
		player.rotation.y = spot["yaw"]
		rig.set("ortho_size", spot.get("size", 10.0))
		rig.set("_yaw_wanted", spot["yaw"])
		for i in 60:
			await get_tree().process_frame
		var culler := get_node("Main/FloorCuller")
		print("  %s storey=%d floors=[%s,%s,%s]" % [spot["name"], culler.call("current_storey"),
			get_node("Main/Floor1").visible, get_node("Main/Floor2").visible,
			get_node("Main/Floor3").visible])
		var cam: Camera3D = get_viewport().get_camera_3d()
		print("  player=%s cam=%s pitchdeg=%.1f arm=%.2f ortho=%s size=%.1f" % [
			player.global_position, cam.global_position,
			rad_to_deg(cam.global_rotation.x), rig.get_node("Yaw/Pitch/SpringArm3D").get_hit_length(),
			cam.projection == Camera3D.PROJECTION_ORTHOGONAL, cam.size])
		var img: Image = get_viewport().get_texture().get_image()
		img.save_png("%s/%s.png" % [dir, spot["name"]])
		print("wrote ", spot["name"])
	get_tree().quit()
