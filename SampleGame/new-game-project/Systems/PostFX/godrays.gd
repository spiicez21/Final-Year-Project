extends MeshInstance3D
## Feeds the god-ray shader the sun's position on screen.
##
## The sun is a directional light with no position, so we place a virtual one a
## long way up its beam and project that. Rays fade out as it approaches the
## edge of the frame and switch off entirely once it is behind the camera.

@export var sun_path: NodePath
## How far up the beam the virtual sun sits. Only needs to clear the scene.
@export var sun_distance: float = 4000.0
@export var fade_speed: float = 6.0
## Rays are strongest looking straight into the sun and fade as it moves off.
@export_range(0.0, 1.0) var edge_fade: float = 0.55

var _sun: DirectionalLight3D
var _mat: ShaderMaterial
var _visibility: float = 0.0


func _ready() -> void:
	_mat = material_override as ShaderMaterial
	_sun = get_node_or_null(sun_path) as DirectionalLight3D
	if _sun == null:
		# fall back to the first directional light in the scene
		for node in get_tree().get_nodes_in_group("sun"):
			_sun = node as DirectionalLight3D
			break
	set_process(_mat != null)


func _process(delta: float) -> void:
	var cam := get_viewport().get_camera_3d()
	if cam == null or _sun == null:
		return

	# A DirectionalLight3D shines down its local -Z, so the sun sits back up
	# that axis from the viewer.
	var beam := -_sun.global_transform.basis.z
	var sun_world := cam.global_position - beam * sun_distance

	var target := 0.0
	if not cam.is_position_behind(sun_world):
		var size := Vector2(get_viewport().get_visible_rect().size)
		var screen := cam.unproject_position(sun_world)
		var uv := screen / size
		_mat.set_shader_parameter("sun_pos", uv)

		# strongest when looking into the sun, gone when it is well off frame
		var facing: float = -cam.global_transform.basis.z.dot(beam)
		target = clampf(inverse_lerp(edge_fade, 1.0, facing), 0.0, 1.0)
		# and damp it when the sun is near the horizon or below it
		target *= clampf(inverse_lerp(-0.05, 0.25, beam.y), 0.0, 1.0)

	_visibility = lerpf(_visibility, target, 1.0 - exp(-fade_speed * delta))
	_mat.set_shader_parameter("visibility", _visibility)
	visible = _visibility > 0.002
