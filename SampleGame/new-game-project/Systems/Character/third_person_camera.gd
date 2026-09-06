extends Node3D
## GTA-style third person camera rig.
##
## Sits on a pivot that is top_level, so it TRAILS the character rather than
## being welded to it. The pieces that give this its feel:
##
##   * vertical follow is much softer than horizontal, so stairs, slopes and
##     jumps do not jolt the view the way a rigid parent would
##   * the rig drifts back behind the character once the mouse goes idle
##   * distance and FOV open up with speed, so sprinting feels faster
##   * the pivot leads slightly in the direction of travel
##   * a SpringArm keeps the camera out of walls
##
## The owner feeds motion in through set_motion(); nothing here reads the
## character directly, so the same rig works for any actor.

@export_group("Target")
@export var target_path: NodePath
@export var pivot_height: float = 1.5
## How far ahead of the character the pivot drifts at full speed.
@export var look_ahead: float = 0.55

@export_group("Follow")
@export var follow_speed_h: float = 14.0
## Deliberately slower than horizontal -- this is what stops jumps and stairs
## from snapping the camera vertically.
@export var follow_speed_v: float = 5.5

@export_group("Look")
@export var mouse_sensitivity: float = 0.0032
@export var invert_y: bool = false
@export var min_pitch: float = -60.0
@export var max_pitch: float = 32.0
## Easing on raw mouse input. Higher is more direct, lower is more floaty.
@export var look_smoothing: float = 24.0

@export_group("Distance")
@export var distance: float = 18.0
@export var min_distance: float = 10.0
@export var max_distance: float = 30.0
@export var zoom_step: float = 0.6
@export var shoulder_offset: float = 0.45
@export var sprint_pullback: float = 1.3

@export_group("Auto-align")
@export var auto_align: bool = true
## Seconds of mouse stillness before the rig starts drifting back behind.
@export var align_delay: float = 1.1
@export var align_speed: float = 2.2
## Below this speed the character is not really travelling, so do not recentre.
@export var align_min_speed: float = 1.2

@export_group("Feel")
@export var base_fov: float = 62.0
@export var sprint_fov: float = 72.0
@export var fov_speed: float = 4.0

@export_group("HD-2D")
## Orthogonal projection with a locked tilt - the Octopath / HD-2D diorama look.
## The sprite keeps a constant on-screen size however close the camera gets, so
## zoom works on the ortho size rather than on the spring arm.
@export var orthogonal: bool = true
## Width of the view in metres. Smaller is more zoomed in.
@export var ortho_size: float = 10.0
@export var ortho_min: float = 5.0
@export var ortho_max: float = 26.0
@export var ortho_step: float = 1.0
## How much wider the view goes while sprinting, in metres.
@export var sprint_widen: float = 1.6
## The fixed diorama tilt, in degrees below the horizon.
@export var fixed_pitch: float = -28.0
## Hold the tilt no matter what the mouse does. Free pitch breaks the look.
@export var lock_pitch: bool = true
## Snap yaw to this many degrees, the way HD-2D games rotate the diorama in
## steps. 0 leaves the yaw free.
@export var yaw_snap: float = 0.0
## Keep the tilt-shift blur tied to how far the view reaches. Fixed distances
## look right at one zoom and smear the whole background to white at another,
## because an ortho view can be 10 m or 60 m deep for the same camera.
@export var dof_follows_zoom: bool = true
## Where the far blur starts, as a multiple of the ortho size.
@export var dof_far_scale: float = 1.2

@onready var _yaw: Node3D = $Yaw
@onready var _pitch: Node3D = $Yaw/Pitch
@onready var _arm: SpringArm3D = $Yaw/Pitch/SpringArm3D
@onready var _cam: Camera3D = $Yaw/Pitch/SpringArm3D/Camera3D

var _target: Node3D
var _yaw_angle: float = 0.0
var _pitch_angle: float = -0.12
var _yaw_wanted: float = 0.0
var _pitch_wanted: float = -0.12
var _mouse_idle: float = 0.0
var _velocity := Vector3.ZERO
var _running := false
var _zoom: float = 5.0
var _size: float = 12.0


func _ready() -> void:
	_target = get_node_or_null(target_path) as Node3D
	if _target == null:
		_target = get_parent() as Node3D
	top_level = true                      # follow by hand, do not inherit motion
	_zoom = distance
	_arm.spring_length = distance
	_size = ortho_size
	if orthogonal:
		_cam.projection = Camera3D.PROJECTION_ORTHOGONAL
		_cam.size = ortho_size
		# Ortho framing does not care how far back the camera sits, so it sits a
		# long way back on purpose: close in, the near plane slices through
		# whatever is behind the character. Nothing may shorten the arm either -
		# a collapsed arm brings the slicing straight back. What the camera then
		# looks through is WallFade's job, not the spring arm's.
		_arm.collision_mask = 0
	if lock_pitch:
		_pitch_angle = deg_to_rad(fixed_pitch)
		_pitch_wanted = _pitch_angle
	if _target != null:
		global_position = _target.global_position + Vector3.UP * pivot_height
		_yaw_angle = _target.global_rotation.y
		_yaw_wanted = _yaw_angle
		var body := _target as PhysicsBody3D
		if body != null:
			_arm.add_excluded_object(body.get_rid())
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


## Called every physics frame by the character.
func set_motion(world_velocity: Vector3, running: bool) -> void:
	_velocity = world_velocity
	_running = running


## Basis whose -Z is the direction the camera looks, flattened. Movement input
## should be built from this so "forward" means "away from the camera".
func get_yaw_basis() -> Basis:
	return _yaw.global_transform.basis


func get_eye_position() -> Vector3:
	return _cam.global_position


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
	elif event is InputEventMouseButton:
		var click := event as InputEventMouseButton
		if click.pressed:
			if click.button_index == MOUSE_BUTTON_WHEEL_UP:
				if orthogonal:
					ortho_size = clampf(ortho_size - ortho_step, ortho_min, ortho_max)
				else:
					distance = clampf(distance - zoom_step, min_distance, max_distance)
			elif click.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				if orthogonal:
					ortho_size = clampf(ortho_size + ortho_step, ortho_min, ortho_max)
				else:
					distance = clampf(distance + zoom_step, min_distance, max_distance)
			elif Input.mouse_mode != Input.MOUSE_MODE_CAPTURED:
				Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	elif event is InputEventMouseMotion and Input.mouse_mode == Input.MOUSE_MODE_CAPTURED:
		var motion := event as InputEventMouseMotion
		_yaw_wanted -= motion.relative.x * mouse_sensitivity
		if not lock_pitch:
			var dy: float = motion.relative.y * mouse_sensitivity
			_pitch_wanted = clampf(_pitch_wanted + (dy if invert_y else -dy),
				deg_to_rad(min_pitch), deg_to_rad(max_pitch))
		_mouse_idle = 0.0


func _process(delta: float) -> void:
	if _target == null:
		return
	_mouse_idle += delta

	var flat := Vector3(_velocity.x, 0.0, _velocity.z)
	var speed := flat.length()

	# --- drift back behind the character once the player stops steering ------
	if auto_align and _mouse_idle > align_delay and speed > align_min_speed:
		# yaw whose -Z points along travel puts the camera behind the character
		var behind := atan2(-flat.x, -flat.z)
		var blend: float = 1.0 - exp(-align_speed * delta)
		_yaw_wanted += angle_difference(_yaw_wanted, behind) * blend

	# --- ease the look angles (frame-rate independent) ----------------------
	var look_t: float = 1.0 - exp(-look_smoothing * delta)
	var yaw_goal := _yaw_wanted
	if yaw_snap > 0.0:
		var step := deg_to_rad(yaw_snap)
		yaw_goal = round(_yaw_wanted / step) * step
	_yaw_angle += angle_difference(_yaw_angle, yaw_goal) * look_t
	if lock_pitch:
		_pitch_wanted = deg_to_rad(fixed_pitch)
	_pitch_angle = lerpf(_pitch_angle, _pitch_wanted, look_t)
	_yaw.rotation.y = _yaw_angle
	_pitch.rotation.x = _pitch_angle

	# --- follow, with a softer vertical spring than horizontal ---------------
	var lead := Vector3.ZERO
	if speed > 0.05:
		lead = flat.normalized() * look_ahead * clampf(speed / maxf(1.0, distance), 0.0, 1.0)
	var goal: Vector3 = _target.global_position + Vector3.UP * pivot_height + lead
	var th: float = 1.0 - exp(-follow_speed_h * delta)
	var tv: float = 1.0 - exp(-follow_speed_v * delta)
	var pos := global_position
	pos.x = lerpf(pos.x, goal.x, th)
	pos.z = lerpf(pos.z, goal.z, th)
	pos.y = lerpf(pos.y, goal.y, tv)
	global_position = pos

	# --- distance, shoulder offset and FOV open up with speed ---------------
	var want_zoom := distance + (sprint_pullback if _running else 0.0)
	_zoom = lerpf(_zoom, want_zoom, th)
	_arm.spring_length = _zoom
	_arm.position.x = lerpf(_arm.position.x, shoulder_offset, th)
	var ease: float = 1.0 - exp(-fov_speed * delta)
	if orthogonal:
		# ortho has no FOV: the view opens up by widening the box instead
		var want_size := ortho_size + (sprint_widen if _running else 0.0)
		_size = lerpf(_size, want_size, ease)
		_cam.size = _size
		if dof_follows_zoom:
			var attrs := _cam.attributes as CameraAttributesPractical
			if attrs != null:
				# Focus sits on the character, not at a fixed distance: the
				# camera is metres back, so absolute distances would blur the
				# entire scene. The sharp band widens as you zoom out.
				var focus: float = _zoom
				var band: float = _size * dof_far_scale * 0.5
				attrs.dof_blur_far_distance = focus + band
				attrs.dof_blur_far_transition = maxf(1.0, band)
				attrs.dof_blur_near_distance = maxf(0.4, focus - band)
				attrs.dof_blur_near_transition = maxf(1.0, band * 0.8)
	else:
		var want_fov := sprint_fov if _running else base_fov
		_cam.fov = lerpf(_cam.fov, want_fov, ease)
