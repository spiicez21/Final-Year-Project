extends CharacterBody3D
## A 2D billboard character moving through the 3D building.
##
## The sprite sheet holds 8 facing directions. Which one to draw depends on the
## character's heading RELATIVE TO THE CAMERA, not its world heading -- turn the
## camera around a standing character and you should see their back. So the
## angle below is measured between "which way the character faces" and "which
## way the camera looks", then quantised to the nearest 45 degrees.
##
## Camera behaviour lives in third_person_camera.gd on the CameraPivot child.

## Sheet order: DOWN, DOWN-RIGHT, RIGHT, UP-RIGHT, UP, UP-LEFT, LEFT, DOWN-LEFT.
const DIRS: PackedStringArray = ["s", "se", "e", "ne", "n", "nw", "w", "sw"]

@export_group("Movement")
@export var walk_speed: float = 3.2
@export var run_speed: float = 6.5
@export var acceleration: float = 16.0
@export var friction: float = 20.0
@export var jump_velocity: float = 4.8
@export var air_control: float = 0.35
## How quickly the drawn facing catches up, so a fast turn does not strobe
## through every one of the eight directions.
@export var turn_smoothing: float = 12.0

@onready var _sprite: AnimatedSprite3D = $Sprite
@onready var _rig: Node3D = $CameraPivot

var _facing := Vector3.BACK          # smoothed heading used to pick the sprite
var _gravity: float = ProjectSettings.get_setting("physics/3d/default_gravity", 9.8)
var _state := "idle"
var _was_on_floor := true
var _land_timer := 0.0


func _ready() -> void:
	_update_sprite()


func _physics_process(delta: float) -> void:
	var on_floor := is_on_floor()

	# movement is relative to where the camera is looking
	var input := Input.get_vector("move_left", "move_right", "move_forward", "move_back")
	var basis_yaw: Basis = _rig.get_yaw_basis()
	var dir := basis_yaw.x * input.x + basis_yaw.z * input.y
	dir.y = 0.0
	var moving := dir.length_squared() > 0.0001
	if moving:
		dir = dir.normalized()
		# ease the facing instead of snapping, so quick flicks do not strobe
		var t: float = 1.0 - exp(-turn_smoothing * delta)
		_facing = _facing.lerp(dir, t)
		if _facing.length_squared() > 0.0001:
			_facing = _facing.normalized()
	else:
		dir = Vector3.ZERO

	var running := Input.is_action_pressed("sprint") and moving
	var target := dir * (run_speed if running else walk_speed)
	var rate := acceleration if on_floor else acceleration * air_control
	var damp := friction if on_floor else friction * air_control
	var horizontal := Vector3(velocity.x, 0.0, velocity.z)
	horizontal = horizontal.move_toward(target, (rate if moving else damp) * delta)
	velocity.x = horizontal.x
	velocity.z = horizontal.z

	if not on_floor:
		velocity.y -= _gravity * delta
	elif Input.is_action_just_pressed("jump"):
		velocity.y = jump_velocity
		_state = "jump_takeoff"
		_land_timer = 0.18

	move_and_slide()

	if on_floor and not _was_on_floor:
		_state = "jump_land"
		_land_timer = 0.16
	_was_on_floor = on_floor

	_land_timer = maxf(0.0, _land_timer - delta)
	_update_state(on_floor, horizontal.length(), running)
	_update_sprite()
	_rig.set_motion(Vector3(velocity.x, 0.0, velocity.z), running)


func _update_state(on_floor: bool, speed: float, running: bool) -> void:
	if not on_floor:
		# hold the takeoff pose briefly, then switch to the airborne pose
		_state = "jump_takeoff" if _land_timer > 0.0 and _state == "jump_takeoff" else "jump_air"
		return
	if _land_timer > 0.0 and _state == "jump_land":
		return
	if speed < 0.15:
		_state = "idle"
	elif running and speed > walk_speed * 0.9:
		_state = "run"
	else:
		_state = "walk"


func _update_sprite() -> void:
	_play(_state + "_" + DIRS[_facing_index()])


## Index into DIRS for the character's heading as seen from the camera.
func _facing_index() -> int:
	# Direction from the character towards the camera, flattened.
	var eye: Vector3 = _rig.get_eye_position()
	var to_cam := eye - global_position
	to_cam.y = 0.0
	if to_cam.length_squared() < 0.0001:
		to_cam = Vector3.BACK
	# Facing the camera reads as 0 rad, which is the DOWN/front sprite.
	var rel := angle_difference(atan2(to_cam.x, to_cam.z), atan2(_facing.x, _facing.z))
	var idx := int(round(rel / (TAU / 8.0))) % 8
	return idx + 8 if idx < 0 else idx


func _play(anim: StringName) -> void:
	if _sprite.sprite_frames == null or not _sprite.sprite_frames.has_animation(anim):
		return
	if _sprite.animation != anim or not _sprite.is_playing():
		_sprite.play(anim)
