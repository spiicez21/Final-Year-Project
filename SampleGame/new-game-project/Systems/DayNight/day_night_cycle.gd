@tool
extends Node3D
## Drives a sun + sky through a school day, 08:00 to 18:00.
##
## Sun elevation follows a sine arc that peaks at solar noon (the midpoint of the
## span), so it rises in the east, passes overhead, and sets in the west. Light
## colour, sky tint, fog and ambient energy are all derived from that elevation,
## which keeps them consistent instead of animating each one separately.

signal hour_passed(hour: int)

@export_group("Time")
## Current time in 24h decimal hours (8.5 == 08:30).
@export_range(0.0, 24.0, 0.01) var time_of_day: float = 8.0:
	set(value):
		time_of_day = value
		_refresh()
@export_range(0.0, 24.0, 0.01) var start_hour: float = 8.0
@export_range(0.0, 24.0, 0.01) var end_hour: float = 18.0
## In-game minutes that pass per real second. 60 => one game hour per real minute.
@export var minutes_per_second: float = 20.0
@export var running: bool = true
## When the day ends, jump back to start_hour instead of holding at dusk.
@export var loop_day: bool = true

@export_group("Sun")
@export_range(0.0, 90.0) var max_elevation: float = 68.0
@export var azimuth_start: float = -100.0
@export var azimuth_end: float = 100.0
@export var sun_energy: float = 1.5
@export var color_noon: Color = Color(1.0, 0.97, 0.92)
@export var color_horizon: Color = Color(1.0, 0.56, 0.28)

@export_group("Sky")
@export var sky_top_noon: Color = Color(0.28, 0.48, 0.86)
@export var sky_horizon_noon: Color = Color(0.73, 0.83, 0.93)
@export var sky_top_dusk: Color = Color(0.19, 0.25, 0.45)
@export var sky_horizon_dusk: Color = Color(0.96, 0.62, 0.36)

@export_group("Clouds")
@export_range(0.0, 1.0) var cloud_coverage: float = 0.46
@export var cloud_lit_noon: Color = Color(1.0, 0.99, 0.97)
@export var cloud_lit_dusk: Color = Color(1.0, 0.72, 0.5)
@export var cloud_shade_noon: Color = Color(0.38, 0.42, 0.52)
@export var cloud_shade_dusk: Color = Color(0.3, 0.27, 0.38)

@export_group("Ambient")
## Floors 1 and 2 have the storey above as their ceiling, so almost no sky light
## reaches them -- the low end has to stay lifted or interiors read as caves.
@export var ambient_min: float = 0.7
@export var ambient_max: float = 1.25
@export var fog_density_min: float = 0.0008
@export var fog_density_max: float = 0.0022

@onready var _sun: DirectionalLight3D = $Sun
@onready var _world_env: WorldEnvironment = $WorldEnvironment

var _last_hour := -1


func _ready() -> void:
	# The Environment and Sky are shared .tres files. Duplicate them so the
	# per-frame tinting below never writes back into the saved resources.
	_make_local()
	_last_hour = int(time_of_day)
	_refresh()


func _process(delta: float) -> void:
	if Engine.is_editor_hint() or not running:
		return
	if end_hour - start_hour <= 0.0:
		return
	var next := time_of_day + (minutes_per_second * delta) / 60.0
	if next >= end_hour:
		next = start_hour if loop_day else end_hour
	time_of_day = next

	var hour := int(time_of_day)
	if hour != _last_hour:
		_last_hour = hour
		hour_passed.emit(hour)


## 0.0 at start_hour, 1.0 at end_hour.
func day_progress() -> float:
	var span := end_hour - start_hour
	if span <= 0.0:
		return 0.0
	return clampf((time_of_day - start_hour) / span, 0.0, 1.0)


## Sun height as 0.0 (on the horizon) to 1.0 (at its peak).
func sun_height() -> float:
	return sin(day_progress() * PI)


func time_string() -> String:
	var h := int(time_of_day)
	var m := int((time_of_day - float(h)) * 60.0)
	return "%02d:%02d" % [h, m]


func _make_local() -> void:
	if _world_env == null or _world_env.environment == null:
		return
	var env: Environment = _world_env.environment.duplicate(true)
	if env.sky != null:
		env.sky = env.sky.duplicate(true)
		if env.sky.sky_material != null:
			env.sky.sky_material = env.sky.sky_material.duplicate(true)
	_world_env.environment = env


func _refresh() -> void:
	if not is_inside_tree():
		return
	# @onready vars are still null while the scene is loading, and the exported
	# setters fire before _ready, so fall back to a direct lookup.
	var sun: DirectionalLight3D = _sun
	if sun == null:
		sun = get_node_or_null("Sun") as DirectionalLight3D
	var we: WorldEnvironment = _world_env
	if we == null:
		we = get_node_or_null("WorldEnvironment") as WorldEnvironment
	if sun == null:
		return

	var progress := day_progress()
	var height := sin(progress * PI)                      # 0 -> 1 -> 0
	var elevation := height * max_elevation
	var azimuth := lerpf(azimuth_start, azimuth_end, progress)

	# A DirectionalLight3D shines down its local -Z, so -elevation on X tips it
	# from the horizon towards straight down.
	sun.rotation_degrees = Vector3(-elevation, azimuth, 0.0)

	var warmth := pow(1.0 - height, 1.6)                  # 0 at noon, 1 at the horizon
	sun.light_color = color_noon.lerp(color_horizon, warmth)
	sun.light_energy = sun_energy * lerpf(0.25, 1.0, height)
	sun.shadow_enabled = true

	if we == null or we.environment == null:
		return
	var env := we.environment
	var horizon := sky_horizon_noon.lerp(sky_horizon_dusk, warmth)

	var top := sky_top_noon.lerp(sky_top_dusk, warmth)
	if env.sky != null:
		# The sky is a cloud shader by default, but still support the plain
		# procedural material in case it gets swapped back.
		var shader_sky := env.sky.sky_material as ShaderMaterial
		if shader_sky != null:
			shader_sky.set_shader_parameter("sky_top_color", top)
			shader_sky.set_shader_parameter("sky_horizon_color", horizon)
			# Clouds pick up the sun's warmth, and their shaded side leans
			# towards the sky colour rather than staying neutral grey.
			shader_sky.set_shader_parameter("cloud_color",
				cloud_lit_noon.lerp(cloud_lit_dusk, warmth))
			shader_sky.set_shader_parameter("cloud_dark",
				cloud_shade_noon.lerp(cloud_shade_dusk, warmth))
			shader_sky.set_shader_parameter("coverage", cloud_coverage)
		else:
			var mat := env.sky.sky_material as ProceduralSkyMaterial
			if mat != null:
				mat.sky_top_color = top
				mat.sky_horizon_color = horizon
				mat.ground_horizon_color = horizon

	env.ambient_light_energy = lerpf(ambient_min, ambient_max, height)
	env.fog_light_color = horizon
	env.fog_density = lerpf(fog_density_max, fog_density_min, height)
	env.fog_sun_scatter = lerpf(0.35, 0.1, height)
