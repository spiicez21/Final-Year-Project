extends OmniLight3D
## Makes a fixture misbehave. Three flavours, because a whole building of
## identically-wobbling lamps reads as an effect rather than as wiring.
##
##   HUM     barely-there ripple, the mains hum of a healthy tube
##   FAULTY  irregular flicker with the occasional dropout
##   DYING   mostly dark, stuttering back to life at random

enum Mode { HUM, FAULTY, DYING }

@export var mode: Mode = Mode.HUM
@export_range(0.0, 1.0) var amount: float = 0.08
@export var speed: float = 7.0
## Mean seconds between dropouts. 0 disables them.
@export var dropout_interval: float = 11.0
@export var dropout_length: float = 0.09
## Skip the whole update when the player is far away -- there are a lot of
## these and an off-screen lamp does not need to be simulated.
@export var cull_distance: float = 45.0

var _base_energy: float = 1.0
var _phase: float = 0.0
var _dropout_left: float = 0.0
var _next_dropout: float = 0.0
var _camera: Camera3D = null
var _check: float = 0.0


func _ready() -> void:
	_base_energy = light_energy
	# Desync every fixture, otherwise the whole building pulses in unison.
	_phase = randf() * 128.0
	_next_dropout = _roll_dropout()
	match mode:
		Mode.FAULTY:
			amount = maxf(amount, 0.26)
			speed = maxf(speed, 10.0)
		Mode.DYING:
			amount = maxf(amount, 0.5)
			speed = maxf(speed, 15.0)
			dropout_interval = minf(dropout_interval, 4.5)


func _roll_dropout() -> float:
	if dropout_interval <= 0.0:
		return -1.0
	return dropout_interval * (0.45 + randf() * 1.1)


func _process(delta: float) -> void:
	# refresh the camera reference occasionally rather than every frame
	_check -= delta
	if _check <= 0.0:
		_check = 0.5
		_camera = get_viewport().get_camera_3d()
	if _camera != null and global_position.distance_squared_to(_camera.global_position) > cull_distance * cull_distance:
		# Restore before bailing out, otherwise a lamp culled mid-dropout stays
		# dark for good.
		if not is_equal_approx(light_energy, _base_energy):
			light_energy = _base_energy
			_dropout_left = 0.0
		return

	_phase += delta * speed

	if _dropout_left > 0.0:
		_dropout_left -= delta
		light_energy = _base_energy * 0.06
		return
	if _next_dropout > 0.0:
		_next_dropout -= delta
		if _next_dropout <= 0.0:
			_dropout_left = dropout_length * (0.5 + randf())
			_next_dropout = _roll_dropout()
			return

	# Layered sines at incommensurable rates read as irregular without the
	# cost (or the frame-to-frame jitter) of sampling random noise.
	var n := sin(_phase) * 0.55 + sin(_phase * 2.37 + 1.7) * 0.3 + sin(_phase * 5.11 + 0.4) * 0.15
	if mode == Mode.DYING:
		n = sign(n) * pow(absf(n), 0.35)      # harder, more binary stutter
	var factor := 1.0 - amount * (0.5 - n * 0.5)
	light_energy = _base_energy * clampf(factor, 0.0, 1.6)
