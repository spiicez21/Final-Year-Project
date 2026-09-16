extends Node
class_name VoiceInput

## Records the player's microphone while push-to-talk is held.
##
## The microphone plays into its own muted bus ("PlayerMic") whose
## AudioEffectCapture collects the samples, so the player never hears
## themselves. Samples are drained every frame (the capture ring buffer only
## holds a second) and handed back on stop() as raw 32-bit float frames at the
## engine's mix rate. The server downmixes, resamples and transcribes them
## (backend/stt.py) -- nothing about speech recognition lives in the game.
##
## Needs `audio/driver/enable_input=true` in project.godot, and on Windows the
## "Let desktop apps access your microphone" privacy setting. Without either,
## recording yields silence and the server answers "nothing heard".

const BUS := "PlayerMic"
const MAX_SECONDS := 30.0   # Whisper's window; recording stops itself here

signal auto_stopped

var _player: AudioStreamPlayer
var _capture: AudioEffectCapture
var _frames := PackedVector2Array()
var _recording := false
var _elapsed := 0.0
var _monitoring := false   # settings panel level bar: listen without keeping samples
var _peak := 0.0


func _ready() -> void:
	var index := AudioServer.get_bus_index(BUS)
	if index == -1:
		index = AudioServer.bus_count
		AudioServer.add_bus(index)
		AudioServer.set_bus_name(index, BUS)
	# Muted: effects still run on the bus, but nothing reaches the speakers.
	AudioServer.set_bus_mute(index, true)
	_capture = AudioEffectCapture.new()
	_capture.buffer_length = 1.0
	AudioServer.add_bus_effect(index, _capture)

	_player = AudioStreamPlayer.new()
	_player.stream = AudioStreamMicrophone.new()
	_player.bus = BUS
	add_child(_player)


func is_recording() -> bool:
	return _recording


func start() -> void:
	if _recording:
		return
	_frames = PackedVector2Array()
	_elapsed = 0.0
	_capture.clear_buffer()
	if not _player.playing:
		_player.play()
	_recording = true


## Live input level for the settings panel, without recording.
func set_monitoring(on: bool) -> void:
	_monitoring = on
	_peak = 0.0
	if on and not _player.playing:
		_capture.clear_buffer()
		_player.play()
	elif not on and not _recording:
		_player.stop()


## After the input device changes, so the stream reopens on the new one.
func restart_monitoring() -> void:
	if _recording:
		return
	_player.stop()
	_capture.clear_buffer()
	if _monitoring:
		_player.play()


## Peak sample level of recent audio, 0..1.
func level() -> float:
	return _peak


## Stops recording. Returns {"pcm": float32 interleaved stereo bytes,
## "sample_rate": int, "channels": 2, "seconds": float}.
func stop() -> Dictionary:
	if not _recording:
		return {"pcm": PackedByteArray(), "sample_rate": int(AudioServer.get_mix_rate()), "channels": 2, "seconds": 0.0}
	_drain()
	if not _monitoring:
		_player.stop()
	_recording = false
	var rate := int(AudioServer.get_mix_rate())
	return {"pcm": _frames.to_byte_array(), "sample_rate": rate, "channels": 2,
		"seconds": float(_frames.size()) / rate}


func _process(delta: float) -> void:
	if _monitoring and not _recording:
		var chunk := _capture.get_buffer(_capture.get_frames_available())
		var peak := 0.0
		for f in chunk:
			peak = maxf(peak, maxf(absf(f.x), absf(f.y)))
		# Decay rather than drop to zero between frames, so the bar is readable.
		_peak = maxf(peak, _peak * 0.85)
		return
	if not _recording:
		return
	_drain()
	_elapsed += delta
	if _elapsed >= MAX_SECONDS:
		auto_stopped.emit()


func _drain() -> void:
	var available := _capture.get_frames_available()
	if available > 0:
		_frames.append_array(_capture.get_buffer(available))
