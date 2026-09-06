extends CharacterBody2D
class_name Player

## WASD/arrow-key player. Movement is read with raw key polling rather than
## InputMap actions — see the note in project.godot.

const SPEED := 260.0
const RADIUS := 20.0

## Set false by Main.gd while a dialogue is open, so typing "was" into the
## chat box does not also walk the player across the street.
var movement_enabled := true


func _ready() -> void:
	var shape := CollisionShape2D.new()
	var circle := CircleShape2D.new()
	circle.radius = RADIUS
	shape.shape = circle
	add_child(shape)

	queue_redraw()

	var label := Label.new()
	label.text = "you"
	label.add_theme_font_size_override("font_size", 11)
	label.add_theme_color_override("font_color", Color(0.8, 0.86, 1.0))
	label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	label.size = Vector2(120, 14)
	label.position = Vector2(-60, -RADIUS - 22)
	add_child(label)


func _draw() -> void:
	draw_circle(Vector2(0, RADIUS * 0.92), RADIUS * 0.62, Color(0, 0, 0, 0.16))
	draw_circle(Vector2.ZERO, RADIUS, Color(0.35, 0.72, 1.0))
	draw_arc(Vector2.ZERO, RADIUS, 0, TAU, 32, Color(0.7, 0.9, 1.0), 2.0, true)


func _physics_process(_delta: float) -> void:
	if not movement_enabled:
		velocity = Vector2.ZERO
		move_and_slide()
		return

	var direction := Vector2(
		float(Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT))
			- float(Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT)),
		float(Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN))
			- float(Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP))
	)

	velocity = direction.normalized() * SPEED
	move_and_slide()
