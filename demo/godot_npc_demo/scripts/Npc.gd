extends Area2D
class_name Npc

## One city NPC. Built entirely from primitives so the project carries no
## binary art assets — see the note in project.godot.
##
## The archetype string is the contract with the server: it is passed
## straight through to /chat and must match a key in gguf_server.ARCHETYPE_GGUF
## ("police officer", not "police_officer" or "PoliceOfficer").

const RADIUS := 26.0
const INTERACT_RADIUS := 78.0

var archetype: String = ""
var display_name: String = ""
var tint: Color = Color.WHITE

var _prompt: Label


static func create(p_archetype: String, p_display_name: String, p_tint: Color, p_position: Vector2) -> Npc:
	var npc := Npc.new()
	npc.archetype = p_archetype
	npc.display_name = p_display_name
	npc.tint = p_tint
	npc.position = p_position
	return npc


func _ready() -> void:
	# The Area2D itself is only used as an interaction trigger; it never
	# collides with the player's body, so the player can walk "through" an
	# NPC. That is intentional for a demo — getting stuck on an NPC while
	# trying to talk to it is a worse failure than clipping.
	var shape := CollisionShape2D.new()
	var circle := CircleShape2D.new()
	circle.radius = INTERACT_RADIUS
	shape.shape = circle
	add_child(shape)

	queue_redraw()

	var name_label := Label.new()
	name_label.text = display_name
	name_label.add_theme_font_size_override("font_size", 13)
	name_label.add_theme_color_override("font_color", Color(0.92, 0.94, 1.0))
	name_label.add_theme_color_override("font_shadow_color", Color(0, 0, 0, 0.8))
	name_label.add_theme_constant_override("shadow_offset_y", 1)
	name_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	name_label.size = Vector2(180, 18)
	name_label.position = Vector2(-90, -RADIUS - 34)
	add_child(name_label)

	var role_label := Label.new()
	role_label.text = archetype
	role_label.add_theme_font_size_override("font_size", 10)
	role_label.add_theme_color_override("font_color", tint)
	role_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	role_label.size = Vector2(180, 14)
	role_label.position = Vector2(-90, -RADIUS - 20)
	add_child(role_label)

	_prompt = Label.new()
	_prompt.text = "[E] talk"
	_prompt.add_theme_font_size_override("font_size", 11)
	_prompt.add_theme_color_override("font_color", Color(1, 0.86, 0.4))
	_prompt.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_prompt.size = Vector2(180, 14)
	_prompt.position = Vector2(-90, RADIUS + 8)
	_prompt.visible = false
	add_child(_prompt)


func _draw() -> void:
	# Simple stylised figure: shadow, body, head. Enough to read as a person
	# at a glance without any texture import. Labels are child nodes, so they
	# draw on top of this automatically.
	draw_circle(Vector2(0, RADIUS * 0.92), RADIUS * 0.62, Color(0, 0, 0, 0.16))
	draw_circle(Vector2.ZERO, RADIUS, tint)
	draw_arc(Vector2.ZERO, RADIUS, 0, TAU, 40, tint.lightened(0.35), 2.5, true)
	draw_circle(Vector2(0, -RADIUS * 0.55), RADIUS * 0.42, tint.lightened(0.55))


func set_prompt_visible(value: bool) -> void:
	if _prompt:
		_prompt.visible = value


## True when the player is close enough to start a conversation. Compared
## against the same radius the CollisionShape2D uses, so what the prompt says
## and what the interact key does can never disagree.
func is_player_in_range(player_position: Vector2) -> bool:
	return global_position.distance_to(player_position) <= INTERACT_RADIUS
