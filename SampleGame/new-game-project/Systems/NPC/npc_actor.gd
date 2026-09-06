extends Area3D
class_name NpcActor

## A talkable campus NPC: billboard sprite + floating name plate, matching the
## HD-2D look of the player in Systems/Character/Player.tscn.
##
## `archetype` is the contract with the inference server. It is POSTed to
## /chat verbatim and must match a key in gguf_server.ARCHETYPE_GGUF
## ("police officer", not "police_officer").
##
## The sprite reuses the hero sheet with a per-NPC tint. That reads clearly as
## a placeholder rather than pretending to be finished character art.

const HERO_FRAMES := "res://Resources/Characters/hero_frames.tres"

## How close the player must be to talk. Also the radius of the trigger shape,
## so the "[E] talk" prompt and the interact key can never disagree.
const INTERACT_RANGE := 3.4

var archetype: String = ""
var display_name: String = ""
var role_label_text: String = ""
var tint: Color = Color.WHITE

var _prompt: Label3D


static func create(p_archetype: String, p_name: String, p_role: String,
		p_tint: Color, p_position: Vector3) -> NpcActor:
	var npc := NpcActor.new()
	npc.archetype = p_archetype
	npc.display_name = p_name
	npc.role_label_text = p_role
	npc.tint = p_tint
	npc.position = p_position
	return npc


func _ready() -> void:
	# Trigger volume only — NPCs deliberately have no body collision, because
	# getting wedged on an NPC you are trying to talk to is worse than being
	# able to walk through one.
	var shape := CollisionShape3D.new()
	var sphere := SphereShape3D.new()
	sphere.radius = INTERACT_RANGE
	shape.shape = sphere
	add_child(shape)

	# Values mirror the Sprite node in Systems/Character/Player.tscn so NPCs
	# sit in the world at the same scale and with the same filtering as the
	# player. Kept as literals for exactly that reason: they are a copy of
	# that scene's values, not independently chosen ones.
	var sprite := AnimatedSprite3D.new()
	sprite.sprite_frames = load(HERO_FRAMES)
	sprite.animation = &"idle_s"
	sprite.pixel_size = 0.00963
	sprite.billboard = BaseMaterial3D.BILLBOARD_FIXED_Y
	sprite.shaded = true
	sprite.double_sided = true
	sprite.alpha_cut = SpriteBase3D.ALPHA_CUT_DISCARD
	sprite.alpha_scissor_threshold = 0.35
	sprite.texture_filter = 3  # LINEAR_WITH_MIPMAPS, as in Player.tscn
	sprite.offset = Vector2(0, 97.5)
	sprite.cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_ON
	sprite.modulate = tint
	add_child(sprite)
	sprite.play(&"idle_s")

	add_child(_make_label(display_name, 0.22, 2.30, Color(0.95, 0.96, 1.0)))
	add_child(_make_label(role_label_text, 0.15, 2.06, tint))

	_prompt = _make_label("[E] talk", 0.16, 1.84, Color(1.0, 0.86, 0.4))
	_prompt.visible = false
	add_child(_prompt)


## `size` is the cap height of the text in world units (metres), so a 0.22
## label is about an eighth of the character's height.
##
## Deliberately NOT fixed_size and NOT no_depth_test: with either enabled the
## plates are drawn at constant screen size and through geometry, so every
## NPC on every floor of the building writes its name across the camera at
## once. World-space text with normal depth testing means a plate shrinks with
## distance and is hidden by the floor above it, which is the whole point of
## having three storeys.
func _make_label(text: String, size: float, height: float, color: Color) -> Label3D:
	var label := Label3D.new()
	label.text = text
	label.font_size = 64
	label.pixel_size = size / 64.0
	label.position = Vector3(0.0, height, 0.0)
	label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	label.modulate = color
	label.outline_modulate = Color(0, 0, 0, 0.85)
	label.outline_size = 10
	label.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS
	return label


func set_prompt_visible(value: bool) -> void:
	if _prompt:
		_prompt.visible = value


func is_player_in_range(player_position: Vector3) -> bool:
	return global_position.distance_to(player_position) <= INTERACT_RANGE
