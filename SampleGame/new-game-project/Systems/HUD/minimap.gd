extends Control
## Corner minimap for the HD-2D view.
##
## Drawn rather than rendered: the FloorCuller already knows which 4 m cells each
## storey is built on, so the map is those cells stamped out around the player.
## That costs a handful of draw_rect calls instead of a second pass over the
## whole level through a SubViewport.
##
## North stays up and the player arrow spins, which is the easier of the two to
## read while the diorama itself rotates underneath.

@export var player_path: NodePath
@export var culler_path: NodePath
## Camera rig, so the arrow can point where the view is facing.
@export var rig_path: NodePath
## Pixels per metre. Higher is more zoomed in.
@export var scale_px: float = 2.6
@export var cell_size: float = 4.0
@export var background: Color = Color(0.07, 0.08, 0.11, 0.72)
@export var built: Color = Color(0.78, 0.82, 0.9, 0.55)
@export var below: Color = Color(0.5, 0.55, 0.66, 0.22)
@export var arrow: Color = Color(1.0, 0.82, 0.35, 1.0)
@export var frame: Color = Color(0.9, 0.93, 1.0, 0.35)

var _player: Node3D
var _culler: Node
var _rig: Node3D


func _ready() -> void:
	_player = get_node_or_null(player_path) as Node3D
	_culler = get_node_or_null(culler_path)
	_rig = get_node_or_null(rig_path) as Node3D
	mouse_filter = Control.MOUSE_FILTER_IGNORE


func _process(_delta: float) -> void:
	queue_redraw()


func _draw() -> void:
	var box := Rect2(Vector2.ZERO, size)
	draw_rect(box, background, true)
	if _player == null or _culler == null:
		draw_rect(box, frame, false, 2.0)
		return

	var centre := size * 0.5
	var here: Vector3 = _player.global_position
	var storey: int = _culler.call("current_storey")
	var span: float = maxf(size.x, size.y) / scale_px * 0.5 + cell_size
	# a pixel of air round each cell so the grid reads as rooms, not a slab
	var tile := Vector2(cell_size, cell_size) * scale_px - Vector2.ONE

	# the storey below shows through faintly, so stairwells and voids read
	for level in [storey - 1, storey]:
		var cells: Dictionary = _culler.call("get_cells", level)
		if cells.is_empty():
			continue
		var tint: Color = built if level == storey else below
		for cell in cells:
			var world := Vector2(cell.x * cell_size, cell.y * cell_size)
			var offset := world - Vector2(here.x, here.z)
			if absf(offset.x) > span or absf(offset.y) > span:
				continue
			var at := centre + offset * scale_px
			var r := Rect2(at, tile)
			if box.intersects(r):
				draw_rect(r.intersection(box), tint, true)

	# player arrow, pointing the way the camera looks
	var yaw := 0.0
	if _rig != null:
		yaw = _rig.call("get_yaw_basis").get_euler().y
	var facing := Vector2(-sin(yaw), -cos(yaw))
	var side := Vector2(-facing.y, facing.x)
	var pts := PackedVector2Array([
		centre + facing * 8.0,
		centre - facing * 5.0 + side * 5.0,
		centre - facing * 5.0 - side * 5.0,
	])
	draw_colored_polygon(pts, arrow)
	draw_rect(box, frame, false, 2.0)
