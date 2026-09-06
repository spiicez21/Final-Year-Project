extends Node
## Cut-away culling for the HD-2D orthogonal camera.
##
## An orthogonal view volume is a box, not a cone: the top of the frame looks
## forward from several metres above the camera itself, so indoors it sails
## straight over the room's ceiling and lands on the floor above. Perspective
## hides that problem, ortho does not.
##
## So indoors the level is presented the way an HD-2D diorama is: the storey the
## player is on has its ceiling lifted off, and every storey above it is switched
## off. Storeys below stay on - the current floor's slab covers them, and they
## show through any stairwell.
##
## OUTDOORS none of that applies. Cutting the upper storeys away while the player
## is stood outside just lops the top off the building, so the culling only kicks
## in once there is actually something directly overhead. That is decided from
## occupancy grids built once at load: each storey records which 4 m cells it has
## content in and which it has a ceiling over. The player is "covered" when the
## storey above has content in their cell, or - on the top floor, where there is
## no storey above to test - when they are stood on that storey's own footprint.

@export var player_path: NodePath
## Storeys, ground floor first. Must match the order they are stacked in.
@export var levels: Array[NodePath] = []
## Light rigs belonging to each storey, same order. Optional.
@export var light_rigs: Array[NodePath] = []
## Height of one storey, used to work out which one the player is standing on.
@export var floor_height: float = 3.0
## Lift the ceiling off the storey the player is on.
@export var hide_own_ceiling: bool = true
## Grid the coverage test works on. Matches the 4 m floor tile.
@export var cell_size: float = 4.0
## Node names that count as "the lid on this storey". The room shells and the
## corridor runs end in Ceiling; the top storey's slab is a Roof node.
@export var ceiling_names: Array[String] = ["Ceiling", "Roof"]

var _player: Node3D
var _levels: Array[Node3D] = []
var _lights: Array[Node3D] = []
var _ceilings: Array = []          # one Array[Node3D] of ceiling groups per storey
var _cells: Array = []             # one Dictionary of occupied cells per storey
var _storey: int = -1
var _covered: bool = false


func _ready() -> void:
	_player = get_node_or_null(player_path) as Node3D
	for path in levels:
		var lvl := get_node_or_null(path) as Node3D
		if lvl != null:
			_levels.append(lvl)
			var groups := _find_ceilings(lvl)
			_ceilings.append(groups)
			_cells.append(_scan_cells(lvl))
	for path in light_rigs:
		var rig := get_node_or_null(path) as Node3D
		if rig != null:
			_lights.append(rig)
	_apply(0, false)


## Every ceiling group in a storey: the room shells, the corridor runs and the
## hall all park their tiles under a node whose name ends in "Ceiling", so one
## visibility flag per group covers the lot.
func _find_ceilings(root: Node) -> Array:
	var found: Array = []
	var stack: Array = [root]
	while not stack.is_empty():
		var node: Node = stack.pop_back()
		for child in node.get_children():
			var name := String(child.name)
			var is_lid := false
			for suffix in ceiling_names:
				if name.ends_with(suffix):
					is_lid = true
					break
			if child is Node3D and is_lid:
				found.append(child)
			else:
				stack.append(child)
	return found


## Which 4 m cells this storey is built on. Walked once at load. Only content
## sitting near the storey's own floor counts, so ceiling tiles, hanging lights
## and the storey above's fittings do not smear the footprint outwards.
func _scan_cells(root: Node3D) -> Dictionary:
	var cells := {}
	var base: float = root.global_position.y
	var stack: Array = [root]
	while not stack.is_empty():
		var node: Node = stack.pop_back()
		for child in node.get_children():
			if child is Node3D:
				var p: Vector3 = (child as Node3D).global_position
				if p.y > base - 1.4 and p.y < base + 1.6:
					cells[cell_of(p)] = true
			stack.append(child)
	return cells


func cell_of(p: Vector3) -> Vector2i:
	return Vector2i(int(floor(p.x / cell_size)), int(floor(p.z / cell_size)))


## Occupied cells of a storey, for anything else that wants to draw the layout.
func get_cells(storey: int) -> Dictionary:
	if storey < 0 or storey >= _cells.size():
		return {}
	return _cells[storey]


func current_storey() -> int:
	return _storey


func _process(_delta: float) -> void:
	if _player == null or _levels.is_empty():
		return
	# +0.5 so standing in a sunken floor still counts as that storey
	var idx: int = clampi(
		int(floor((_player.global_position.y + 0.5) / floor_height)),
		0, _levels.size() - 1)
	var cell := cell_of(_player.global_position)
	var covered := false
	if idx + 1 < _cells.size():
		covered = _cells[idx + 1].has(cell)
	else:
		# top storey: nothing above to test, so being on its own footprint is
		# what says "indoors" - and its roof still blocks an ortho camera
		covered = _cells[idx].has(cell)
	if idx != _storey or covered != _covered:
		_apply(idx, covered)


func _apply(idx: int, covered: bool) -> void:
	_storey = idx
	_covered = covered
	for i in _levels.size():
		_levels[i].visible = covered == false or i <= idx
		for group in _ceilings[i]:
			group.visible = not (covered and i == idx and hide_own_ceiling)
	for i in _lights.size():
		_lights[i].visible = covered == false or i <= idx
