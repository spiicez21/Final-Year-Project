extends Node
## Fades walls that stand between the camera and the character.
##
## The ortho camera sits a long way back so the near plane never slices geometry
## off behind the character, which means it now looks straight through whatever
## walls are in the way. Those walls fade out instead of blocking the view.
##
## It works off positions rather than physics. Walls in this project are visuals
## in one node and box colliders in a separate StaticBody per room, so a ray hit
## only ever names the room, never the panel that is actually in the way. So the
## wall panels are indexed into a grid once at load, and each frame the cells the
## camera-to-character line passes through are checked.

@export var player_path: NodePath
@export var rig_path: NodePath
## Roots to index. Normally the storey nodes.
@export var levels: Array[NodePath] = []
## How transparent an occluding wall goes. 1.0 is fully invisible.
@export_range(0.0, 1.0) var fade_to: float = 0.82
## Seconds to fade out and back in.
@export var fade_speed: float = 8.0
## How close to the sight line a panel counts as being in the way.
@export var radius: float = 2.2
## Grid the index is bucketed on.
@export var cell_size: float = 4.0
## Anything nearer the camera than this fraction of the way to the character is
## a candidate. Keeps the character's own room from fading around them.
@export var depth_limit: float = 0.85

var _player: Node3D
var _cam: Camera3D
var _grid: Dictionary = {}        # Vector2i -> Array[Dictionary] of panels
var _panels: Array = []           # every panel, for the fade-back pass
var _hot: Dictionary = {}         # panel index -> true while it is occluding


func _ready() -> void:
	_player = get_node_or_null(player_path) as Node3D
	var rig := get_node_or_null(rig_path)
	if rig != null:
		_cam = rig.get_node_or_null("Yaw/Pitch/SpringArm3D/Camera3D") as Camera3D
	for path in levels:
		var lvl := get_node_or_null(path) as Node3D
		if lvl != null:
			_index(lvl)


## Wall panels, by name: the kit calls them Wall_4m / Window_Wall / Door_Wall and
## the generated room shells WallN01, WallS01 and so on, so "wall" in the name is
## a reliable marker and costs one pass at load.
func _index(root: Node3D) -> void:
	var stack: Array = [root]
	while not stack.is_empty():
		var node: Node = stack.pop_back()
		for child in node.get_children():
			var is_wall := (child is Node3D
				and String(child.name).to_lower().contains("wall"))
			if is_wall:
				var meshes := _meshes_of(child as Node3D)
				if not meshes.is_empty():
					var panel := {
						"node": child,
						"meshes": meshes,
						"alpha": 0.0,
					}
					_panels.append(panel)
					var idx: int = _panels.size() - 1
					var p: Vector3 = (child as Node3D).global_position
					var key := _cell(p)
					# a panel is 4 m long, so register the neighbours too
					for dx in [-1, 0, 1]:
						for dz in [-1, 0, 1]:
							var k := Vector2i(key.x + dx, key.y + dz)
							if not _grid.has(k):
								_grid[k] = []
							_grid[k].append(idx)
			else:
				stack.append(child)


func _meshes_of(root: Node3D) -> Array:
	var found: Array = []
	var stack: Array = [root]
	while not stack.is_empty():
		var node: Node = stack.pop_back()
		if node is GeometryInstance3D:
			found.append(node)
		for child in node.get_children():
			stack.append(child)
	return found


func _cell(p: Vector3) -> Vector2i:
	return Vector2i(int(floor(p.x / cell_size)), int(floor(p.z / cell_size)))


func _process(delta: float) -> void:
	if _player == null or _cam == null:
		return

	var eye: Vector3 = _cam.global_position
	var mark: Vector3 = _player.global_position + Vector3.UP * 0.9
	var line: Vector3 = mark - eye
	var span: float = line.length()
	if span < 0.01:
		return
	var dir: Vector3 = line / span

	_hot.clear()
	# walk the cells the sight line crosses, a cell's worth at a time
	var steps: int = int(ceil(span / cell_size)) + 1
	var seen := {}
	for s in steps:
		var at: Vector3 = eye + dir * (float(s) * cell_size)
		var key := _cell(at)
		for dx in [-1, 0, 1]:
			for dz in [-1, 0, 1]:
				var k := Vector2i(key.x + dx, key.y + dz)
				if seen.has(k) or not _grid.has(k):
					continue
				seen[k] = true
				for idx in _grid[k]:
					if _hot.has(idx):
						continue
					if _occludes(_panels[idx]["node"], eye, dir, span):
						_hot[idx] = true

	var step: float = 1.0 - exp(-fade_speed * delta)
	for i in _panels.size():
		var panel: Dictionary = _panels[i]
		var want: float = fade_to if _hot.has(i) else 0.0
		if is_equal_approx(panel["alpha"], want):
			continue
		panel["alpha"] = lerpf(panel["alpha"], want, step)
		if absf(panel["alpha"] - want) < 0.01:
			panel["alpha"] = want
		for mesh in panel["meshes"]:
			(mesh as GeometryInstance3D).transparency = panel["alpha"]


func _occludes(node: Node3D, eye: Vector3, dir: Vector3, span: float) -> bool:
	var to: Vector3 = node.global_position - eye
	var along: float = to.dot(dir)
	if along <= 0.5 or along > span * depth_limit:
		return false
	return (to - dir * along).length() < radius
