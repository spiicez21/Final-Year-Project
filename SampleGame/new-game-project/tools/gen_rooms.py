"""Generate the right-sized room shells the reduced plan needs.

The department's real numbers put every room well under the size it was built
at, so the shells are re-cut here rather than by hand: floor grid, four walls
with the door on the corridor side and windows on the outer side, collision, and
the furniture instance if a props scene exists for that size.

Run from the project root:  python tools/gen_rooms.py

Sizes come from plan_layout.NEW_SIZE, so the two stay in step.
"""

import os

import tscn

CELL = 4.0
CEILING = 3.0

# glTF Y-up export maps blender -Y onto game +Z, so kit props end up facing +Z.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
WALL_ALONG_X = "0, 0, 1, 0, 1, 0, -1, 0, 0"
WALL_ALONG_Z = FACE_PZ

# Which shells to cut, and where the door goes. `door` is the distance along the
# corridor wall to the middle of the door leaf; the corridor wall is the z = D
# side for every room in this building.
ROOMS = [
    # Wider than deep: the board goes on the short end wall, the corridor and
    # its door run along one long side and the windows along the other, which is
    # how the real rooms are laid out.
    ("Classroom", 12, 8, {"door": 10.5, "windows": "far"}),
    ("Lab", 16, 12, {"door": 14.5, "windows": "far"}),
    ("Faculty", 12, 8, {"door": 10.5, "windows": "far"}),
    ("Faculty", 8, 8, {"door": 6.5, "windows": "far"}),   # the HOD's room
    ("Toilet", 12, 8, {"door": 10.5, "windows": "none"}),
    ("StoreRoom", 8, 8, {"door": 1.0, "windows": "none"}),
    ("EntranceHall", 16, 16, {"door": 3.0, "windows": "far"}),
]

HEAD = '''[gd_scene format=3]

[ext_resource type="PackedScene" uid="uid://cgrpb6wire5c3" path="res://Assets/Floor.glb" id="1_floor"]
[ext_resource type="PackedScene" uid="uid://b04icmqe3k3eu" path="res://Assets/Door_Wall.glb" id="2_door"]
[ext_resource type="PackedScene" uid="uid://c46nkyfudvx" path="res://Assets/Wall_2m.glb" id="3_w2"]
[ext_resource type="PackedScene" uid="uid://c6hj4j5vvbxcy" path="res://Assets/Wall_4m.glb" id="4_w4"]
[ext_resource type="PackedScene" uid="uid://b88wvq7av8osy" path="res://Assets/Window_Wall.glb" id="5_win"]
%(furn_ext)s
[sub_resource type="BoxShape3D" id="sh_floor"]
size = Vector3(%(w)s, 0.2, %(d)s)

[sub_resource type="BoxShape3D" id="sh_wx4"]
size = Vector3(4, 3, 0.2)

[sub_resource type="BoxShape3D" id="sh_wx2"]
size = Vector3(2, 3, 0.2)

[sub_resource type="BoxShape3D" id="sh_wz4"]
size = Vector3(0.2, 3, 4)

[sub_resource type="BoxShape3D" id="sh_wz2"]
size = Vector3(0.2, 3, 2)

[node name="%(kind)s_%(w)sx%(d)s" type="Node3D" unique_id=%(root)d]
'''


def n(v):
    return "%g" % round(v, 4)


def inst(name, parent, res, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" parent="%s" unique_id=%d instance=ExtResource("%s")]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            % (name, parent, tscn.new_id(), res, rot, n(x), n(y), n(z)))


def group(name, parent="."):
    return '\n[node name="%s" type="Node3D" parent="%s" unique_id=%d]\n' % (
        name, parent, tscn.new_id())


def col(name, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="CollisionShape3D" parent="Collision" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def run(length, door=None, windows=False):
    """Fill a wall run with kit pieces.

    Works in 2 m segments because that is what the door leaf and the short
    filler are; pairs of plain segments merge back into a 4 m panel so the run
    uses as few instances as it can.
    """
    count = int(length // 2)
    kinds = []
    for i in range(count):
        mid = i * 2.0 + 1.0
        if door is not None and abs(mid - door) < 1.0:
            kinds.append("door")
        elif windows:
            kinds.append("win")
        else:
            kinds.append("wall")

    pieces = []
    i = 0
    while i < count:
        here = kinds[i]
        if here == "door":
            pieces.append(("2_door", i * 2.0, 2.0))
            i += 1
            continue
        # merge a matching pair into one 4 m panel when it lands on the grid
        if i + 1 < count and kinds[i + 1] == here and i % 2 == 0:
            pieces.append(("5_win" if here == "win" else "4_w4", i * 2.0, 4.0))
            i += 2
        else:
            pieces.append(("3_w2", i * 2.0, 2.0))
            i += 1
    return pieces


def build(kind, w, d, cfg):
    furn = "Scene/Props/%sFurniture_%dx%d.tscn" % (kind, w, d)
    has_furn = os.path.exists(tscn.ROOT + "/" + furn)
    head = HEAD % {
        "kind": kind, "w": w, "d": d, "root": tscn.new_id(),
        "furn_ext": ('[ext_resource type="PackedScene" path="res://%s" id="6_furn"]\n'
                     % furn) if has_furn else "",
    }
    out = [head]
    cols = []

    out.append(group("Floors"))
    k = 0
    for gx in range(int(CELL / 2), int(w), int(CELL)):
        for gz in range(int(CELL / 2), int(d), int(CELL)):
            k += 1
            out.append(inst("Floor%02d" % k, "Floors", "1_floor", gx, 0, gz))
    cols.append(col("CFloor", "sh_floor", w / 2.0, 0, d / 2.0))

    out.append(group("Walls"))
    windows = cfg.get("windows", "far")
    # z = D is the corridor side and carries the door; z = 0 faces out
    for side, gz, door, win in (("S", 0.0, None, windows == "far"),
                                ("N", float(d), cfg.get("door"), False)):
        for i, (res, at, length) in enumerate(run(w, door, win)):
            out.append(inst("Wall%s%02d" % (side, i + 1), "Walls", res,
                            at + length / 2.0, 0, gz, WALL_ALONG_X))
            cols.append(col("CWall%s%02d" % (side, i + 1),
                            "sh_wx4" if length > 2.5 else "sh_wx2",
                            at + length / 2.0, 1.5, gz))
    for side, gx in (("W", 0.0), ("E", float(w))):
        for i, (res, at, length) in enumerate(run(d, None, False)):
            out.append(inst("Wall%s%02d" % (side, i + 1), "Walls", res,
                            gx, 0, at + length / 2.0, WALL_ALONG_Z))
            cols.append(col("CWall%s%02d" % (side, i + 1),
                            "sh_wz4" if length > 2.5 else "sh_wz2",
                            gx, 1.5, at + length / 2.0))

    if has_furn:
        out.append(inst("Furniture", ".", "6_furn", 0, 0, 0))
    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)
    return "".join(out)


def main():
    for kind, w, d, cfg in ROOMS:
        text = build(kind, w, d, cfg)
        path = "Scene/Rooms/%s_%dx%d.tscn" % (kind, w, d)
        tscn.write(path, text)
        print("%-24s %2dx%-3d  %3d nodes" % (path.split("/")[-1], w, d,
                                             text.count("[node ")))


if __name__ == "__main__":
    main()
