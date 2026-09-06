"""Furnish the classrooms: Scene/Props/ClassroomFurniture_12x8.tscn.

The room is 12 x 8 with the board on the x = 0 end wall, the corridor door on
the z = 8 side and windows down the z = 0 side, so the class faces -x.

Seating is the department's real class size: three 3-seat desk-and-bench pairs
across the room, seven rows deep, which is 63 places. Aisles land on the door so
the back row walks straight out.

Run from the project root:  python tools/gen_classroom.py
"""

import tscn

W, D = 12.0, 8.0
CEILING = 3.0

# glTF Y-up export maps blender -Y onto game +Z, so kit props end up facing +Z.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"

BENCH_Z = (1.9, 4.0, 6.1)              # three 1.8 m benches across the depth
ROW_X = [3.4 + 1.15 * i for i in range(7)]   # seven rows back from the dais
SEATS = len(BENCH_Z) * len(ROW_X) * 3

HEAD = '''[gd_scene format=3]

[ext_resource type="PackedScene" uid="uid://d4dy7pis2g3lb" path="res://Assets/writing_desk.glb" id="1_desk"]
[ext_resource type="PackedScene" uid="uid://c8fflcubmkd5e" path="res://Assets/bench_wood.glb" id="2_bench"]
[ext_resource type="PackedScene" path="res://Assets/green_board.glb" id="3_board"]
[ext_resource type="PackedScene" path="res://Assets/desk_wood.glb" id="4_tdesk"]
[ext_resource type="PackedScene" path="res://Assets/lab_chair.glb" id="5_chair"]
[ext_resource type="PackedScene" path="res://Assets/notice_board_cork.glb" id="6_notice"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="7_bin"]
[ext_resource type="PackedScene" path="res://Assets/wall_clock.glb" id="8_clock"]
[ext_resource type="PackedScene" path="res://Assets/ceiling_fan.glb" id="9_fan"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="10_tube"]

[sub_resource type="BoxShape3D" id="sh_desk"]
size = Vector3(0.61, 0.89, 1.8)

[sub_resource type="BoxShape3D" id="sh_bench"]
size = Vector3(0.36, 0.45, 1.6)

[sub_resource type="BoxShape3D" id="sh_tdesk"]
size = Vector3(0.9, 0.75, 1.8)

[sub_resource type="BoxShape3D" id="sh_chair"]
size = Vector3(0.5, 0.86, 0.5)

[sub_resource type="BoxShape3D" id="sh_bin"]
size = Vector3(0.4, 0.62, 0.8)

[node name="ClassroomFurniture" type="Node3D" unique_id=%d]
'''

LAMP = """
[node name="%s" type="OmniLight3D" parent="Lighting" unique_id=%d]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %s, 2.7, %s)
light_color = Color(1, 0.96, 0.9, 1)
light_energy = 1.6
light_volumetric_fog_energy = 1.1
light_specular = 0.2
light_bake_mode = 1
shadow_bias = 0.04
shadow_normal_bias = 1.5
distance_fade_enabled = true
distance_fade_begin = 16.0
distance_fade_shadow = 10.0
distance_fade_length = 5.0
omni_range = 8.0
omni_attenuation = 1.4
"""


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


def build():
    out = [HEAD % tscn.new_id()]
    cols = []

    # --- board and the teacher's end ---------------------------------------- #
    out.append(group("Front"))
    out.append(inst("GreenBoard", "Front", "3_board", 0.06, 0, D / 2))
    out.append(inst("TeacherDesk", "Front", "4_tdesk", 1.9, 0, D / 2 - 1.2, FACE_NX))
    cols.append(col("CTeacherDesk", "sh_tdesk", 1.9, 0.375, D / 2 - 1.2))
    out.append(inst("TeacherChair", "Front", "5_chair", 1.2, 0, D / 2 - 1.2, FACE_NX))
    cols.append(col("CTeacherChair", "sh_chair", 1.2, 0.43, D / 2 - 1.2))

    # --- the class: desk with its bench behind it, facing the board --------- #
    out.append(group("Seating"))
    for r, rx in enumerate(ROW_X):
        for c, cz in enumerate(BENCH_Z):
            tag = "R%dC%d" % (r + 1, c + 1)
            # desk faces -x, bench sits behind it so the row reads front to back
            out.append(inst("Desk" + tag, "Seating", "1_desk", rx, 0, cz, FACE_NX))
            cols.append(col("CDesk" + tag, "sh_desk", rx, 0.44, cz))
            out.append(inst("Bench" + tag, "Seating", "2_bench", rx + 0.62, 0, cz, FACE_NX))
            cols.append(col("CBench" + tag, "sh_bench", rx + 0.62, 0.22, cz))

    # --- wall fittings ------------------------------------------------------ #
    out.append(group("Props"))
    out.append(inst("NoticeBoard", "Props", "6_notice", W - 0.15, 1.1, D / 2, FACE_NX))
    out.append(inst("WallClock", "Props", "8_clock", 0.15, 2.4, 1.2))
    out.append(inst("Dustbins", "Props", "7_bin", W - 0.6, 0, 0.6, FACE_NX))
    cols.append(col("CDustbins", "sh_bin", W - 0.6, 0.31, 0.6))

    # --- lighting: battens down the rows, fans between them ----------------- #
    out.append(group("Lighting"))
    for i, fx in enumerate((3.5, 7.0, 10.0)):
        out.append(inst("Fan%02d" % (i + 1), "Lighting", "9_fan", fx, CEILING, D / 2))
    j = 0
    for lx in (2.5, 6.0, 9.5):
        for lz in (2.2, 5.8):
            j += 1
            out.append(inst("Batten%02d" % j, "Lighting", "10_tube", lx, 2.9, lz))
            out.append(LAMP % ("Lamp%02d" % j, tscn.new_id(), n(lx), n(lz)))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)
    return "".join(out)


if __name__ == "__main__":
    text = build()
    tscn.write("Scene/Props/ClassroomFurniture_%dx%d.tscn" % (W, D), text)
    print("seats:", SEATS, "| nodes:", text.count("[node "))
