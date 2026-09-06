"""Generate the CSE/IT computer lab furniture sets, and the Lab room shells for
the sizes that only existed as classrooms.

Writes Scene/Props/LabFurniture_<W>x<D>.tscn for each size in SIZES, and derives
Scene/Rooms/Lab_<W>x<D>.tscn from the matching Classroom shell.

Every teaching shell has the board wall at z = D, windows on z = 0 and the door
at x = 1 on the board wall, so students face +z. Layout is derived from the room
width: benches sit on a 4.6 m column pitch, the server corner and storage hug the
x = W wall, and the entry wall (x = 0) carries the notice board and extinguisher.

All numbers are game metres. The kit is authored at half scale and the .import
files apply nodes/root_scale = 2.0.
"""

import os

import tscn

SIZES = [(16, 12)]

CEILING = 3.0
BENCH_TOP = 0.75
SEAT_DX = 0.6           # two workstations per bench
COL0 = 3.4              # first bench column
COL_PITCH = 4.6
ROW_PITCH = 2.4         # bench row to bench row, leaving room for the chairs
ROW_FRONT = 1.6         # clear strip between the front bench and the board wall


def bench_rows(d):
    """Bench rows back from the board wall, front row first.

    Derived from the depth rather than fixed, so a shallower lab simply loses
    rows instead of pushing benches through the wall.
    """
    rows = []
    z = d - ROW_FRONT - 1.4
    while z > 2.6:
        rows.append(round(z, 2))
        z -= ROW_PITCH
    return tuple(rows)
DOOR_X = 1.0

# glTF Y-up export maps blender -Y onto game +Z, so a kit prop modelled facing -Y
# comes out of Godot facing +Z. Names below are the direction the prop ends up
# looking, which is the opposite of the wall it hangs on.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"      # mounts on the z=0 wall
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"    # mounts on the z=D wall
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"     # mounts on the x=0 wall
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"     # mounts on the x=W wall

# Green Board / White Board is modelled thin on X, so it keeps the kit's own basis.
BOARD_BASIS = FACE_NX

HEADER = '''[gd_scene format=3]

[ext_resource type="PackedScene" path="res://Assets/white_board.glb" id="1_board"]
[ext_resource type="PackedScene" uid="uid://d4dy7pis2g3lb" path="res://Assets/writing_desk.glb" id="2_wdesk"]
[ext_resource type="PackedScene" path="res://Assets/lab_bench_long.glb" id="3_bench"]
[ext_resource type="PackedScene" path="res://Assets/lab_chair.glb" id="4_chair"]
[ext_resource type="PackedScene" path="res://Assets/monitor_lcd.glb" id="5_mon"]
[ext_resource type="PackedScene" path="res://Assets/keyboard_mouse.glb" id="6_kb"]
[ext_resource type="PackedScene" path="res://Assets/pc_tower.glb" id="7_pc"]
[ext_resource type="PackedScene" path="res://Assets/server_rack.glb" id="8_rack"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="9_tube"]
[ext_resource type="PackedScene" path="res://Assets/steel_cupboard.glb" id="10_cup"]
[ext_resource type="PackedScene" path="res://Assets/projector_ceiling.glb" id="11_proj"]
[ext_resource type="PackedScene" path="res://Assets/projector_screen.glb" id="12_scr"]
[ext_resource type="PackedScene" path="res://Assets/ac_split_unit.glb" id="13_ac"]
[ext_resource type="PackedScene" path="res://Assets/ups_unit.glb" id="14_ups"]
[ext_resource type="PackedScene" path="res://Assets/printer_desktop.glb" id="15_prn"]
[ext_resource type="PackedScene" path="res://Assets/cable_duct.glb" id="16_duct"]
[ext_resource type="PackedScene" path="res://Assets/ceiling_fan.glb" id="17_fan"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="18_bin"]
[ext_resource type="PackedScene" path="res://Assets/wall_clock.glb" id="19_clock"]
[ext_resource type="PackedScene" path="res://Assets/exit_sign.glb" id="20_exit"]
[ext_resource type="PackedScene" path="res://Assets/fire_extinguisher.glb" id="21_fire"]
[ext_resource type="PackedScene" path="res://Assets/notice_board_cork.glb" id="22_notice"]
[ext_resource type="PackedScene" path="res://Assets/potted_plant.glb" id="23_plant"]

[sub_resource type="BoxShape3D" id="sh_board"]
size = Vector3(3.74, 1.37, 0.12)

[sub_resource type="BoxShape3D" id="sh_wdesk"]
size = Vector3(1.1, 0.88, 0.6)

[sub_resource type="BoxShape3D" id="sh_bench"]
size = Vector3(2.4, 0.87, 0.8)

[sub_resource type="BoxShape3D" id="sh_chair"]
size = Vector3(0.5, 0.86, 0.5)

[sub_resource type="BoxShape3D" id="sh_rack"]
size = Vector3(0.6, 2.0, 0.83)

[sub_resource type="BoxShape3D" id="sh_cupboard"]
size = Vector3(0.55, 1.9, 0.93)

[sub_resource type="BoxShape3D" id="sh_ups"]
size = Vector3(0.5, 0.6, 0.44)

[sub_resource type="BoxShape3D" id="sh_bin"]
size = Vector3(0.4, 0.62, 0.81)

[sub_resource type="BoxShape3D" id="sh_plant"]
size = Vector3(0.64, 0.95, 0.64)

[node name="LabFurniture" type="Node3D" unique_id=%d]
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
distance_fade_begin = 18.0
distance_fade_shadow = 11.0
distance_fade_length = 6.0
omni_range = 9.0
omni_attenuation = 1.4
"""


def n(v):
    return ("%g" % round(v, 4))


def inst(name, parent, res, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" parent="%s" unique_id=%d instance=ExtResource("%s")]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            % (name, parent, tscn.new_id(), res, rot, n(x), n(y), n(z)))


def group(name, parent="."):
    return '\n[node name="%s" type="Node3D" parent="%s" unique_id=%d]\n' % (
        name, parent, tscn.new_id())


def shape(name, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="CollisionShape3D" parent="Collision" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def bench_cols(w):
    """Bench column centres: as many 2.4 m benches as fit on a 4.6 m pitch."""
    cols = []
    x = COL0
    while x + 1.2 <= w - 1.0:
        cols.append(round(x, 2))
        x += COL_PITCH
    return tuple(cols)


def build(w, d):
    cols = bench_cols(w)
    board_x = w / 2.0
    screen_x = board_x + 3.6
    teacher_z = d - 1.4
    board_z = d - 0.1

    out = [HEADER % tscn.new_id()]
    colliders = []

    # --- teaching wall ---
    out.append(inst("WhiteBoard", ".", "1_board", board_x, 0, board_z, BOARD_BASIS))
    colliders.append(shape("Board", "sh_board", board_x, 1.5, board_z - 0.05))

    out.append(inst("TeacherDesk", ".", "2_wdesk", board_x, 0, teacher_z))
    colliders.append(shape("TeacherDesk", "sh_wdesk", board_x, 0.44, teacher_z))
    # teacher stands behind the desk looking down the room, so their screen faces +z
    out.append(inst("TeacherMonitor", ".", "5_mon", board_x + 0.25, 0.885, teacher_z - 0.1))
    out.append(inst("TeacherKeyboard", ".", "6_kb", board_x + 0.25, 0.885, teacher_z + 0.15))

    # --- student workstations ---
    # Benches are turned to put their cable riser on the board side; students sit
    # on the -z side of each bench and look +z at both screen and board.
    out.append(group("Workstations"))
    i = 0
    for bz in bench_rows(d):
        for bx in cols:
            i += 1
            tag = "%02d" % i
            out.append(inst("Bench" + tag, "Workstations", "3_bench", bx, 0, bz, FACE_NZ))
            colliders.append(shape("CBench" + tag, "sh_bench", bx, 0.435, bz))
            for s, dx in enumerate((-SEAT_DX, SEAT_DX)):
                st = "%s%s" % (tag, "AB"[s])
                out.append(inst("Monitor" + st, "Workstations", "5_mon",
                                bx + dx, BENCH_TOP, bz + 0.22, FACE_NZ))
                out.append(inst("Keyboard" + st, "Workstations", "6_kb",
                                bx + dx, BENCH_TOP, bz - 0.15, FACE_NZ))
                out.append(inst("Tower" + st, "Workstations", "7_pc",
                                bx + dx + 0.32, 0, bz + 0.2, FACE_NZ))
                out.append(inst("Chair" + st, "Workstations", "4_chair",
                                bx + dx, 0, bz - 0.95))
                colliders.append(shape("CChair" + st, "sh_chair",
                                       bx + dx, 0.43, bz - 0.95))

    # --- server corner, tucked against the board wall on the far side ---
    out.append(group("ServerCorner"))
    for k, rx in enumerate((w - 2.8, w - 1.8)):
        tag = "%02d" % (k + 1)
        out.append(inst("Rack" + tag, "ServerCorner", "8_rack", rx, 0, d - 0.8, FACE_NZ))
        colliders.append(shape("CRack" + tag, "sh_rack", rx, 1.0, d - 0.8))

    # --- storage / equipment on the x = W wall ---
    out.append(group("Equipment"))
    for k, cz in enumerate((d - 4.7, d - 3.5)):
        tag = "%02d" % (k + 1)
        out.append(inst("Cupboard" + tag, "Equipment", "10_cup", w - 0.35, 0, cz, FACE_NX))
        colliders.append(shape("CCupboard" + tag, "sh_cupboard", w - 0.35, 0.95, cz))
    out.append(inst("Printer", "Equipment", "15_prn", w - 0.35, 1.9, d - 3.5, FACE_NX))
    for k, ux in enumerate((w - 3.6, w - 1.1)):
        tag = "%02d" % (k + 1)
        out.append(inst("UPS" + tag, "Equipment", "14_ups", ux, 0, d - 1.7, FACE_NZ))
        colliders.append(shape("CUPS" + tag, "sh_ups", ux, 0.3, d - 1.7))

    # --- projection: screen on the board wall, projector throwing +z at it ---
    out.append(inst("ProjectorScreen", ".", "12_scr", screen_x, 0, board_z - 0.05, FACE_NZ))
    out.append(inst("Projector", ".", "11_proj", screen_x, CEILING, 11.0))

    # --- wall fittings ---
    out.append(group("WallProps"))
    out.append(inst("AC_Left", "WallProps", "13_ac", 0.15, 2.5, 6.0, FACE_PX))
    out.append(inst("AC_Right", "WallProps", "13_ac", w - 0.15, 2.5, 6.0, FACE_NX))
    for k, dx in enumerate(cols):
        out.append(inst("CableDuct%02d" % (k + 1), "WallProps", "16_duct",
                        dx, 1.1, 0.12))
    out.append(inst("WallClock", "WallProps", "19_clock", board_x, 2.4, 0.15))
    out.append(inst("ExitSign", "WallProps", "20_exit", DOOR_X, 2.6, d - 0.2, FACE_NZ))
    out.append(inst("FireExtinguisher", "WallProps", "21_fire", 0.25, 1.0, d - 2.0, FACE_PX))
    out.append(inst("NoticeBoard", "WallProps", "22_notice", 0.2, 1.1, 11.0, FACE_PX))

    # --- floor clutter ---
    out.append(group("Clutter"))
    out.append(inst("Dustbins", "Clutter", "18_bin", w - 0.7, 0, 1.2, FACE_NX))
    colliders.append(shape("CDustbins", "sh_bin", w - 0.7, 0.31, 1.2))
    for k, (px, pz) in enumerate(((0.9, d - 1.4), (0.9, 1.2))):
        tag = "%02d" % (k + 1)
        out.append(inst("Plant" + tag, "Clutter", "23_plant", px, 0, pz))
        colliders.append(shape("CPlant" + tag, "sh_plant", px, 0.48, pz))

    # --- lighting ---
    out.append(group("Lighting"))
    fans = [(board_x - 2.3, 7.0), (board_x + 2.3, 7.0), (board_x - 2.3, 11.0)]
    for k, (fx, fz) in enumerate(fans):
        out.append(inst("Fan%02d" % (k + 1), "Lighting", "17_fan", fx, CEILING, fz))
    for r, lz in enumerate((13.0, 9.0, 5.0)):
        for c, lx in enumerate(cols):
            tag = "%d%d" % (r + 1, c + 1)
            out.append(inst("Batten" + tag, "Lighting", "9_tube", lx, 3, lz))
            out.append(LAMP % ("Lamp" + tag, tscn.new_id(), n(lx), n(lz)))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(colliders)
    return "".join(out)


def make_room(w, d):
    """Derive Lab_<w>x<d>.tscn from the Classroom shell of the same size."""
    src = "Scene/Rooms/Classroom_%dx%d.tscn" % (w, d)
    dst = "Scene/Rooms/Lab_%dx%d.tscn" % (w, d)
    text = tscn.read(src)
    text = text.replace("Scene/Props/ClassroomFurniture_%dx%d.tscn" % (w, d),
                        "Scene/Props/LabFurniture_%dx%d.tscn" % (w, d))
    text = text.replace('[node name="Classroom_%dx%d"' % (w, d),
                        '[node name="Lab_%dx%d"' % (w, d))
    tscn.write(dst, text)
    return dst


def build_all():
    report = []
    for (w, d) in SIZES:
        props = build(w, d)
        tscn.write("Scene/Props/LabFurniture_%dx%d.tscn" % (w, d), props)
        room = "Scene/Rooms/Lab_%dx%d.tscn" % (w, d)
        # gen_rooms cuts the shells now; only fall back to deriving one from a
        # classroom for the older hand-authored sizes it does not cover.
        if not os.path.exists(tscn.ROOT + "/" + room):
            room = make_room(w, d)
        report.append(("%dx%d" % (w, d), len(bench_cols(w)) * len(bench_rows(d)),
                       props.count("[node "), room))
    return report


if __name__ == "__main__":
    for row in build_all():
        print(row)
