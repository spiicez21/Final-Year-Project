"""Furnish the faculty rooms.

Writes Scene/Props/FacultyFurniture_<W>x<D>.tscn for each faculty size and
patches the matching Scene/Rooms/Faculty_<W>x<D>.tscn shell to instance it.

Every faculty shell has the same arrangement: windows along the z = 0 wall and
the door at x = 1 on the z = D wall. The layouts follow from that:

  * staff sit in back-to-back pods - two flat-top tables sharing a partition, a
    chair behind each - so the desks read as a real staff room rather than
    islands scattered over the floor. Tower, monitor and keyboard all stand on
    the table top, nothing on the floor;
  * pods are laid in bands parallel to the window wall with a walking aisle
    between bands, and the band nearest the window is the one staff face away
    from, so no screen takes direct glare;
  * the strip in front of the door wall stays clear for circulation and carries
    the visitor lounge, water cooler and bins;
  * steel almirahs line the far short wall, with the printer on a table at a
    height someone can actually reach.

All numbers are game metres. The prop kit is authored at half scale and the
.import files apply nodes/root_scale = 2.0.
"""

import tscn

CEILING = 3.0
DESK_TOP = 0.75          # desk_wood working height - a flat, solid top
DESK_W = 1.8             # desk_wood width
DESK_D = 0.9             # desk_wood depth, pods sit back to back on this
PARTITION_H = 0.5        # screen height above the desk top
CABIN_H = 2.1            # senior cabin partition height, open above

# glTF Y-up export maps blender -Y onto game +Z, so kit props end up facing +Z.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"

# --------------------------------------------------------------------------- #
# Per-size layouts, all room-local.
#
#   pods      - (centre z of the pod spine, [x of each desk column])
#               two desks per column, one either side of the spine, each with a
#               chair, monitor, keyboard and tower placed off it automatically
#   senior    - (x, z) of the HOD/senior desk, faced by two visitor chairs
#   cabin     - (x0, z0, x1, z1) partition runs walling the senior desk off,
#               2.1 m high with the gap between runs left as the cabin doorway
#   meeting    - (x, z) of a two-table meeting bay with four chairs round it
#   almirahs  - (x, z, facing) of each steel cupboard
#   printer   - (x, z, facing) of the printer table
#   cooler    - (x, z, facing) of the water cooler
#   lounge    - visitor seating near the door
# --------------------------------------------------------------------------- #
ROOMS = {
    # Year staff room: ten desks in back-to-back pods along the window wall,
    # with the year in-charge partitioned off in the far corner.
    (12, 8): {
        "pods": [(2.9, [1.4, 3.7, 6.0, 8.3, 10.6])],
        "senior": (2.2, 6.6),
        "cabin": [(0.0, 5.2, 4.6, 5.2), (4.6, 5.2, 4.6, 8.0)],
        "meeting": None,
        "almirahs": [(8.4, 7.6, FACE_NZ), (9.4, 7.6, FACE_NZ)],
        "printer": (10.9, 7.3, FACE_NZ),
        "cooler": (11.4, 5.6, FACE_NX),
        "lounge": {"sofa": (6.2, 6.4, FACE_NZ), "table": (6.2, 5.4)},
        "notice": (0.2, 3.0, FACE_PX),
        "plants": [(11.4, 1.0)],
        "bins": [(9.9, 5.6, FACE_NZ)],
        "battens": [(2.5, 2.9), (6.0, 2.9), (9.5, 2.9), (7.5, 6.4)],
        "fans": [(3.0, 2.9), (8.0, 2.9), (7.0, 6.4)],
        "acs": [(0.2, 2.0, FACE_PX), (11.8, 2.0, FACE_NX)],
        "clock": (6.0, 7.85, FACE_NZ),
        "exit": (10.5, 7.8),
    },
    # The HOD's own room: no staff desks, just the desk, visitors and storage.
    (8, 8): {
        "pods": [],
        "senior": (4.0, 3.2),
        "cabin": [],
        "meeting": None,
        "almirahs": [(7.6, 1.6, FACE_NX), (7.6, 2.8, FACE_NX)],
        "printer": (7.3, 5.4, FACE_NZ),
        "cooler": (0.6, 5.4, FACE_PX),
        "lounge": {"sofa": (2.6, 6.6, FACE_NZ), "table": (2.6, 5.5)},
        "notice": (0.2, 2.0, FACE_PX),
        "plants": [(7.2, 6.8)],
        "bins": [(5.4, 7.4, FACE_NZ)],
        "battens": [(2.5, 2.5), (5.5, 2.5), (4.0, 6.0)],
        "fans": [(4.0, 3.0), (4.0, 6.2)],
        "acs": [(7.8, 2.0, FACE_NX)],
        "clock": (4.0, 7.85, FACE_NZ),
        "exit": (6.5, 7.8),
    },
    # Deep rooms: two pod bands, storage down the far long wall.
    (12, 16): {
        "pods": [(3.3, [3.0, 5.9, 8.8]), (8.3, [3.0, 5.9, 8.8])],
        "senior": None,
        "cabin": [],
        "meeting": (8.4, 11.8),
        "almirahs": [(11.6, 11.2, FACE_NX), (11.6, 12.2, FACE_NX),
                     (11.6, 13.2, FACE_NX)],
        "benches": [(2.6, 11.2, FACE_PX)],
        "printer": (10.9, 14.6, FACE_NZ),
        "cooler": (0.6, 12.0, FACE_PX),
        "lounge": {"sofa": (5.2, 14.7, FACE_NZ), "table": (5.2, 13.4)},
        "notice": (0.2, 6.0, FACE_PX),
        "plants": [(0.8, 14.9), (11.4, 9.6)],
        "bins": [(2.2, 15.2, FACE_NZ)],
        "battens": [(1.6, 3.3), (6.4, 3.3), (11.2, 3.3),
                    (1.6, 8.3), (6.4, 8.3), (11.2, 8.3), (6.4, 13.4)],
        "fans": [(4.6, 3.3), (9.4, 3.3), (4.6, 8.3), (9.4, 8.3), (6.0, 13.4)],
        "acs": [(0.2, 3.0, FACE_PX), (11.8, 6.0, FACE_NX)],
        "clock": (6.0, 15.85, FACE_NZ),
        "exit": (1.0, 15.8),
    },
    # Shallow rooms: a single pod band under the windows, storage on the door
    # wall away from the door itself.
    (16, 8): {
        "pods": [(3.0, [3.0, 5.5, 8.0, 10.5, 13.0])],
        "senior": None,
        "cabin": [],
        "meeting": None,
        "almirahs": [(9.0, 7.6, FACE_NZ), (10.0, 7.6, FACE_NZ), (11.0, 7.6, FACE_NZ)],
        "printer": (12.6, 7.3, FACE_NZ),
        "cooler": (15.4, 6.6, FACE_NX),
        "lounge": {"sofa": (5.0, 7.3, FACE_NZ), "table": (5.0, 6.1)},
        "notice": (0.2, 4.0, FACE_PX),
        "plants": [(15.4, 1.0)],
        "bins": [(2.4, 7.4, FACE_NZ)],
        "battens": [(2.0, 3.0), (6.0, 3.0), (10.0, 3.0), (14.0, 3.0)],
        "fans": [(4.9, 3.0), (11.7, 3.0), (8.0, 6.6)],
        "acs": [(0.2, 4.0, FACE_PX), (15.8, 3.0, FACE_NX)],
        "clock": (10.0, 7.85, FACE_NZ),
        "exit": (1.0, 7.8),
    },
    (20, 8): {
        "pods": [(3.0, [3.0, 5.5, 8.0, 10.5, 13.0, 15.5, 18.0])],
        "senior": None,
        "cabin": [],
        "meeting": None,
        "almirahs": [(9.0, 7.6, FACE_NZ), (10.0, 7.6, FACE_NZ),
                     (11.0, 7.6, FACE_NZ), (12.0, 7.6, FACE_NZ)],
        "printer": (13.6, 7.3, FACE_NZ),
        "cooler": (19.4, 6.6, FACE_NX),
        "lounge": {"sofa": (5.0, 7.3, FACE_NZ), "table": (5.0, 6.1)},
        "notice": (0.2, 4.0, FACE_PX),
        "plants": [(19.4, 1.0), (16.4, 7.2)],
        "bins": [(2.4, 7.4, FACE_NZ)],
        "battens": [(2.0, 3.0), (6.0, 3.0), (10.0, 3.0), (14.0, 3.0), (18.0, 3.0)],
        "fans": [(4.9, 3.0), (11.7, 3.0), (18.5, 3.0), (8.0, 6.6)],
        "acs": [(0.2, 4.0, FACE_PX), (19.8, 3.0, FACE_NX)],
        "clock": (14.0, 7.85, FACE_NZ),
        "exit": (1.0, 7.8),
    },
    (24, 8): {
        "pods": [(3.0, [3.0, 5.5, 8.0, 10.5, 13.0, 15.5, 18.0, 20.5])],
        "senior": None,
        "cabin": [],
        "meeting": None,
        "almirahs": [(9.0, 7.6, FACE_NZ), (10.0, 7.6, FACE_NZ),
                     (11.0, 7.6, FACE_NZ), (12.0, 7.6, FACE_NZ), (13.0, 7.6, FACE_NZ)],
        "printer": (14.6, 7.3, FACE_NZ),
        "cooler": (23.4, 6.6, FACE_NX),
        "lounge": {"sofa": (5.0, 7.3, FACE_NZ), "table": (5.0, 6.1)},
        "notice": (0.2, 4.0, FACE_PX),
        "plants": [(23.4, 1.0), (17.4, 7.2)],
        "bins": [(2.4, 7.4, FACE_NZ)],
        "battens": [(2.0, 3.0), (6.0, 3.0), (10.0, 3.0),
                    (14.0, 3.0), (18.0, 3.0), (22.0, 3.0)],
        "fans": [(4.9, 3.0), (11.7, 3.0), (18.5, 3.0), (8.0, 6.6), (20.0, 6.6)],
        "acs": [(0.2, 4.0, FACE_PX), (23.8, 3.0, FACE_NX)],
        "clock": (18.0, 7.85, FACE_NZ),
        "exit": (1.0, 7.8),
    },
    # The department staff room: two pod bands of four, a senior desk in its own
    # corner, and the almirah wall behind it.
    (24, 16): {
        "pods": [(3.3, [2.9, 5.4, 7.9, 10.4, 12.9]),
                 (8.3, [2.9, 5.4, 7.9, 10.4, 12.9])],
        "senior": (20.0, 3.4),
        "cabin": [(17.6, 6.6, 24.0, 6.6), (16.2, 0.2, 16.2, 6.6)],
        "meeting": (19.2, 10.2),
        "almirahs": [(23.6, 7.0, FACE_NX), (23.6, 8.0, FACE_NX),
                     (23.6, 9.0, FACE_NX), (23.6, 10.0, FACE_NX),
                     (23.6, 11.0, FACE_NX),
                     (17.4, 15.6, FACE_NZ), (18.4, 15.6, FACE_NZ),
                     (19.4, 15.6, FACE_NZ), (20.4, 15.6, FACE_NZ)],
        "benches": [(14.0, 15.4, FACE_NZ)],
        "printer": (22.9, 13.0, FACE_NZ),
        "cooler": (0.6, 12.4, FACE_PX),
        "lounge": {"sofa": (5.6, 14.7, FACE_NZ), "sofa2": (8.4, 13.4, FACE_NX),
                   "table": (5.6, 13.4)},
        "notice": (0.2, 6.0, FACE_PX),
        "plants": [(0.8, 14.9), (16.4, 14.6), (17.0, 1.0)],
        "bins": [(2.2, 15.2, FACE_NZ)],
        "battens": [(1.6, 3.3), (6.4, 3.3), (11.2, 3.3), (16.0, 3.3),
                    (1.6, 8.3), (6.4, 8.3), (11.2, 8.3),
                    (6.4, 13.4), (14.0, 13.4), (21.0, 6.0)],
        "fans": [(4.6, 3.3), (9.4, 3.3), (14.2, 3.3),
                 (4.6, 8.3), (9.4, 8.3), (12.0, 13.4), (20.0, 4.0), (20.0, 10.0)],
        "acs": [(0.2, 3.0, FACE_PX), (23.8, 3.0, FACE_NX), (12.0, 15.8, FACE_NZ)],
        "clock": (12.0, 15.85, FACE_NZ),
        "exit": (1.0, 15.8),
    },
}

HEAD = '''[gd_scene format=3]

[ext_resource type="PackedScene" uid="uid://d4dy7pis2g3lb" path="res://Assets/desk_wood.glb" id="1_desk"]
[ext_resource type="PackedScene" path="res://Assets/lab_chair.glb" id="2_chair"]
[ext_resource type="PackedScene" path="res://Assets/monitor_lcd.glb" id="3_mon"]
[ext_resource type="PackedScene" path="res://Assets/keyboard_mouse.glb" id="4_kb"]
[ext_resource type="PackedScene" path="res://Assets/pc_tower.glb" id="5_pc"]
[ext_resource type="PackedScene" path="res://Assets/steel_cupboard.glb" id="6_cup"]
[ext_resource type="PackedScene" path="res://Assets/printer_desktop.glb" id="7_prn"]
[ext_resource type="PackedScene" path="res://Assets/office_sofa.glb" id="8_sofa"]
[ext_resource type="PackedScene" path="res://Assets/coffee_table.glb" id="9_ctable"]
[ext_resource type="PackedScene" path="res://Assets/notice_board_cork.glb" id="10_notice"]
[ext_resource type="PackedScene" path="res://Assets/potted_plant.glb" id="11_plant"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="12_bin"]
[ext_resource type="PackedScene" path="res://Assets/wall_clock.glb" id="13_clock"]
[ext_resource type="PackedScene" path="res://Assets/ceiling_fan.glb" id="14_fan"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="15_tube"]
[ext_resource type="PackedScene" path="res://Assets/desk_wood.glb" id="16_deskw"]
[ext_resource type="PackedScene" path="res://Assets/canteen_table_4.glb" id="17_table"]
[ext_resource type="PackedScene" path="res://Assets/water_cooler_ro.glb" id="18_cooler"]
[ext_resource type="PackedScene" path="res://Assets/ac_split_unit.glb" id="19_ac"]
[ext_resource type="PackedScene" path="res://Assets/exit_sign.glb" id="20_exit"]
[ext_resource type="PackedScene" path="res://Assets/canteen_chair.glb" id="21_cchair"]
[ext_resource type="PackedScene" path="res://Assets/bench_wood.glb" id="22_bench"]

[sub_resource type="StandardMaterial3D" id="m_partition"]
albedo_color = Color(0.35, 0.4, 0.45, 1)
roughness = 1.0

[sub_resource type="StandardMaterial3D" id="m_cabin"]
albedo_color = Color(0.72, 0.7, 0.64, 1)
roughness = 0.95

[sub_resource type="BoxMesh" id="mesh_partition"]
material = SubResource("m_partition")
size = Vector3(1.84, 0.5, 0.06)

[sub_resource type="BoxShape3D" id="sh_pod"]
size = Vector3(1.8, 0.75, 1.8)

[sub_resource type="BoxShape3D" id="sh_desk"]
size = Vector3(1.8, 0.75, 0.9)

[sub_resource type="BoxShape3D" id="sh_deskw"]
size = Vector3(1.8, 0.75, 0.9)

[sub_resource type="BoxShape3D" id="sh_chair"]
size = Vector3(0.5, 0.86, 0.5)

[sub_resource type="BoxShape3D" id="sh_cupboard"]
size = Vector3(0.55, 1.9, 0.93)

[sub_resource type="BoxShape3D" id="sh_cupboard_z"]
size = Vector3(0.93, 1.9, 0.55)

[sub_resource type="BoxShape3D" id="sh_table"]
size = Vector3(0.9, 0.75, 0.9)

[sub_resource type="BoxShape3D" id="sh_cooler"]
size = Vector3(0.62, 1.3, 0.58)

[sub_resource type="BoxShape3D" id="sh_sofa"]
size = Vector3(1.6, 0.74, 0.77)

[sub_resource type="BoxShape3D" id="sh_sofa_z"]
size = Vector3(0.77, 0.74, 1.6)

[sub_resource type="BoxShape3D" id="sh_ctable"]
size = Vector3(1.0, 0.42, 0.55)

[sub_resource type="BoxShape3D" id="sh_plant"]
size = Vector3(0.64, 0.95, 0.64)

[sub_resource type="BoxShape3D" id="sh_bin"]
size = Vector3(0.8, 0.62, 0.4)

[sub_resource type="BoxShape3D" id="sh_cchair"]
size = Vector3(0.42, 0.84, 0.46)

[sub_resource type="BoxShape3D" id="sh_bench"]
size = Vector3(1.6, 0.45, 0.36)

[sub_resource type="BoxShape3D" id="sh_bench_z"]
size = Vector3(0.36, 0.45, 1.6)

[node name="FacultyFurniture" type="Node3D" unique_id=%d]
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
    return "%g" % round(v, 4)


def inst(name, parent, res, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" parent="%s" unique_id=%d instance=ExtResource("%s")]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            % (name, parent, tscn.new_id(), res, rot, n(x), n(y), n(z)))


def group(name, parent="."):
    return '\n[node name="%s" type="Node3D" parent="%s" unique_id=%d]\n' % (
        name, parent, tscn.new_id())


def mesh(name, parent, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="MeshInstance3D" parent="%s" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'mesh = SubResource("%s")\n'
            % (name, parent, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def box_mesh(sid, size, material):
    return ('\n[sub_resource type="BoxMesh" id="%s"]\n'
            'material = SubResource("%s")\n'
            'size = Vector3(%s, %s, %s)\n'
            % (sid, material, n(size[0]), n(size[1]), n(size[2])))


def box_shape(sid, size):
    return ('\n[sub_resource type="BoxShape3D" id="%s"]\nsize = Vector3(%s, %s, %s)\n'
            % (sid, n(size[0]), n(size[1]), n(size[2])))


def col(name, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="CollisionShape3D" parent="Collision" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def workstation(out, cols, tag, dx, dz, facing):
    """One desk with its chair, screen, keyboard and tower.

    facing is FACE_NZ for a desk whose user sits on the -z side of it, FACE_PZ
    for one whose user sits on the +z side; the sign follows from that so the
    monitor always looks back at the chair.
    """
    s = 1.0 if facing == FACE_NZ else -1.0
    back = FACE_PZ if facing == FACE_NZ else FACE_NZ
    out.append(inst("Desk" + tag, "Workstations", "1_desk", dx, 0, dz, facing))
    out.append(inst("Monitor" + tag, "Workstations", "3_mon",
                    dx - 0.3 * s, DESK_TOP, dz + 0.26 * s, facing))
    out.append(inst("Keyboard" + tag, "Workstations", "4_kb",
                    dx - 0.3 * s, DESK_TOP, dz - 0.12 * s, facing))
    out.append(inst("Tower" + tag, "Workstations", "5_pc",
                    dx + 0.62 * s, DESK_TOP, dz + 0.2 * s, facing))
    out.append(inst("Chair" + tag, "Workstations", "2_chair",
                    dx, 0, dz - 1.05 * s, back))
    cols.append(col("CChair" + tag, "sh_chair", dx, 0.43, dz - 1.05 * s))


def build(w, d):
    cfg = ROOMS[(w, d)]
    out = []
    cols = []
    extra = []          # sub_resources sized per instance (cabin partitions)

    # --- staff pods: two desks back to back across a partition -------------- #
    out.append(group("Workstations"))
    for b, (spine, columns) in enumerate(cfg["pods"]):
        for c, dx in enumerate(columns):
            near = spine - DESK_D / 2          # user sits further from the spine
            far = spine + DESK_D / 2
            workstation(out, cols, "%d%02dA" % (b + 1, c + 1), dx, near, FACE_NZ)
            workstation(out, cols, "%d%02dB" % (b + 1, c + 1), dx, far, FACE_PZ)
            out.append(mesh("Partition%d%02d" % (b + 1, c + 1), "Workstations",
                            "mesh_partition", dx, DESK_TOP + PARTITION_H / 2, spine))
            cols.append(col("CPod%d%02d" % (b + 1, c + 1), "sh_pod", dx, 0.375, spine))

    # --- senior / HOD desk with visitor chairs ------------------------------ #
    if cfg["senior"]:
        sx, sz = cfg["senior"]
        out.append(group("Senior"))
        out.append(inst("SeniorDesk", "Senior", "16_deskw", sx, 0, sz, FACE_NZ))
        cols.append(col("CSeniorDesk", "sh_deskw", sx, 0.375, sz))
        out.append(inst("SeniorMonitor", "Senior", "3_mon",
                        sx - 0.45, 0.75, sz + 0.2, FACE_NZ))
        out.append(inst("SeniorKeyboard", "Senior", "4_kb",
                        sx - 0.45, 0.75, sz - 0.1, FACE_NZ))
        out.append(inst("SeniorTower", "Senior", "5_pc",
                        sx + 0.7, 0.75, sz + 0.25, FACE_NZ))
        out.append(inst("SeniorChair", "Senior", "2_chair", sx, 0, sz + 0.95, FACE_NZ))
        cols.append(col("CSeniorChair", "sh_chair", sx, 0.43, sz + 0.95))
        for i, vx in enumerate((sx - 0.5, sx + 0.5)):
            out.append(inst("VisitorChair%02d" % (i + 1), "Senior", "2_chair",
                            vx, 0, sz - 1.0))
            cols.append(col("CVisitorChair%02d" % (i + 1), "sh_chair", vx, 0.43, sz - 1.0))

    # --- cabin partitions round the senior desk ----------------------------- #
    if cfg.get("cabin"):
        out.append(group("Cabin"))
        for i, (x0, z0, x1, z1) in enumerate(cfg["cabin"]):
            tag = "%02d" % (i + 1)
            size = (max(abs(x1 - x0), 0.1), CABIN_H, max(abs(z1 - z0), 0.1))
            extra.append(box_mesh("mesh_cabin" + tag, size, "m_cabin"))
            extra.append(box_shape("sh_cabin" + tag, size))
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            out.append(mesh("CabinWall" + tag, "Cabin", "mesh_cabin" + tag,
                            cx, CABIN_H / 2, cz))
            cols.append(col("CCabinWall" + tag, "sh_cabin" + tag, cx, CABIN_H / 2, cz))

    # --- meeting bay: two tables pushed together, two chairs a side --------- #
    if cfg.get("meeting"):
        mx, mz = cfg["meeting"]
        out.append(group("Meeting"))
        for i, tx in enumerate((mx - 0.45, mx + 0.45)):
            out.append(inst("MeetTable%02d" % (i + 1), "Meeting", "17_table", tx, 0, mz))
            cols.append(col("CMeetTable%02d" % (i + 1), "sh_table", tx, 0.375, mz))
        for i, (chx, chz, crot) in enumerate((
                (mx - 0.45, mz - 0.75, FACE_PZ), (mx + 0.45, mz - 0.75, FACE_PZ),
                (mx - 0.45, mz + 0.75, FACE_NZ), (mx + 0.45, mz + 0.75, FACE_NZ))):
            out.append(inst("MeetChair%02d" % (i + 1), "Meeting", "21_cchair",
                            chx, 0, chz, crot))
            cols.append(col("CMeetChair%02d" % (i + 1), "sh_cchair", chx, 0.42, chz))

    # --- storage, printer table, water cooler ------------------------------- #
    out.append(group("Storage"))
    for i, (cx, cz, rot) in enumerate(cfg["almirahs"]):
        tag = "%02d" % (i + 1)
        out.append(inst("Almirah" + tag, "Storage", "6_cup", cx, 0, cz, rot))
        sub = "sh_cupboard" if rot in (FACE_NX, FACE_PX) else "sh_cupboard_z"
        cols.append(col("CAlmirah" + tag, sub, cx, 0.95, cz))
    for i, (bx, bz, brot) in enumerate(cfg.get("benches", [])):
        tag = "%02d" % (i + 1)
        out.append(inst("Bench" + tag, "Storage", "22_bench", bx, 0, bz, brot))
        cols.append(col("CBench" + tag, "sh_bench" if brot in (FACE_PZ, FACE_NZ)
                        else "sh_bench_z", bx, 0.22, bz))
    px, pz, prot = cfg["printer"]
    out.append(inst("PrinterTable", "Storage", "17_table", px, 0, pz, prot))
    cols.append(col("CPrinterTable", "sh_table", px, 0.375, pz))
    out.append(inst("Printer", "Storage", "7_prn", px, 0.75, pz, prot))
    wx, wz, wrot = cfg["cooler"]
    out.append(inst("WaterCooler", "Storage", "18_cooler", wx, 0, wz, wrot))
    cols.append(col("CWaterCooler", "sh_cooler", wx, 0.65, wz, wrot))

    # --- visitor lounge inside the door ------------------------------------- #
    out.append(group("Lounge"))
    lounge = cfg["lounge"]
    sx, sz, srot = lounge["sofa"]
    out.append(inst("Sofa", "Lounge", "8_sofa", sx, 0, sz, srot))
    cols.append(col("CSofa", "sh_sofa" if srot in (FACE_PZ, FACE_NZ) else "sh_sofa_z",
                    sx, 0.37, sz))
    if "sofa2" in lounge:
        s2x, s2z, s2rot = lounge["sofa2"]
        out.append(inst("Sofa2", "Lounge", "8_sofa", s2x, 0, s2z, s2rot))
        cols.append(col("CSofa2", "sh_sofa" if s2rot in (FACE_PZ, FACE_NZ) else "sh_sofa_z",
                        s2x, 0.37, s2z))
    tx, tz = lounge["table"]
    out.append(inst("CoffeeTable", "Lounge", "9_ctable", tx, 0, tz))
    cols.append(col("CCoffeeTable", "sh_ctable", tx, 0.21, tz))

    # --- wall fittings and clutter ------------------------------------------ #
    out.append(group("Props"))
    nx, nz, nrot = cfg["notice"]
    out.append(inst("NoticeBoard", "Props", "10_notice", nx, 1.1, nz, nrot))
    cx, cz, crot = cfg["clock"]
    out.append(inst("WallClock", "Props", "13_clock", cx, 2.4, cz, crot))
    ex, ez = cfg["exit"]
    out.append(inst("ExitSign", "Props", "20_exit", ex, 2.55, ez, FACE_NZ))
    for i, (ax, az, arot) in enumerate(cfg["acs"]):
        out.append(inst("AC%02d" % (i + 1), "Props", "19_ac", ax, 2.35, az, arot))
    for i, (bx, bz, brot) in enumerate(cfg["bins"]):
        tag = "%02d" % (i + 1)
        out.append(inst("Dustbins" + tag, "Props", "12_bin", bx, 0, bz, brot))
        cols.append(col("CDustbins" + tag, "sh_bin", bx, 0.31, bz))
    for i, (plx, plz) in enumerate(cfg["plants"]):
        tag = "%02d" % (i + 1)
        out.append(inst("Plant" + tag, "Props", "11_plant", plx, 0, plz))
        cols.append(col("CPlant" + tag, "sh_plant", plx, 0.48, plz))

    # --- lighting ----------------------------------------------------------- #
    out.append(group("Lighting"))
    for i, (fx, fz) in enumerate(cfg["fans"]):
        out.append(inst("Fan%02d" % (i + 1), "Lighting", "14_fan", fx, CEILING, fz))
    for i, (lx, lz) in enumerate(cfg["battens"]):
        tag = "%02d" % (i + 1)
        out.append(inst("Batten" + tag, "Lighting", "15_tube", lx, 2.9, lz))
        out.append(LAMP % ("Lamp" + tag, tscn.new_id(), n(lx), n(lz)))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)

    head = HEAD % tscn.new_id()
    # the generated sub_resources have to land in the header, before the nodes
    at = head.index('[node name="FacultyFurniture"')
    head = head[:at] + "".join(extra).lstrip("\n") + "\n" + head[at:]
    return head + "".join(out)


def patch_room(w, d):
    """Add the furniture instance to the Faculty_<w>x<d> shell, once."""
    path = "Scene/Rooms/Faculty_%dx%d.tscn" % (w, d)
    text = tscn.read(path)
    if "FacultyFurniture_%dx%d" % (w, d) in text:
        return "already patched"
    res_id = "r_faculty_furn"
    text = tscn.insert_after_last(
        text, "ext_resource",
        '[ext_resource type="PackedScene" '
        'path="res://Scene/Props/FacultyFurniture_%dx%d.tscn" id="%s"]\n'
        % (w, d, res_id))
    text = text.rstrip() + (
        '\n\n[node name="Furniture" parent="." instance=ExtResource("%s")]\n'
        'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)\n' % res_id)
    tscn.write(path, text)
    return "patched"


def build_all():
    report = []
    for (w, d) in sorted(ROOMS):
        props = build(w, d)
        tscn.write("Scene/Props/FacultyFurniture_%dx%d.tscn" % (w, d), props)
        report.append(("%dx%d" % (w, d), props.count("[node "), patch_room(w, d)))
    return report


if __name__ == "__main__":
    for row in build_all():
        print(row)
