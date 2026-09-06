"""Generate the seminar hall shell and its furniture, sized from W x D below.

Footprint is 24 x 24. The stage sits against the z = D wall, the audience faces
+z. All numbers are game metres; the prop kit is authored at half scale and the
.import files apply nodes/root_scale = 2.0, while the Floor/Wall kit is 1:1.

The house is raked the way a real hall is: you come in at the back at ground
floor level and the floor STEPS DOWN towards the stage, one 0.15 m step every
two rows, so the stage ends up about half a metre above the front row without
any deck being built. The 1F rooms sit directly on this hall, so its walls stay
3 m - the height at the front comes from digging the house down, not from
stacking walls.

  z = 0..4     rear landing at ground level, doors in both corners, AV control
  z = 4..15.9  raked house, 10 rows x 8 four-seat banks = 320 seats
  z = 15.9..18 front cross aisle, a flight of steps at each end up to the stage
  z = 18..24   stage, drapes and screen on the back wall, lectern stage-left,
               dais table for the dignitaries stage-right

The stage is only as deep as it needs to be: everything on it is positioned off
the back wall, so no dead floor is left stranded behind the dais.
"""

import re

import tscn

W, D = 24.0, 24.0
CELL = 4.0
CEILING = 3.0
FLOOR_TOP = 0.1             # Floor.glb slab is 0.1 thick, props sit at y = 0

# glTF Y-up export maps blender -Y onto game +Z, so kit props end up facing +Z.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"
WALL_ALONG_X = FACE_NX      # kit wall panels run along X with this basis
WALL_ALONG_Z = FACE_PZ

# --- the rake -------------------------------------------------------------- #
HOUSE_Z0 = 4.0              # top of the rake, flush with the rear landing
STAGE_Z0 = 18.0             # stage face, on the 4 m floor grid
ROW_PITCH = 1.1
ROW_N = 10
ROW_Z = [5.4 + ROW_PITCH * i for i in range(ROW_N)]
TIER_ROWS = 2               # rows per step
TIER_DROP = 0.15            # how far each step goes down
TIER_N = (ROW_N + TIER_ROWS - 1) // TIER_ROWS
TIER_D = ROW_PITCH * TIER_ROWS
HOUSE_Z1 = ROW_Z[0] - 0.55 + TIER_D * TIER_N        # bottom of the last step
FLOOR_BOTTOM = FLOOR_TOP - TIER_DROP * (TIER_N - 1)  # front cross aisle level

# --- stage steps: two flights up out of the front cross aisle -------------- #
STEP_X = (5.0, 19.0)
STEP_W = 2.4
STEP_N = 5
STEP_RUN = (STAGE_Z0 - HOUSE_Z1) / STEP_N
STEP_RISE = (FLOOR_TOP - FLOOR_BOTTOM) / STEP_N

# --- house ----------------------------------------------------------------- #
# Four 2.36 m banks per block, blocks either side of a 1.92 m centre aisle, and
# a 1.6 m side aisle against each long wall lining up with the corner doors.
BLOCK_L = 1.6
BANK_W = 2.36
SEAT_BLOCKS = tuple(BLOCK_L + BANK_W * (i + 0.5) for i in range(4)) + \
              tuple(W - BLOCK_L - BANK_W * (i + 0.5) for i in reversed(range(4)))
AISLE_C = W / 2.0
AISLE_W = W - 2 * (BLOCK_L + 4 * BANK_W)

DOOR_X = (1.0, 23.0)        # doors sit in the corners, inside the side aisles


def tier_of(i):
    return i // TIER_ROWS


def tier_top(level):
    """Walking surface of a step."""
    return FLOOR_TOP - TIER_DROP * level


def tier_range(level):
    """(z0, z1) the step covers. The first reaches back to the rear landing and
    the last runs on to the stage face, taking in the front cross aisle."""
    z0 = HOUSE_Z0 if level == 0 else ROW_Z[0] - 0.55 + TIER_D * level
    z1 = STAGE_Z0 if level == TIER_N - 1 else ROW_Z[0] - 0.55 + TIER_D * (level + 1)
    return z0, z1


PROPS_HEAD = '''[gd_scene format=3 uid="uid://cseminarprops01"]

[ext_resource type="PackedScene" path="res://Assets/seminar_seat_row4.glb" id="1_row"]
[ext_resource type="PackedScene" path="res://Assets/lab_chair.glb" id="2_chair"]
[ext_resource type="PackedScene" path="res://Assets/lectern.glb" id="3_lect"]
[ext_resource type="PackedScene" path="res://Assets/mic_gooseneck.glb" id="4_mic"]
[ext_resource type="PackedScene" path="res://Assets/pa_speaker.glb" id="5_pa"]
[ext_resource type="PackedScene" uid="uid://d4dy7pis2g3lb" path="res://Assets/desk_wood.glb" id="6_dais"]
[ext_resource type="PackedScene" path="res://Assets/stage_curtain.glb" id="7_curt"]
[ext_resource type="PackedScene" path="res://Assets/projector_screen_large.glb" id="8_scr"]
[ext_resource type="PackedScene" path="res://Assets/stage_light_bar.glb" id="9_bar"]
[ext_resource type="PackedScene" path="res://Assets/av_control_desk.glb" id="10_av"]
[ext_resource type="PackedScene" path="res://Assets/standee_banner.glb" id="11_std"]
[ext_resource type="PackedScene" path="res://Assets/server_rack.glb" id="12_rack"]
[ext_resource type="PackedScene" path="res://Assets/projector_ceiling.glb" id="13_proj"]
[ext_resource type="PackedScene" path="res://Assets/monitor_lcd.glb" id="14_mon"]
[ext_resource type="PackedScene" path="res://Assets/exit_sign.glb" id="15_exit"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="16_bin"]
[ext_resource type="PackedScene" path="res://Assets/potted_plant.glb" id="17_plant"]
[ext_resource type="PackedScene" path="res://Assets/fire_extinguisher.glb" id="18_fire"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="19_tube"]
[ext_resource type="PackedScene" path="res://Assets/ac_split_unit.glb" id="20_ac"]
[ext_resource type="PackedScene" path="res://Assets/wall_clock.glb" id="21_clock"]
[ext_resource type="PackedScene" path="res://Assets/cable_duct.glb" id="22_duct"]

[sub_resource type="StandardMaterial3D" id="m_riser"]
albedo_color = Color(0.42, 0.42, 0.44, 1)
roughness = 0.95

[sub_resource type="StandardMaterial3D" id="m_fascia"]
albedo_color = Color(0.13, 0.09, 0.07, 1)
roughness = 0.9

[sub_resource type="StandardMaterial3D" id="m_deck"]
albedo_color = Color(0.33, 0.25, 0.19, 1)
roughness = 0.9

[sub_resource type="StandardMaterial3D" id="m_carpet"]
albedo_color = Color(0.29, 0.08, 0.11, 1)
roughness = 1.0

[sub_resource type="StandardMaterial3D" id="m_panel"]
albedo_color = Color(0.27, 0.19, 0.15, 1)
roughness = 0.95

[sub_resource type="BoxShape3D" id="sh_row"]
size = Vector3(2.36, 0.93, 0.6)

[sub_resource type="BoxShape3D" id="sh_lect"]
size = Vector3(0.72, 1.16, 0.52)

[sub_resource type="BoxShape3D" id="sh_dais"]
size = Vector3(1.8, 0.75, 0.9)

[sub_resource type="BoxShape3D" id="sh_chair"]
size = Vector3(0.5, 0.86, 0.5)

[sub_resource type="BoxShape3D" id="sh_pa"]
size = Vector3(0.54, 1.56, 0.62)

[sub_resource type="BoxShape3D" id="sh_av"]
size = Vector3(1.6, 0.87, 0.7)

[sub_resource type="BoxShape3D" id="sh_rack"]
size = Vector3(0.6, 2.0, 0.83)

[sub_resource type="BoxShape3D" id="sh_std"]
size = Vector3(0.85, 1.98, 0.34)

[sub_resource type="BoxShape3D" id="sh_plant"]
size = Vector3(0.64, 0.95, 0.64)

[node name="SeminarFurniture" type="Node3D" unique_id=%(root)d]
'''

LAMP = """
[node name="%s" type="OmniLight3D" parent="Lighting" unique_id=%d]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %s, 2.7, %s)
light_color = Color(1, 0.95, 0.88, 1)
light_energy = 1.8
light_volumetric_fog_energy = 1.1
light_specular = 0.2
light_bake_mode = 1
shadow_bias = 0.04
shadow_normal_bias = 1.5
distance_fade_enabled = true
distance_fade_begin = 20.0
distance_fade_shadow = 12.0
distance_fade_length = 6.0
omni_range = 10.0
omni_attenuation = 1.4
"""

# Stage wash: aimed down and forward from the light bars onto the deck.
SPOT = """
[node name="%s" type="SpotLight3D" parent="Lighting" unique_id=%d]
transform = Transform3D(1, 0, 0, 0, -0.642788, 0.766044, 0, -0.766044, -0.642788, %s, 2.95, %s)
light_color = Color(1, 0.93, 0.82, 1)
light_energy = 3.0
light_volumetric_fog_energy = 1.2
light_specular = 0.4
light_bake_mode = 1
shadow_enabled = true
shadow_bias = 0.04
shadow_normal_bias = 1.5
distance_fade_enabled = true
distance_fade_begin = 24.0
distance_fade_shadow = 16.0
distance_fade_length = 6.0
spot_range = 12.0
spot_attenuation = 1.2
spot_angle = 38.0
spot_angle_attenuation = 0.6
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


def col(name, parent, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="CollisionShape3D" parent="%s" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, parent, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def col_raw(name, sub, basis, x, y, z):
    """Collision shape on an arbitrary basis - used for the walking ramps."""
    return ('\n[node name="%s" type="CollisionShape3D" parent="Collision" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, tscn.new_id(), basis, n(x), n(y), n(z), sub))


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


def slope_basis(run, rise):
    """Basis whose local +Z climbs `rise` over `run` and whose local +Y is the
    slope normal, so a box on it presents an inclined top face."""
    length = (run * run + rise * rise) ** 0.5
    c, s = run / length, rise / length
    # .tscn writes a Transform3D basis row by row, so this is the transpose of
    # the column form: local +Z climbs the slope, local +Y is its normal.
    return ("1, 0, 0, 0, %s, %s, 0, %s, %s" % (n(c), n(s), n(-s), n(c)), length, c, s)


# --------------------------------------------------------------------------- #
def build_props():
    extra = []          # sub_resources sized per instance
    out = []
    cols = []

    # --- raked floor: one slab per step, thick enough to seal underneath ---- #
    out.append(group("Rake"))
    SLAB_T = 1.8
    for L in range(TIER_N):
        z0, z1 = tier_range(L)
        top = tier_top(L)
        sid = "mesh_tier%02d" % L
        extra.append(box_mesh(sid, (W, SLAB_T, z1 - z0), "m_riser"))
        out.append(mesh("Tier%02d" % L, "Rake", sid, W / 2, top - SLAB_T / 2,
                        (z0 + z1) / 2))
    # skirts closing the gap between the sunk floor and the bottom of the walls
    skirt_d = STAGE_Z0 - HOUSE_Z0
    extra.append(box_mesh("mesh_skirt", (0.12, 1.4, skirt_d), "m_riser"))
    extra.append(box_shape("sh_skirt", (0.12, 1.4, skirt_d)))
    for i, sx in enumerate((0.06, W - 0.06)):
        out.append(mesh("Skirt%02d" % (i + 1), "Rake", "mesh_skirt",
                        sx, FLOOR_TOP - 0.7, HOUSE_Z0 + skirt_d / 2))
        cols.append(col("CSkirt%02d" % (i + 1), "Collision", "sh_skirt",
                        sx, FLOOR_TOP - 0.7, HOUSE_Z0 + skirt_d / 2))
    # The hall lays its own ground slab: the level strips every floor tile in the
    # footprint, because the rake cuts below them and a 4 m tile grid cannot
    # follow the stage face. Landing at the back, boarded stage at the front.
    extra.append(box_mesh("mesh_landing", (W, FLOOR_TOP, HOUSE_Z0), "m_riser"))
    out.append(mesh("Landing", "Rake", "mesh_landing", W / 2, FLOOR_TOP / 2,
                    HOUSE_Z0 / 2))
    extra.append(box_mesh("mesh_deck", (W, FLOOR_TOP + 0.04, D - STAGE_Z0), "m_deck"))
    out.append(mesh("StageDeck", "Rake", "mesh_deck", W / 2, (FLOOR_TOP + 0.04) / 2,
                    (STAGE_Z0 + D) / 2))

    # stage face: seals the void under the stage floor and reads as its fascia
    face_h = FLOOR_TOP - FLOOR_BOTTOM
    extra.append(box_mesh("mesh_face", (W, face_h, 0.1), "m_fascia"))
    extra.append(box_shape("sh_face", (W, face_h, 0.2)))
    out.append(mesh("StageFace", "Rake", "mesh_face", W / 2,
                    FLOOR_TOP + 0.04 - face_h / 2, STAGE_Z0 + 0.05))
    cols.append(col("CStageFace", "Collision", "sh_face", W / 2,
                    FLOOR_TOP - face_h / 2, STAGE_Z0 + 0.1))

    # walking collision: one long ramp down the rake, flat at the bottom. The
    # 3.3 deg slope reads as the steps do but never trips the character body.
    basis, length, rc, rs = slope_basis(HOUSE_Z1 - HOUSE_Z0, FLOOR_BOTTOM - FLOOR_TOP)
    extra.append(box_shape("sh_rake", (W, 0.6, length)))
    mid_y, mid_z = (FLOOR_TOP + FLOOR_BOTTOM) / 2, (HOUSE_Z0 + HOUSE_Z1) / 2
    cols.append(col_raw("CRake", "sh_rake", basis, W / 2,
                        mid_y - 0.3 * rc, mid_z + 0.3 * rs))
    extra.append(box_shape("sh_pit", (W, 0.2, STAGE_Z0 - HOUSE_Z1)))
    cols.append(col("CPit", "Collision", "sh_pit", W / 2, FLOOR_BOTTOM - 0.1,
                    (HOUSE_Z1 + STAGE_Z0) / 2))
    # GFLevel instances this scene on its own, without the room shell, and its
    # bay has no floor collision of its own - so carry the landing and the stage
    # here too. Standalone the room's own boxes just overlap these.
    extra.append(box_shape("sh_landing", (W, 0.2, HOUSE_Z0)))
    cols.append(col("CLanding", "Collision", "sh_landing", W / 2, 0, HOUSE_Z0 / 2))
    extra.append(box_shape("sh_stagefloor", (W, 0.2, D - STAGE_Z0)))
    cols.append(col("CStageFloor", "Collision", "sh_stagefloor", W / 2, 0,
                    (STAGE_Z0 + D) / 2))

    # --- two flights up out of the pit onto the stage ---------------------- #
    out.append(group("Steps", "Rake"))
    for f, sx in enumerate(STEP_X):
        for k in range(STEP_N):
            h = STEP_RISE * (k + 1)
            sid = "mesh_step%d_%d" % (f + 1, k + 1)
            extra.append(box_mesh(sid, (STEP_W, h, STEP_RUN), "m_riser"))
            out.append(mesh("Step%d_%02d" % (f + 1, k + 1), "Rake/Steps", sid,
                            sx, FLOOR_BOTTOM + h / 2,
                            HOUSE_Z1 + STEP_RUN * (k + 0.5)))
        basis, length, cz, sz = slope_basis(STAGE_Z0 - HOUSE_Z1,
                                            FLOOR_TOP - FLOOR_BOTTOM)
        sid = "sh_ramp%d" % (f + 1)
        extra.append(box_shape(sid, (STEP_W, 0.6, length)))
        cols.append(col_raw("CRamp%d" % (f + 1), sid, basis, sx,
                            (FLOOR_TOP + FLOOR_BOTTOM) / 2 - 0.3 * cz,
                            (HOUSE_Z1 + STAGE_Z0) / 2 + 0.3 * sz))

    # --- drapes: four 6.12 m widths across the back plus a leg each side --- #
    out.append(group("Stage"))
    for i, cx in enumerate((3.0, 9.0, 15.0, 21.0)):
        out.append(inst("Curtain%02d" % (i + 1), "Stage", "7_curt", cx, 0, D - 0.15, FACE_NZ))
    for i, (lx, rot) in enumerate(((2.1, FACE_NX), (21.9, FACE_PX))):
        out.append(inst("Leg%02d" % (i + 1), "Stage", "7_curt", lx, 0, (STAGE_Z0 + D) / 2, rot))
    out.append(inst("Screen", "Stage", "8_scr", W / 2, 0, D - 0.5, FACE_NZ))

    # --- speaker's position, stage-left ------------------------------------ #
    out.append(inst("Lectern", "Stage", "3_lect", 6.5, 0, D - 3.4, FACE_NZ))
    cols.append(col("CLectern", "Collision", "sh_lect", 6.5, 0.58, D - 3.4))
    out.append(inst("LecternMic", "Stage", "4_mic", 6.5, 1.16, D - 3.55, FACE_NZ))

    # --- dais table for the dignitaries, stage-right ----------------------- #
    out.append(group("Dais", "Stage"))
    for i, tx in enumerate((14.0, 15.8, 17.6)):
        out.append(inst("DaisTable%02d" % (i + 1), "Stage/Dais", "6_dais",
                        tx, 0, D - 2.6, FACE_NZ))
        cols.append(col("CDaisTable%02d" % (i + 1), "Collision", "sh_dais",
                        tx, 0.375, D - 2.6))
        out.append(inst("DaisMic%02d" % (i + 1), "Stage/Dais", "4_mic",
                        tx + 0.3, 0.75, D - 2.85, FACE_NZ))
    extra.append(box_mesh("mesh_dais_skirt", (5.5, 0.72, 0.05), "m_carpet"))
    out.append(mesh("DaisSkirt", "Stage/Dais", "mesh_dais_skirt", 15.8, 0.36, D - 3.07))
    for i, cx in enumerate((13.5, 14.8, 16.1, 17.4, 18.7)):
        out.append(inst("DaisChair%02d" % (i + 1), "Stage/Dais", "2_chair",
                        cx, 0, D - 1.7, FACE_NZ))
        cols.append(col("CDaisChair%02d" % (i + 1), "Collision", "sh_chair",
                        cx, 0.43, D - 1.7))

    # --- PA stacks, banners and greenery on the stage ---------------------- #
    for i, px in enumerate((2.6, 21.4)):
        out.append(inst("PASpeaker%02d" % (i + 1), "Stage", "5_pa", px, 0, STAGE_Z0 + 1.4, FACE_NZ))
        cols.append(col("CPA%02d" % (i + 1), "Collision", "sh_pa", px, 0.78, STAGE_Z0 + 1.4))
    for i, bx in enumerate((3.4, 20.6)):
        out.append(inst("Standee%02d" % (i + 1), "Stage", "11_std", bx, 0, STAGE_Z0 + 0.8, FACE_NZ))
        cols.append(col("CStandee%02d" % (i + 1), "Collision", "sh_std", bx, 0.99, STAGE_Z0 + 0.8))
    for i, px in enumerate((1.2, 22.8)):
        out.append(inst("StagePlant%02d" % (i + 1), "Stage", "17_plant", px, 0, STAGE_Z0 + 1.0))
        cols.append(col("CStagePlant%02d" % (i + 1), "Collision", "sh_plant",
                        px, 0.48, STAGE_Z0 + 1.0))

    # --- audience seating, each row sitting on its own step ----------------- #
    out.append(group("Seating"))
    for r, sz in enumerate(ROW_Z):
        sy = tier_top(tier_of(r)) - FLOOR_TOP
        for c, sx in enumerate(SEAT_BLOCKS):
            tag = "R%02dB%d" % (r + 1, c + 1)
            out.append(inst("Row" + tag, "Seating", "1_row", sx, sy, sz))
            cols.append(col("C" + tag, "Collision", "sh_row", sx, sy + 0.465, sz))

    # --- aisle carpet, laid step by step ------------------------------------ #
    out.append(group("Carpet"))
    aisles = (("Centre", AISLE_C, AISLE_W), ("SideW", BLOCK_L / 2, BLOCK_L),
              ("SideE", W - BLOCK_L / 2, BLOCK_L))
    runs = []
    for name, cx, cw in aisles:
        runs.append(("%sRear" % name, cx, cw, HOUSE_Z0 / 2, HOUSE_Z0))
        for L in range(TIER_N):
            z0, z1 = tier_range(L)
            runs.append(("%s%02d" % (name, L), cx, cw, (z0 + z1) / 2, z1 - z0,
                         tier_top(L)))
    for run in runs:
        name, cx, cw, cz, cd = run[:5]
        top = run[5] if len(run) > 5 else FLOOR_TOP
        sid = "mesh_carpet_%s" % name.lower()
        extra.append(box_mesh(sid, (cw, 0.03, cd), "m_carpet"))
        out.append(mesh("Carpet" + name, "Carpet", sid, cx, top + 0.015, cz))

    # --- control position on the rear landing ------------------------------- #
    out.append(group("Control"))
    out.append(inst("AVDesk", "Control", "10_av", 12.0, 0, 2.0, FACE_NZ))
    cols.append(col("CAVDesk", "Collision", "sh_av", 12.0, 0.435, 2.0))
    for i, dx in enumerate((-0.42, 0.42)):
        out.append(inst("AVMonitor%02d" % (i + 1), "Control", "14_mon",
                        12.0 + dx, 0.87, 2.16, FACE_NZ))
    out.append(inst("AVChair", "Control", "2_chair", 12.0, 0, 1.1))
    cols.append(col("CAVChair", "Collision", "sh_chair", 12.0, 0.43, 1.1))
    out.append(inst("AmpRack", "Control", "12_rack", 9.6, 0, 1.0, FACE_PZ))
    cols.append(col("CAmpRack", "Collision", "sh_rack", 9.6, 1.0, 1.0))
    out.append(inst("CableDuct", "Control", "22_duct", 12.0, 0.02, 3.4))
    out.append(inst("Projector", "Control", "13_proj", W / 2, CEILING, HOUSE_Z1 * 0.55))

    # --- house fittings ----------------------------------------------------- #
    out.append(group("HouseProps"))
    for i, ex in enumerate(DOOR_X):
        out.append(inst("ExitSign%02d" % (i + 1), "HouseProps", "15_exit",
                        ex, 2.55, 0.22))
    for i, (fx, rot) in enumerate(((0.25, FACE_PX), (W - 0.25, FACE_NX))):
        out.append(inst("FireExtinguisher%02d" % (i + 1), "HouseProps", "18_fire",
                        fx, 1.0, 3.0, rot))
    for i, bx in enumerate((2.6, 21.4)):
        out.append(inst("Dustbins%02d" % (i + 1), "HouseProps", "16_bin", bx, 0, 0.5))
    out.append(inst("WallClock", "HouseProps", "21_clock", 12.0, 2.5, 0.15))
    for i, (px, pz) in enumerate(((0.8, 0.7), (23.2, 0.7))):
        out.append(inst("Plant%02d" % (i + 1), "HouseProps", "17_plant", px, 0, pz))
        cols.append(col("CPlant%02d" % (i + 1), "Collision", "sh_plant", px, 0.48, pz))
    # split ACs high on the long walls, clear of the window band at z = 4..16
    for i, (ax, rot) in enumerate(((0.2, FACE_PX), (W - 0.2, FACE_NX))):
        for j, az in enumerate((2.5, 9.0, 15.0)):
            out.append(inst("AC%d_%d" % (i + 1, j + 1), "HouseProps", "20_ac",
                            ax, 2.35, az, rot))

    # --- full height wall panelling on the solid stretches ------------------ #
    out.append(group("Panels"))
    # side walls: floor to ceiling, carried down past the rake so no gap shows
    extra.append(box_mesh("mesh_panel_side", (0.1, 4.2, 3.6), "m_panel"))
    extra.append(box_mesh("mesh_panel_rear", (3.6, 2.86, 0.1), "m_panel"))
    for i, px in enumerate((0.15, W - 0.15)):
        for j, pz in enumerate((2.0, 12.0, 16.0)):
            out.append(mesh("PanelSide%d_%d" % (i + 1, j + 1), "Panels",
                            "mesh_panel_side", px, CEILING - 2.16, pz))
    for i, px in enumerate((6.0, 18.0)):
        out.append(mesh("PanelRear%02d" % (i + 1), "Panels", "mesh_panel_rear",
                        px, 1.43, 0.15))

    # --- lighting: house battens plus three bars washing the stage ---------- #
    out.append(group("Lighting"))
    for i, bx in enumerate((6.0, 12.0, 18.0)):
        out.append(inst("StageBar%02d" % (i + 1), "Lighting", "9_bar", bx, CEILING, STAGE_Z0 + 0.8))
        out.append(SPOT % ("StageSpot%02d" % (i + 1), tscn.new_id(), n(bx), n(STAGE_Z0 + 0.8)))
    j = 0
    # over the house three lines of battens; over the stage only two, kept off
    # the drape legs at x = 2 and x = 22 so they do not wash them out
    for lz, xs in ((3.0, (4.0, 12.0, 20.0)), (7.0, (4.0, 12.0, 20.0)),
                   (11.0, (4.0, 12.0, 20.0)), (15.0, (4.0, 12.0, 20.0)),
                   (20.0, (8.0, 16.0)), (23.0, (8.0, 16.0))):
        for lx in xs:
            j += 1
            out.append(inst("Batten%02d" % j, "Lighting", "19_tube", lx, 2.9, lz))
            out.append(LAMP % ("Lamp%02d" % j, tscn.new_id(), n(lx), n(lz)))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)

    head = PROPS_HEAD % {"root": tscn.new_id()}
    # the generated sub_resources have to land in the header, before the nodes
    at = head.index('[node name="SeminarFurniture"')
    head = head[:at] + "".join(extra).lstrip("\n") + "\n" + head[at:]
    return head + "".join(out)


# --------------------------------------------------------------------------- #
ROOM_HEAD = '''[gd_scene format=3 uid="uid://cseminarhall01"]

[ext_resource type="PackedScene" uid="uid://cgrpb6wire5c3" path="res://Assets/Floor.glb" id="1_floor"]
[ext_resource type="PackedScene" uid="uid://c6hj4j5vvbxcy" path="res://Assets/Wall_4m.glb" id="2_w4"]
[ext_resource type="PackedScene" uid="uid://b88wvq7av8osy" path="res://Assets/Window_Wall.glb" id="3_win"]
[ext_resource type="PackedScene" uid="uid://b04icmqe3k3eu" path="res://Assets/Door_Wall.glb" id="4_door"]
[ext_resource type="PackedScene" uid="uid://c46nkyfudvx" path="res://Assets/Wall_2m.glb" id="6_w2"]
[ext_resource type="PackedScene" path="res://Scene/Props/SeminarFurniture_%(w)dx%(d)d.tscn" id="5_furn"]
[ext_resource type="PackedScene" path="res://Assets/ceiling_tile.glb" id="7_ceil"]

[sub_resource type="BoxShape3D" id="sh_floor_rear"]
size = Vector3(%(w)d, 0.2, 4)

[sub_resource type="BoxShape3D" id="sh_floor_stage"]
size = Vector3(%(w)d, 0.2, 8)

[sub_resource type="BoxShape3D" id="sh_wallx"]
size = Vector3(4, 3, 0.2)

[sub_resource type="BoxShape3D" id="sh_wallz"]
size = Vector3(0.2, 3, 4)

[sub_resource type="BoxShape3D" id="sh_wallx2"]
size = Vector3(2, 3, 0.2)

[node name="SeminarHall_%(w)dx%(d)d" type="Node3D" unique_id=%(root)d]
'''


def build_room():
    out = [ROOM_HEAD % {"root": tscn.new_id(), "w": int(W), "d": int(D)}]
    cols = []

    # Floor tiles only where the ground level survives: the rear landing and the
    # stage. The house between them is the stepped slab built with the furniture.
    out.append(group("Floors"))
    k = 0
    for gx in range(int(CELL / 2), int(W), int(CELL)):
        for gz in range(int(CELL / 2), int(D), int(CELL)):
            if HOUSE_Z0 <= gz - CELL / 2 and gz + CELL / 2 <= STAGE_Z0:
                continue
            k += 1
            out.append(inst("Floor%02d" % k, "Floors", "1_floor", gx, 0, gz))
    cols.append(col("CFloorRear", "Collision", "sh_floor_rear", W / 2, 0, HOUSE_Z0 / 2))
    cols.append(col("CFloorStage", "Collision", "sh_floor_stage", W / 2, 0,
                    (STAGE_Z0 + D) / 2))

    out.append(group("Walls"))
    # z = 0 (back of house). Door_Wall is only 2 m wide, so the two corner bays
    # are made of a door leaf plus a 2 m infill; everything else is a 4 m panel.
    corner = {1.0: "4_door", 3.0: "6_w2", 21.0: "6_w2", 23.0: "4_door"}
    for gx in sorted(corner):
        out.append(inst("WallS%s" % n(gx), "Walls", corner[gx], gx, 0, 0, WALL_ALONG_X))
        cols.append(col("CWallS%s" % n(gx), "Collision", "sh_wallx2", gx, 1.5, 0))
    for i, gx in enumerate((6, 10, 14, 18)):
        out.append(inst("WallS%02d" % (i + 1), "Walls", "2_w4", gx, 0, 0, WALL_ALONG_X))
        cols.append(col("CWallS%02d" % (i + 1), "Collision", "sh_wallx", gx, 1.5, 0))
    # z = 32 (behind the stage): solid, the drapes hang against it
    for i, gx in enumerate(range(int(CELL / 2), int(W), int(CELL))):
        out.append(inst("WallN%02d" % (i + 1), "Walls", "2_w4", gx, 0, D, WALL_ALONG_X))
        cols.append(col("CWallN%02d" % (i + 1), "Collision", "sh_wallx", gx, 1.5, D))
    # x = 0 and x = 24: a window band over the middle of the house only, so the
    # stage end and the projection screen stay in shade
    for side, gx in (("W", 0.0), ("E", W)):
        for i, gz in enumerate(range(int(CELL / 2), int(D), int(CELL))):
            res = "3_win" if gz in (6, 10) else "2_w4"
            out.append(inst("Wall%s%02d" % (side, i + 1), "Walls", res,
                            gx, 0, gz, WALL_ALONG_Z))
            cols.append(col("CWall%s%02d" % (side, i + 1), "Collision", "sh_wallz",
                            gx, 1.5, gz))

    # the ceiling is laid here rather than by gen_ceilings: that derives tiles
    # from the floor tiles, and the house floor is the stepped slab instead
    out.append(group("Ceiling"))
    k = 0
    for gx in range(int(CELL / 2), int(W), int(CELL)):
        for gz in range(int(CELL / 2), int(D), int(CELL)):
            k += 1
            out.append(inst("CeilTile%02d" % k, "Ceiling", "7_ceil", gx, CEILING, gz))

    out.append(inst("Furniture", ".", "5_furn", 0, 0, 0))
    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)
    return "".join(out)


# --------------------------------------------------------------------------- #
LEVEL_PATH = "Scene/GFLevel.tscn"
LEVEL_ORIGIN = (0.0, 40.0)      # where GFLevel instances the hall furniture

FLOOR_TILE = (4.0, 0.2, 4.0)    # the collision box GFLevel gives each floor tile


def patch_level():
    """Cut the level slab out from under the raked house.

    GFLevel instances only the furniture scene - the bay's floor, walls and
    ceiling belong to the level itself. That slab sits at y = 0 straight over the
    sunken house, so without this the rake is buried and the hall reads as one
    big flat empty room. Only the house strip goes: the rear landing and the
    stage still stand on the level's own floor.

    Idempotent, and it leaves the ceiling and everything else alone.
    """
    ox, oz = LEVEL_ORIGIN
    z0, z1 = oz, oz + D
    text = tscn.read(LEVEL_PATH)
    header, blocks = tscn.split_blocks(text)
    paths = tscn.ext_paths(header)
    sizes = dict(re.findall(
        r'\[sub_resource type="BoxShape3D" id="([^"]+)"\]\s*\nsize = Vector3\(([^)]*)\)',
        header))

    def in_strip(t):
        return (ox <= t[9] <= ox + W and z0 <= t[11] <= z1 and abs(t[10]) < 0.5)

    want = "res://Scene/Props/SeminarFurniture_%dx%d.tscn" % (W, D)
    header = re.sub(r'res://Scene/Props/SeminarFurniture_\d+x\d+\.tscn', want, header)

    keep, dropped = [], {"tiles": 0, "collision": 0}
    for b in blocks:
        t3 = tscn.block_transform(b)
        ext = tscn.block_ext(b)
        if (tscn.block_parent(b) == "." and ext
                and paths.get(ext, "").endswith("/Floor.glb") and in_strip(t3)):
            dropped["tiles"] += 1
            continue
        if 'type="CollisionShape3D"' in b and in_strip(t3):
            m = re.search(r'shape = SubResource\("([^"]+)"\)', b)
            size = sizes.get(m.group(1), "") if m else ""
            dims = [float(v) for v in size.split(",")] if size.count(",") == 2 else None
            if dims and all(abs(a - b2) < 0.01 for a, b2 in zip(dims, FLOOR_TILE)):
                dropped["collision"] += 1
                continue
        keep.append(b)

    new_text = header + "".join(keep)
    if new_text != text:
        tscn.write(LEVEL_PATH, new_text)
    return dropped


if __name__ == "__main__":
    p = build_props()
    tscn.write("Scene/Props/SeminarFurniture_%dx%d.tscn" % (W, D), p)
    r = build_room()
    tscn.write("Scene/Rooms/SeminarHall_%dx%d.tscn" % (W, D), r)
    print("props nodes:", p.count("[node "), "room nodes:", r.count("[node "))
    print("level bay stripped:", patch_level())
    print("seats:", ROW_N * len(SEAT_BLOCKS) * 4,
          "| rake %s -> %s over %s steps" % (n(FLOOR_TOP), n(FLOOR_BOTTOM), TIER_N))
