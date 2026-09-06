"""Generate Scene/Rooms/LunchHall_24x48.tscn - a working cafeteria.

Footprint is 24 x 48, dropped into the 1F west hall by build_school.py. The
service line runs across the far (z = 48) end, dining fills the rest. All numbers
are game metres; the kit is authored at half scale and the .import files apply
nodes/root_scale = 2.0.
"""

import tscn

W, D = 24.0, 48.0
CEILING = 3.0
COUNTER_TOP = 1.03      # serving / billing counter working height

# glTF Y-up export maps blender -Y onto game +Z, so kit props end up facing +Z.
FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"

TABLE_COLS = (3.0, 7.5, 12.0, 16.5, 21.0)
TABLE_ROWS = [3.5 + 3.4 * i for i in range(11)]
CHAIR_OFF = 0.85

HEAD = '''[gd_scene format=3 uid="uid://cgflunchhall01"]

[ext_resource type="PackedScene" path="res://Assets/canteen_table_4.glb" id="1_table"]
[ext_resource type="PackedScene" path="res://Assets/canteen_chair.glb" id="2_chair"]
[ext_resource type="PackedScene" path="res://Assets/serving_counter.glb" id="3_serv"]
[ext_resource type="PackedScene" path="res://Assets/steam_table.glb" id="4_steam"]
[ext_resource type="PackedScene" path="res://Assets/billing_counter.glb" id="5_bill"]
[ext_resource type="PackedScene" path="res://Assets/beverage_cooler.glb" id="6_cool"]
[ext_resource type="PackedScene" path="res://Assets/water_cooler_ro.glb" id="7_water"]
[ext_resource type="PackedScene" path="res://Assets/handwash_sink.glb" id="8_sink"]
[ext_resource type="PackedScene" path="res://Assets/menu_board_wall.glb" id="9_menu"]
[ext_resource type="PackedScene" path="res://Assets/tray_stack.glb" id="10_tray"]
[ext_resource type="PackedScene" path="res://Assets/snack_rack.glb" id="11_snack"]
[ext_resource type="PackedScene" path="res://Assets/coffee_urn.glb" id="12_urn"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="13_bin"]
[ext_resource type="PackedScene" path="res://Assets/potted_plant.glb" id="14_plant"]
[ext_resource type="PackedScene" path="res://Assets/exit_sign.glb" id="15_exit"]
[ext_resource type="PackedScene" path="res://Assets/wall_clock.glb" id="16_clock"]
[ext_resource type="PackedScene" path="res://Assets/ceiling_fan.glb" id="17_fan"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="18_tube"]

[sub_resource type="BoxShape3D" id="sh_floor"]
size = Vector3(24, 0.2, 48)

[sub_resource type="BoxShape3D" id="sh_wallx"]
size = Vector3(24, 3, 0.2)

[sub_resource type="BoxShape3D" id="sh_wallz"]
size = Vector3(0.2, 3, 48)

[sub_resource type="BoxShape3D" id="sh_table"]
size = Vector3(0.9, 0.75, 0.9)

[sub_resource type="BoxShape3D" id="sh_chair"]
size = Vector3(0.45, 0.85, 0.5)

[sub_resource type="BoxShape3D" id="sh_serv"]
size = Vector3(2.48, 1.1, 0.9)

[sub_resource type="BoxShape3D" id="sh_steam"]
size = Vector3(1.68, 0.88, 0.76)

[sub_resource type="BoxShape3D" id="sh_bill"]
size = Vector3(1.28, 1.1, 0.76)

[sub_resource type="BoxShape3D" id="sh_cooler"]
size = Vector3(0.7, 1.9, 0.73)

[sub_resource type="BoxShape3D" id="sh_water"]
size = Vector3(0.62, 1.3, 0.58)

[sub_resource type="BoxShape3D" id="sh_sink"]
size = Vector3(1.8, 1.2, 0.5)

[sub_resource type="BoxShape3D" id="sh_snack"]
size = Vector3(1.0, 1.64, 0.4)

[sub_resource type="BoxShape3D" id="sh_bin"]
size = Vector3(0.4, 0.62, 0.81)

[sub_resource type="BoxShape3D" id="sh_plant"]
size = Vector3(0.64, 0.95, 0.64)

[node name="LunchHall_24x48" type="Node3D" unique_id=%d]
'''

LAMP = """
[node name="%s" type="OmniLight3D" parent="Lighting" unique_id=%d]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %s, 2.7, %s)
light_color = Color(1, 0.96, 0.9, 1)
light_energy = 1.7
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

    # ------------------------------------------------------------------ #
    # Service line across the far end. Diners queue on the -z side, so the
    # counters face -z and the kitchen equipment sits behind them.
    # ------------------------------------------------------------------ #
    out.append(group("ServiceLine"))
    for i, sx in enumerate((4.5, 7.1, 9.7)):
        tag = "%02d" % (i + 1)
        out.append(inst("Counter" + tag, "ServiceLine", "3_serv", sx, 0, 44.5, FACE_NZ))
        cols.append(col("CCounter" + tag, "sh_serv", sx, 0.55, 44.5))
    for i, sx in enumerate((5.5, 8.5)):
        tag = "%02d" % (i + 1)
        out.append(inst("SteamTable" + tag, "ServiceLine", "4_steam", sx, 0, 46.2, FACE_NZ))
        cols.append(col("CSteam" + tag, "sh_steam", sx, 0.44, 46.2))
    out.append(inst("TrayStack", "ServiceLine", "10_tray", 3.4, COUNTER_TOP, 44.4, FACE_NZ))
    out.append(inst("CoffeeUrn", "ServiceLine", "12_urn", 10.6, COUNTER_TOP, 44.6, FACE_NZ))

    out.append(inst("BillingCounter", "ServiceLine", "5_bill", 13.5, 0, 44.5, FACE_NZ))
    cols.append(col("CBilling", "sh_bill", 13.5, 0.55, 44.5))

    for i, sx in enumerate((17.0, 19.2)):
        tag = "%02d" % (i + 1)
        out.append(inst("SnackRack" + tag, "ServiceLine", "11_snack", sx, 0, 47.6, FACE_NZ))
        cols.append(col("CSnack" + tag, "sh_snack", sx, 0.82, 47.6))
    for i, sz in enumerate((43.6, 45.2)):
        tag = "%02d" % (i + 1)
        out.append(inst("Cooler" + tag, "ServiceLine", "6_cool", 23.4, 0, sz, FACE_NX))
        cols.append(col("CCooler" + tag, "sh_cooler", 23.4, 0.95, sz, FACE_NX))
    out.append(inst("MenuBoard", "ServiceLine", "9_menu", 7.0, 1.8, 47.85, FACE_NZ))

    # ------------------------------------------------------------------ #
    # Washing / drinking, down the side walls
    # ------------------------------------------------------------------ #
    out.append(group("Utilities"))
    for i, sz in enumerate((38.0, 40.0)):
        tag = "%02d" % (i + 1)
        out.append(inst("Sink" + tag, "Utilities", "8_sink", 0.35, 0, sz, FACE_PX))
        cols.append(col("CSink" + tag, "sh_sink", 0.35, 0.6, sz, FACE_PX))
    for i, sz in enumerate((30.0, 32.0)):
        tag = "%02d" % (i + 1)
        out.append(inst("WaterCooler" + tag, "Utilities", "7_water", 23.7, 0, sz, FACE_NX))
        cols.append(col("CWater" + tag, "sh_water", 23.7, 0.65, sz, FACE_NX))

    # ------------------------------------------------------------------ #
    # Dining: 55 four-tops, chairs on all four sides facing in
    # ------------------------------------------------------------------ #
    out.append(group("Dining"))
    for r, tz in enumerate(TABLE_ROWS):
        for c, tx in enumerate(TABLE_COLS):
            tag = "R%02dC%d" % (r + 1, c + 1)
            out.append(inst("Table" + tag, "Dining", "1_table", tx, 0, tz))
            cols.append(col("CTable" + tag, "sh_table", tx, 0.375, tz))
            seats = (("N", tx, tz - CHAIR_OFF, FACE_PZ),
                     ("S", tx, tz + CHAIR_OFF, FACE_NZ),
                     ("W", tx - CHAIR_OFF, tz, FACE_PX),
                     ("E", tx + CHAIR_OFF, tz, FACE_NX))
            for side, cx, cz, rot in seats:
                out.append(inst("Chair%s%s" % (tag, side), "Dining", "2_chair",
                                cx, 0, cz, rot))
                cols.append(col("CChair%s%s" % (tag, side), "sh_chair",
                                cx, 0.425, cz, rot))

    # ------------------------------------------------------------------ #
    out.append(group("HouseProps"))
    for i, (bx, bz) in enumerate(((23.3, 41.0), (0.7, 44.0))):
        tag = "%02d" % (i + 1)
        rot = FACE_NX if bx > 12 else FACE_PX
        out.append(inst("Dustbins" + tag, "HouseProps", "13_bin", bx, 0, bz, rot))
        cols.append(col("CDustbins" + tag, "sh_bin", bx, 0.31, bz, rot))
    for i, (px, pz) in enumerate(((1.0, 1.5), (23.0, 1.5), (1.0, 36.0), (23.0, 36.0))):
        tag = "%02d" % (i + 1)
        out.append(inst("Plant" + tag, "HouseProps", "14_plant", px, 0, pz))
        cols.append(col("CPlant" + tag, "sh_plant", px, 0.48, pz))
    for i, ex in enumerate((3.0, 21.0)):
        out.append(inst("ExitSign%02d" % (i + 1), "HouseProps", "15_exit", ex, 2.6, 0.15))
    out.append(inst("WallClock", "HouseProps", "16_clock", 12.0, 2.4, 0.15))

    # ------------------------------------------------------------------ #
    out.append(group("Lighting"))
    j = 0
    for lz in range(4, int(D), 8):
        for lx in (4.0, 12.0, 20.0):
            j += 1
            out.append(inst("Batten%02d" % j, "Lighting", "18_tube", lx, 2.9, lz))
            out.append(LAMP % ("Lamp%02d" % j, tscn.new_id(), n(lx), n(lz)))
    f = 0
    for fz in (8.0, 16.0, 24.0, 32.0):
        for fx in (7.0, 17.0):
            f += 1
            out.append(inst("Fan%02d" % f, "Lighting", "17_fan", fx, CEILING, fz))

    # 1FLevel lays the hall's floor tiles but gives them no collision at all, so
    # the slab and its perimeter are carried here - without this you drop
    # straight through the cafeteria floor.
    cols.insert(0, col("CFloor", "sh_floor", W / 2, 0, D / 2))
    for i, wz in enumerate((0.0, D)):
        cols.append(col("CWallX%02d" % (i + 1), "sh_wallx", W / 2, 1.5, wz))
    for i, wx in enumerate((0.0, W)):
        cols.append(col("CWallZ%02d" % (i + 1), "sh_wallz", wx, 1.5, D / 2))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)
    return "".join(out)


if __name__ == "__main__":
    text = build()
    tscn.write("Scene/Rooms/LunchHall_24x48.tscn", text)
    print("nodes:", text.count("[node "))
