"""Furnish the toilets: Scene/Props/ToiletFurniture_12x8.tscn.

The shell is 12 x 8 with the door at x = 10.5 on the z = 8 wall and no windows.
There are no sanitaryware models in the kit, so the room is built the way the
rest of the project builds what it lacks - box meshes for the cubicle dividers
and the mirror, with the kit's wash trough and bins doing the rest.

Run from the project root:  python tools/gen_toilet.py
"""

import tscn

W, D = 12.0, 8.0
CEILING = 3.0

FACE_PZ = "1, 0, 0, 0, 1, 0, 0, 0, 1"
FACE_NZ = "-1, 0, 0, 0, 1, 0, 0, 0, -1"
FACE_PX = "0, 0, -1, 0, 1, 0, 1, 0, 0"
FACE_NX = "0, 0, 1, 0, 1, 0, -1, 0, 0"

STALL_X = (0.4, 1.7, 3.0, 4.3, 5.6, 6.9)   # six dividers -> five cubicles
STALL_D = 1.5                              # how far the cubicle runs out
STALL_H = 2.0
TROUGH_X = (8.0, 10.0)                     # two 1.8 m wash troughs

HEAD = '''[gd_scene format=3]

[ext_resource type="PackedScene" path="res://Assets/handwash_sink.glb" id="1_sink"]
[ext_resource type="PackedScene" path="res://Assets/dustbin_pair.glb" id="2_bin"]
[ext_resource type="PackedScene" path="res://Assets/exit_sign.glb" id="3_exit"]
[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="4_tube"]

[sub_resource type="StandardMaterial3D" id="m_stall"]
albedo_color = Color(0.62, 0.66, 0.68, 1)
roughness = 0.85

[sub_resource type="StandardMaterial3D" id="m_mirror"]
albedo_color = Color(0.72, 0.78, 0.82, 1)
roughness = 0.08
metallic = 0.85

[sub_resource type="BoxMesh" id="mesh_divider"]
material = SubResource("m_stall")
size = Vector3(0.06, %(stall_h)s, %(stall_d)s)

[sub_resource type="BoxMesh" id="mesh_mirror"]
material = SubResource("m_mirror")
size = Vector3(4.0, 1.0, 0.04)

[sub_resource type="BoxShape3D" id="sh_divider"]
size = Vector3(0.06, %(stall_h)s, %(stall_d)s)

[sub_resource type="BoxShape3D" id="sh_trough"]
size = Vector3(1.8, 1.2, 0.5)

[sub_resource type="BoxShape3D" id="sh_bin"]
size = Vector3(0.8, 0.62, 0.4)

[node name="ToiletFurniture" type="Node3D" unique_id=%(root)d]
'''

LAMP = """
[node name="%s" type="OmniLight3D" parent="Lighting" unique_id=%d]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %s, 2.7, %s)
light_color = Color(0.94, 0.97, 1, 1)
light_energy = 1.4
light_volumetric_fog_energy = 1.0
light_specular = 0.3
light_bake_mode = 1
shadow_bias = 0.04
shadow_normal_bias = 1.5
distance_fade_enabled = true
distance_fade_begin = 14.0
distance_fade_shadow = 9.0
distance_fade_length = 5.0
omni_range = 7.0
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


def col(name, sub, x, y, z, rot=FACE_PZ):
    return ('\n[node name="%s" type="CollisionShape3D" parent="Collision" unique_id=%d]\n'
            'transform = Transform3D(%s, %s, %s, %s)\n'
            'shape = SubResource("%s")\n'
            % (name, tscn.new_id(), rot, n(x), n(y), n(z), sub))


def build():
    out = [HEAD % {"root": tscn.new_id(), "stall_h": n(STALL_H), "stall_d": n(STALL_D)}]
    cols = []

    # --- cubicles against the blind wall ------------------------------------ #
    out.append(group("Cubicles"))
    for i, sx in enumerate(STALL_X):
        out.append(mesh("Divider%02d" % (i + 1), "Cubicles", "mesh_divider",
                        sx, STALL_H / 2, STALL_D / 2))
        cols.append(col("CDivider%02d" % (i + 1), "sh_divider",
                        sx, STALL_H / 2, STALL_D / 2))

    # --- wash troughs and the mirror over them ------------------------------ #
    out.append(group("Wash"))
    for i, tx in enumerate(TROUGH_X):
        out.append(inst("Trough%02d" % (i + 1), "Wash", "1_sink", tx, 0, 0.3))
        cols.append(col("CTrough%02d" % (i + 1), "sh_trough", tx, 0.6, 0.3))
    out.append(mesh("Mirror", "Wash", "mesh_mirror", 9.0, 1.75, 0.12))

    # --- fittings ----------------------------------------------------------- #
    out.append(group("Props"))
    out.append(inst("Dustbins", "Props", "2_bin", 11.4, 0, 2.4, FACE_NX))
    cols.append(col("CDustbins", "sh_bin", 11.4, 0.31, 2.4, FACE_NX))
    out.append(inst("ExitSign", "Props", "3_exit", 10.5, 2.55, D - 0.2, FACE_NZ))

    # --- lighting ----------------------------------------------------------- #
    out.append(group("Lighting"))
    j = 0
    for lx in (3.0, 9.0):
        j += 1
        out.append(inst("Batten%02d" % j, "Lighting", "4_tube", lx, 2.9, 3.4))
        out.append(LAMP % ("Lamp%02d" % j, tscn.new_id(), n(lx), n(3.4)))

    out.append('\n[node name="Collision" type="StaticBody3D" parent="." unique_id=%d]\n'
               % tscn.new_id())
    out.extend(cols)
    return "".join(out)


if __name__ == "__main__":
    text = build()
    tscn.write("Scene/Props/ToiletFurniture_%dx%d.tscn" % (W, D), text)
    print("cubicles:", len(STALL_X) - 1, "| troughs:", len(TROUGH_X),
          "| nodes:", text.count("[node "))
