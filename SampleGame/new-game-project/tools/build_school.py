"""Generates the ground floor, the 1F lunch hall and the 2F roof.

Run from the project root:  python tools/build_school.py

Everything it writes is idempotent - the generated blocks are wrapped between
GEN markers so re-running replaces them instead of stacking duplicates.
"""

import re
import tscn
from tscn import read, write, new_id

CELL = 4.0

GEN_BEGIN = "; --- GEN {tag} BEGIN ---"
GEN_END = "; --- GEN {tag} END ---"


def gen_block(tag, body):
    return f"\n{GEN_BEGIN.format(tag=tag)}\n{body}\n{GEN_END.format(tag=tag)}\n"


def strip_gen(text, tag):
    pat = re.compile(
        r"\n?" + re.escape(GEN_BEGIN.format(tag=tag)) + r".*?" + re.escape(GEN_END.format(tag=tag)) + r"\n?",
        re.S,
    )
    return pat.sub("\n", text)


def inst(name, parent, ext, x=0.0, y=0.0, z=0.0, basis="1, 0, 0, 0, 1, 0, 0, 0, 1"):
    return (
        f'[node name="{name}" parent="{parent}" unique_id={new_id()} instance=ExtResource("{ext}")]\n'
        f"transform = Transform3D({basis}, {x}, {y}, {z})\n"
    )


def mesh_node(name, parent, mesh_id, x, y, z, mat_id=None):
    s = (
        f'[node name="{name}" type="MeshInstance3D" parent="{parent}" unique_id={new_id()}]\n'
        f"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x}, {y}, {z})\n"
        f'mesh = SubResource("{mesh_id}")\n'
    )
    if mat_id:
        s += f'surface_material_override/0 = SubResource("{mat_id}")\n'
    return s


def col_node(name, parent, shape_id, x, y, z, basis="1, 0, 0, 0, 1, 0, 0, 0, 1"):
    return (
        f'[node name="{name}" type="CollisionShape3D" parent="{parent}" unique_id={new_id()}]\n'
        f"transform = Transform3D({basis}, {x}, {y}, {z})\n"
        f'shape = SubResource("{shape_id}")\n'
    )


# Walls are 4 m wide, 3 m tall and sit centred on their placement position.
FACE_Z = "0, 0, 1, 0, 1, 0, -1, 0, 0"   # wall running along X (its normal faces Z)
FACE_X = "1, 0, 0, 0, 1, 0, 0, 0, 1"    # wall running along Z


# --------------------------------------------------------------------------- #
# Entrance hall - 32 x 16, drops into the north face of the ground floor.
# --------------------------------------------------------------------------- #
def build_entrance_hall():
    W, D = 32, 16
    head = [
        '[gd_scene format=3 uid="uid://cgfentrance001"]',
        "",
        '[ext_resource type="PackedScene" uid="uid://cgrpb6wire5c3" path="res://Assets/Floor.glb" id="1_floor"]',
        '[ext_resource type="PackedScene" uid="uid://c6hj4j5vvbxcy" path="res://Assets/Wall_4m.glb" id="4_w4"]',
        '[ext_resource type="PackedScene" uid="uid://b88wvq7av8osy" path="res://Assets/Window_Wall.glb" id="5_win"]',
        '[ext_resource type="PackedScene" uid="uid://csy2yb0athubv" path="res://Assets/Wall_Beam.glb" id="6_beam"]',
        '[ext_resource type="PackedScene" uid="uid://d4dy7pis2g3lb" path="res://Assets/writing_desk.glb" id="7_wdesk"]',
        '[ext_resource type="PackedScene" uid="uid://c8fflcubmkd5e" path="res://Assets/bench_wood.glb" id="8_bench"]',
        '[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="9_tube"]',
        "",
        '[sub_resource type="BoxShape3D" id="sh_floor"]',
        f"size = Vector3({W}, 0.2, {D})",
        "",
        '[sub_resource type="BoxShape3D" id="sh_wall4"]',
        "size = Vector3(0.2, 3, 4)",
        "",
        '[sub_resource type="BoxShape3D" id="sh_desk"]',
        "size = Vector3(1.1, 0.88, 0.6)",
        "",
        '[sub_resource type="BoxShape3D" id="sh_bench"]',
        "size = Vector3(1.6, 0.45, 0.36)",
        "",
        '[sub_resource type="BoxShape3D" id="sh_beam"]',
        "size = Vector3(0.5, 3, 0.5)",
        "",
        '[node name="EntranceHall_32x16" type="Node3D" unique_id=%d]' % new_id(),
        "",
    ]
    body = []
    n = 0
    for gx in range(2, W, 4):
        for gz in range(2, D, 4):
            n += 1
            body.append(inst(f"Floor{n}", ".", "1_floor", gx, 0, gz))

    # North face: 8 m doorway centred on the hall (x 12..20 stays open).
    for i, gx in enumerate(range(2, W, 4)):
        if gx in (14, 18):
            continue
        ext = "5_win" if gx in (10, 22) else "4_w4"
        body.append(inst(f"NorthWall{i}", ".", ext, gx, 0, 0, FACE_Z))

    # Portal columns either side of the doorway.
    for i, gx in enumerate((12, 20)):
        body.append(inst(f"PortalBeam{i}", ".", "6_beam", gx, 0, 0.25))

    # Free-standing columns carrying the double-height volume.
    for i, (gx, gz) in enumerate(((8, 8), (24, 8))):
        body.append(inst(f"HallBeam{i}", ".", "6_beam", gx, 0, gz))

    body.append(inst("Reception", ".", "7_wdesk", 27, 0, 13))
    for i, gz in enumerate((4, 6, 10, 12)):
        body.append(inst(f"WaitBench{i}", ".", "8_bench", 1.2, 0, gz, FACE_Z))
    for i, gx in enumerate(range(4, W, 8)):
        for j, gz in enumerate((4, 12)):
            body.append(inst(f"Tube{i}_{j}", ".", "9_tube", gx, 2.9, gz))

    body.append(f'[node name="Collision" type="StaticBody3D" parent="." unique_id={new_id()}]\n')
    body.append(col_node("CFloor", "Collision", "sh_floor", W / 2, 0, D / 2))
    k = 0
    for gx in range(2, W, 4):
        if gx in (14, 18):
            continue
        k += 1
        body.append(col_node(f"CNorth{k}", "Collision", "sh_wall4", gx, 1.5, 0, FACE_Z))
    for i, (gx, gz) in enumerate(((12, 0.25), (20, 0.25), (8, 8), (24, 8))):
        body.append(col_node(f"CBeam{i}", "Collision", "sh_beam", gx, 1.5, gz))
    body.append(col_node("CReception", "Collision", "sh_desk", 27, 0.44, 13))

    write("Scene/Rooms/EntranceHall_32x16.tscn", "\n".join(head) + "\n".join(body))


# --------------------------------------------------------------------------- #
# Lunch hall - fills the existing 24 x 48 west hall on 1F (floor already there).
# --------------------------------------------------------------------------- #
def build_lunch_hall():
    """The cafeteria layout lives in gen_cafeteria.py - just re-emit it here."""
    import gen_cafeteria

    write("Scene/Rooms/LunchHall_24x48.tscn", gen_cafeteria.build())
    return int(gen_cafeteria.W), int(gen_cafeteria.D)


def _build_lunch_hall_plain_seating():
    W, D = 24, 48
    head = [
        '[gd_scene format=3 uid="uid://cgflunchhall01"]',
        "",
        '[ext_resource type="PackedScene" uid="uid://b4ps43pgh5qlr" path="res://Assets/desk_wood.glb" id="1_table"]',
        '[ext_resource type="PackedScene" uid="uid://c8fflcubmkd5e" path="res://Assets/bench_wood.glb" id="2_bench"]',
        '[ext_resource type="PackedScene" uid="uid://4yuvs3kis081" path="res://Assets/tube_light_ceiling.glb" id="6_tube"]',
        "",
        '[sub_resource type="BoxShape3D" id="sh_table"]',
        "size = Vector3(1.8, 0.75, 0.9)",
        "",
        '[sub_resource type="BoxShape3D" id="sh_bench"]',
        "size = Vector3(1.6, 0.45, 0.36)",
        "",
        '[node name="LunchHall_24x48" type="Node3D" unique_id=%d]' % new_id(),
        "",
    ]
    body = []

    # Dining tables only - no serving counter, the whole hall is seating.
    tables = []
    for row, gz in enumerate(range(4, D - 2, 6)):
        for col, gx in enumerate((5, 12, 19)):
            tables.append((f"T{row}_{col}", gx, gz))
    for name, gx, gz in tables:
        body.append(inst(f"Table_{name}", ".", "1_table", gx, 0, gz))
        body.append(inst(f"BenchA_{name}", ".", "2_bench", gx, 0, gz - 1.1))
        body.append(inst(f"BenchB_{name}", ".", "2_bench", gx, 0, gz + 1.1))
    for i, gx in enumerate((6, 18)):
        for j, gz in enumerate(range(8, D, 8)):
            body.append(inst(f"Tube{i}_{j}", ".", "6_tube", gx, 2.9, gz))

    body.append(f'[node name="Collision" type="StaticBody3D" parent="." unique_id={new_id()}]\n')
    for name, gx, gz in tables:
        body.append(col_node(f"CT_{name}", "Collision", "sh_table", gx, 0.375, gz))
        body.append(col_node(f"CBA_{name}", "Collision", "sh_bench", gx, 0.225, gz - 1.1))
        body.append(col_node(f"CBB_{name}", "Collision", "sh_bench", gx, 0.225, gz + 1.1))

    write("Scene/Rooms/LunchHall_24x48.tscn", "\n".join(head) + "\n".join(body))
    return W, D


# --------------------------------------------------------------------------- #
# Ground floor - the 1F plan with the two centre classrooms replaced by the
# entrance hall, plus the porch and canopy outside the north face.
# --------------------------------------------------------------------------- #
ENTRANCE_X0, ENTRANCE_X1 = 36.0, 68.0   # world span of the hall
DROP_ROOMS = ("Classroom2", "Classroom3")
DROP_SEAL_X = (52.0,)                   # partition between the two dropped rooms


def build_ground_floor():
    # The lunch hall is 1F-only, so build the ground floor from the plain plan.
    text = strip_gen(read("Scene/1FLevel.tscn"), "LUNCH HALL")
    # 1F is the CSE/IT department floor; the ground floor keeps plain classrooms.
    import gen_level_wiring

    text = gen_level_wiring.lab_to_classroom(text)
    header, blocks = tscn.split_blocks(text)

    kept = []
    for b in blocks:
        name = tscn.block_name(b)
        parent = tscn.block_parent(b)
        if parent == "." and name in DROP_ROOMS:
            continue
        t = tscn.block_transform(b)
        x, z = t[9], t[11]
        drop_wall = any(abs(x - dx) < 0.6 for dx in DROP_SEAL_X) and -0.5 < z < 16.5
        if parent == "." and name.startswith("Seal") and drop_wall:
            continue
        if parent == "Collision" and drop_wall:
            continue
        kept.append(b)

    header = header.replace(
        '[gd_scene format=3 uid="uid://cun0lgi1ax80o"]',
        '[gd_scene format=3 uid="uid://cgfground0001"]',
    )
    header = tscn.insert_after_last(
        header,
        "ext_resource",
        '[ext_resource type="PackedScene" path="res://Scene/Rooms/EntranceHall_32x16.tscn" id="r_entrance_32x16"]\n\n',
    )
    subres = "\n".join(
        [
            '[sub_resource type="StandardMaterial3D" id="mat_canopy"]',
            "albedo_color = Color(0.55, 0.56, 0.58, 1)",
            "roughness = 0.85",
            "",
            '[sub_resource type="StandardMaterial3D" id="mat_step"]',
            "albedo_color = Color(0.62, 0.6, 0.56, 1)",
            "roughness = 0.9",
            "",
            '[sub_resource type="BoxMesh" id="mesh_canopy"]',
            "size = Vector3(28, 0.4, 14)",
            "",
            '[sub_resource type="BoxShape3D" id="sh_porch"]',
            "size = Vector3(24, 0.2, 12)",
            "",
            '[sub_resource type="BoxShape3D" id="sh_beam"]',
            "size = Vector3(0.5, 3, 0.5)",
            "",
            "",
        ]
    )

    # Porch steps: three 0.12 m treads running the width of the entrance.
    for i in range(3):
        subres += (
            f'[sub_resource type="BoxMesh" id="mesh_step{i}"]\n'
            f"size = Vector3({24 - i * 1.5}, 0.12, {1.2})\n\n"
            f'[sub_resource type="BoxShape3D" id="sh_step{i}"]\n'
            f"size = Vector3({24 - i * 1.5}, 0.12, {1.2})\n\n"
        )
    header = tscn.insert_after_last(header, "sub_resource", subres)

    body = []
    body.append(
        f'[node name="EntranceHall" parent="." unique_id={new_id()} '
        f'instance=ExtResource("r_entrance_32x16")]\n'
        f"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {ENTRANCE_X0}, 0, 0)\n"
    )
    body.append(f'[node name="Porch" type="Node3D" parent="." unique_id={new_id()}]\n')
    n = 0
    for gx in range(42, 64, 4):
        for gz in (-2, -6, -10):
            n += 1
            body.append(inst(f"PorchFloor{n}", "Porch", "15_hikcn", gx, 0, gz))
    for i, (gx, gz) in enumerate(((42, -2), (62, -2), (42, -10), (62, -10))):
        body.append(inst(f"PorchBeam{i}", "Porch", "21_l5kqx", gx, 0, gz))
    body.append(mesh_node("Canopy", "Porch", "mesh_canopy", 52, 3.2, -6, "mat_canopy"))
    for i in range(3):
        body.append(
            mesh_node(f"Step{i}", "Porch", f"mesh_step{i}", 52, -0.06 - i * 0.12, -12.6 - i * 1.2, "mat_step")
        )

    body.append(f'[node name="PorchCollision" type="StaticBody3D" parent="Porch" unique_id={new_id()}]\n')
    body.append(col_node("CPorch", "Porch/PorchCollision", "sh_porch", 52, 0, -6))
    for i, (gx, gz) in enumerate(((42, -2), (62, -2), (42, -10), (62, -10))):
        body.append(col_node(f"CPorchBeam{i}", "Porch/PorchCollision", "sh_beam", gx, 1.5, gz))
    for i in range(3):
        body.append(
            col_node(f"CStep{i}", "Porch/PorchCollision", f"sh_step{i}", 52, -0.06 - i * 0.12, -12.6 - i * 1.2)
        )

    out = header + "".join(kept).rstrip() + "\n" + gen_block("GROUND ENTRANCE", "\n".join(body).rstrip())
    write("Scene/GFLevel.tscn", out)
    # The seminar hall sits in the ground-floor west hall, under the cafeteria.
    gen_level_wiring.place_seminar_gf()


# --------------------------------------------------------------------------- #
# 1F: drop the lunch hall into the empty west hall (x 0..24, z 24..72).
# --------------------------------------------------------------------------- #
def patch_first_floor():
    text = strip_gen(read("Scene/1FLevel.tscn"), "LUNCH HALL")
    if "LunchHall_24x48.tscn" not in text:
        text = text.replace(
            '[ext_resource type="PackedScene" path="res://Scene/Rooms/Toilet_12x16.tscn"',
            '[ext_resource type="PackedScene" path="res://Scene/Rooms/LunchHall_24x48.tscn" id="r_lunch_24x48"]\n'
            '[ext_resource type="PackedScene" path="res://Scene/Rooms/Toilet_12x16.tscn"',
            1,
        )
    node = (
        f'[node name="LunchHall" parent="." unique_id={new_id()} instance=ExtResource("r_lunch_24x48")]\n'
        f"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 24)\n"
    )
    write("Scene/1FLevel.tscn", text.rstrip() + "\n" + gen_block("LUNCH HALL", node.rstrip()))


# --------------------------------------------------------------------------- #
# 2F: flat roof deck + parapet over the whole 2F footprint.
# --------------------------------------------------------------------------- #
ROOF_Y = 3.0        # top of the 3 m walls
DECK_T = 0.4
PARAPET_H = 1.0
PARAPET_T = 0.3

# 2F has no floor over the double-height lunch hall, so the deck has to bridge
# it - otherwise that bay is left open to the sky. Courtyards stay open.
ROOF_EXTRA = [(0.0, 24.0, 24.0, 72.0)]


def parapet_runs(cells):
    """Boundary edges of the cell set, merged into runs. Yields (axis, a, b, fixed, outward)."""
    edges = {"h": [], "v": []}
    for c, r in cells:
        if (c, r - 1) not in cells:
            edges["h"].append((r, c, -1))
        if (c, r + 1) not in cells:
            edges["h"].append((r + 1, c, 1))
        if (c - 1, r) not in cells:
            edges["v"].append((c, r, -1))
        if (c + 1, r) not in cells:
            edges["v"].append((c + 1, r, 1))
    runs = []
    for axis in ("h", "v"):
        by_line = {}
        for fixed, idx, out in edges[axis]:
            by_line.setdefault((fixed, out), []).append(idx)
        for (fixed, out), idxs in by_line.items():
            idxs.sort()
            start = prev = idxs[0]
            for i in idxs[1:]:
                if i == prev + 1:
                    prev = i
                    continue
                runs.append((axis, start, prev + 1, fixed, out))
                start = prev = i
            runs.append((axis, start, prev + 1, fixed, out))
    return runs


def patch_second_floor():
    text = strip_gen(strip_gen(read("Scene/2FLevel.tscn"), "ROOF RES"), "ROOF")
    cells = tscn.occupied_cells(text)
    for x0, z0, x1, z1 in ROOF_EXTRA:
        for c in range(int(x0 / CELL), int(x1 / CELL)):
            for r in range(int(z0 / CELL), int(z1 / CELL)):
                cells.add((c, r))
    rects = tscn.merge_rects(cells)
    runs = parapet_runs(cells)

    subres = [
        '[sub_resource type="StandardMaterial3D" id="mat_roof_deck"]',
        "albedo_color = Color(0.48, 0.47, 0.45, 1)",
        "roughness = 0.95",
        "",
        '[sub_resource type="StandardMaterial3D" id="mat_roof_parapet"]',
        "albedo_color = Color(0.66, 0.64, 0.6, 1)",
        "roughness = 0.85",
        "",
    ]
    nodes = [f'[node name="Roof" type="Node3D" parent="." unique_id={new_id()}]\n']
    colls = [f'[node name="RoofCollision" type="StaticBody3D" parent="Roof" unique_id={new_id()}]\n']

    for i, (c0, r0, c1, r1) in enumerate(rects):
        w, d = (c1 - c0) * CELL, (r1 - r0) * CELL
        cx, cz = (c0 + c1) / 2 * CELL, (r0 + r1) / 2 * CELL
        subres += [
            f'[sub_resource type="BoxMesh" id="mesh_deck{i}"]',
            f"size = Vector3({w}, {DECK_T}, {d})",
            "",
            f'[sub_resource type="BoxShape3D" id="sh_deck{i}"]',
            f"size = Vector3({w}, {DECK_T}, {d})",
            "",
        ]
        y = ROOF_Y + DECK_T / 2
        nodes.append(mesh_node(f"Deck{i}", "Roof", f"mesh_deck{i}", cx, y, cz, "mat_roof_deck"))
        colls.append(col_node(f"CDeck{i}", "Roof/RoofCollision", f"sh_deck{i}", cx, y, cz))

    for i, (axis, a, b, fixed, out) in enumerate(runs):
        length = (b - a) * CELL
        mid = (a + b) / 2 * CELL
        edge = fixed * CELL
        # Sit the parapet just inside the edge so it reads as a wall on the deck.
        off = -out * PARAPET_T / 2
        if axis == "h":
            w, d = length, PARAPET_T
            cx, cz = mid, edge + off
        else:
            w, d = PARAPET_T, length
            cx, cz = edge + off, mid
        subres += [
            f'[sub_resource type="BoxMesh" id="mesh_para{i}"]',
            f"size = Vector3({w}, {PARAPET_H}, {d})",
            "",
            f'[sub_resource type="BoxShape3D" id="sh_para{i}"]',
            f"size = Vector3({w}, {PARAPET_H}, {d})",
            "",
        ]
        y = ROOF_Y + DECK_T + PARAPET_H / 2
        nodes.append(mesh_node(f"Parapet{i}", "Roof", f"mesh_para{i}", cx, y, cz, "mat_roof_parapet"))
        colls.append(col_node(f"CPara{i}", "Roof/RoofCollision", f"sh_para{i}", cx, y, cz))

    header, blocks = tscn.split_blocks(text)
    header = tscn.insert_after_last(
        header, "sub_resource", gen_block("ROOF RES", "\n".join(subres).rstrip()).lstrip("\n") + "\n"
    )
    body = "\n".join(nodes) + "\n" + "\n".join(colls)
    out = header + "".join(blocks).rstrip() + "\n" + gen_block("ROOF", body.rstrip())
    write("Scene/2FLevel.tscn", out)
    return len(rects), len(runs)


# --------------------------------------------------------------------------- #
# Main: stack ground / 1F / 2F on exact 3 m steps and spawn at the entrance.
# --------------------------------------------------------------------------- #
def patch_main():
    text = read("Scene/Main.tscn")
    text = text.replace(
        '[ext_resource type="PackedScene" uid="uid://bc2uv3660an1o" path="res://Scene/2FLevel.tscn" id="1_floor"]',
        '[ext_resource type="PackedScene" path="res://Scene/GFLevel.tscn" id="1_gf"]\n'
        '[ext_resource type="PackedScene" uid="uid://cun0lgi1ax80o" path="res://Scene/1FLevel.tscn" id="1_1f"]\n'
        '[ext_resource type="PackedScene" uid="uid://bc2uv3660an1o" path="res://Scene/2FLevel.tscn" id="1_2f"]',
    )
    text = re.sub(
        r'\[node name="Floor1"([^\]]*)instance=ExtResource\("1_floor"\)\]\n',
        r'[node name="Floor1"\1instance=ExtResource("1_gf")]\n',
        text,
    )
    text = re.sub(
        r'\[node name="Floor2"([^\]]*)instance=ExtResource\("1_floor"\)\]\ntransform = [^\n]*\n',
        r'[node name="Floor2"\1instance=ExtResource("1_1f")]\n'
        r"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0)\n",
        text,
    )
    text = re.sub(
        r'\[node name="Floor3"([^\]]*)instance=ExtResource\("1_floor"\)\]\ntransform = [^\n]*\n',
        r'[node name="Floor3"\1instance=ExtResource("1_2f")]\n'
        r"transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 6, 0)\n",
        text,
    )
    text = re.sub(
        r'(\[node name="Player"[^\]]*\]\n)transform = [^\n]*\n',
        r"\1transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 52, 1, -6)\n",
        text,
    )
    write("Scene/Main.tscn", text)


if __name__ == "__main__":
    build_entrance_hall()
    build_lunch_hall()
    build_ground_floor()
    patch_first_floor()
    n_deck, n_para = patch_second_floor()
    patch_main()
    print(f"roof: {n_deck} deck slabs, {n_para} parapet runs")
