"""Add ceilings to the room shells.

Every room is floored with 4 m Floor.glb tiles, so the ceiling is just the same
grid of ceiling_tile.glb instances lifted to the top of the 3 m walls. The tile
is modelled with its slab above the origin, so instancing at y = CEILING puts the
underside exactly on the wall top and leaves the hanging props (fans, projectors,
light bars, which all hang below their origin at y = 3.0) clear.

Idempotent: the ceiling lives in a GEN block that is stripped and rewritten.

Skipped:
  * Stairwell_* - the stairs run up through the slab.
  * LunchHall_24x48 - furniture only, and its hall is double height with the 2F
    roof deck bridging over it.
  * SeminarHall_24x24 - its house floor is a stepped slab rather than Floor.glb
    tiles, so gen_seminar_hall lays that ceiling itself.
"""

import re

import tscn

CEILING = 3.0
CEILING_RES = "r_ceiling_tile"
TAG = "CEILING"

SKIP = ("Stairwell_", "LunchHall_", "SeminarHall_")

FLOOR_RE = re.compile(
    r'\[ext_resource[^\]]*path="res://Assets/Floor\.glb"[^\]]*id="([^"]+)"\]')
NODE_RE = re.compile(
    r'\[node name="([^"]+)"(?:[^\]]*?parent="([^"]*)")?[^\]]*'
    r'instance=ExtResource\("([^"]+)"\)\]\s*\n'
    r'transform = Transform3D\(([^)]*)\)')

# --------------------------------------------------------------------------- #
# Level-level floors: the corridors, connectors and halls that sit directly in
# GFLevel / 1FLevel rather than inside an instanced room.
#
# 2FLevel is left alone - patch_second_floor() lays a roof deck over the whole
# footprint at y = 3.0..3.4, which already reads as its ceiling.
# --------------------------------------------------------------------------- #
LEVEL_TAG = "CORRIDOR CEILING"

LEVELS = {
    # file: (parents to skip, [exclusion rects as (x0, z0, x1, z1)])
    "Scene/GFLevel.tscn": (
        {"Porch"},                       # outdoor, already has its own canopy
        # the seminar end of the west hall is ceiled by the SEMINAR HALL block
        [(-1.0, 39.0, 25.0, 73.0)],
    ),
    "Scene/1FLevel.tscn": (
        set(),
        # the cafeteria hall is double height - the 2F roof deck bridges over it
        [(-1.0, 23.0, 25.0, 73.0)],
    ),
}


def _inside(x, z, rect):
    x0, z0, x1, z1 = rect
    return x0 <= x <= x1 and z0 <= z <= z1


def gen_block(tag, body):
    return "\n; --- GEN %s BEGIN ---\n%s\n; --- GEN %s END ---\n" % (tag, body, tag)


def strip_gen(text, tag):
    begin = "; --- GEN %s BEGIN ---" % tag
    end = "; --- GEN %s END ---" % tag
    i = text.find(begin)
    if i < 0:
        return text
    j = text.find(end, i)
    if j < 0:
        return text
    return text[:i] + text[j + len(end):]


def floor_cells(text, skip_parents=None, exclude=None):
    """(x, z) of every Floor.glb instance in the scene."""
    m = FLOOR_RE.search(text)
    if not m:
        return []
    floor_id = m.group(1)
    skip_parents = skip_parents or set()
    exclude = exclude or []
    cells = []
    for name, parent, res_id, nums in NODE_RE.findall(text):
        if res_id != floor_id:
            continue
        if parent in skip_parents:
            continue
        parts = [p.strip() for p in nums.split(",")]
        if len(parts) != 12:
            continue
        x, y, z = float(parts[9]), float(parts[10]), float(parts[11])
        if abs(y) > 0.5:            # stair landings and other split levels
            continue
        if any(_inside(x, z, r) for r in exclude):
            continue
        cells.append((x, z))
    return cells


def build_level(path):
    """Ceil the corridors and connectors that live directly in a level file."""
    skip_parents, exclude = LEVELS[path]
    text = tscn.read(path)
    text = strip_gen(text, LEVEL_TAG)
    cells = floor_cells(text, skip_parents, exclude)
    if not cells:
        return 0

    if "res://Assets/ceiling_tile.glb" not in text:
        text = tscn.insert_after_last(
            text, "ext_resource",
            '[ext_resource type="PackedScene" path="res://Assets/ceiling_tile.glb" '
            'id="%s"]\n' % CEILING_RES)

    body = ['[node name="CorridorCeiling" type="Node3D" parent="." unique_id=%d]\n'
            % tscn.new_id()]
    for i, (x, z) in enumerate(cells):
        body.append(
            '\n[node name="CorrCeil%03d" parent="CorridorCeiling" unique_id=%d '
            'instance=ExtResource("%s")]\n'
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %g, %g, %g)\n'
            % (i + 1, tscn.new_id(), CEILING_RES, x, CEILING, z))

    tscn.write(path, text.rstrip() + "\n" + gen_block(LEVEL_TAG, "".join(body).rstrip()))
    return len(cells)


def build(path):
    text = tscn.read(path)
    text = strip_gen(text, TAG)
    cells = floor_cells(text)
    if not cells:
        return 0

    if "res://Assets/ceiling_tile.glb" not in text:
        text = tscn.insert_after_last(
            text, "ext_resource",
            '[ext_resource type="PackedScene" path="res://Assets/ceiling_tile.glb" '
            'id="%s"]\n' % CEILING_RES)

    body = ['[node name="Ceiling" type="Node3D" parent="." unique_id=%d]\n'
            % tscn.new_id()]
    for i, (x, z) in enumerate(cells):
        body.append(
            '\n[node name="CeilTile%03d" parent="Ceiling" unique_id=%d '
            'instance=ExtResource("%s")]\n'
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %g, %g, %g)\n'
            % (i + 1, tscn.new_id(), CEILING_RES, x, CEILING, z))

    tscn.write(path, text.rstrip() + "\n" + gen_block(TAG, "".join(body).rstrip()))
    return len(cells)


def build_all():
    import glob
    import os

    report = []
    root = tscn.ROOT + "/Scene/Rooms"
    for full in sorted(glob.glob(root + "/*.tscn")):
        base = os.path.basename(full)
        if any(base.startswith(s) for s in SKIP):
            report.append((base, "skipped"))
            continue
        report.append((base, build("Scene/Rooms/" + base)))
    for path in LEVELS:
        report.append((path, build_level(path)))
    return report


if __name__ == "__main__":
    for row in build_all():
        print(row)
