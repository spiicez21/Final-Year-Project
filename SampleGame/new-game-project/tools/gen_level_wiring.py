"""Level wiring for the CSE/IT department build.

  * convert_1f_to_labs()  - 1F becomes the department floor: every classroom
    instance is repointed at the matching Lab room and the nodes are renamed.
  * place_seminar_gf()    - drops the seminar hall furniture into the empty
    ground-floor west hall, under the cafeteria.

Both are idempotent, so re-running is safe.

NOTE: build_school.build_ground_floor() derives GFLevel from 1FLevel, so it
reverts the lab resource paths - the ground floor keeps ordinary classrooms.
"""

import tscn

# Classroom sizes that exist as labs.
LAB_SIZES = ((16, 16), (20, 16), (24, 16))

# 1F node renames. Node "Lab" already exists, so the converted rooms start at 2.
LAB_RENAMES = [
    ("Classroom", "Lab2"),
    ("Classroom2", "Lab3"),
    ("Classroom3", "Lab4"),
    ("Classroom4", "Lab5"),
    ("Classroom7", "Lab6"),
    ("Classroom9", "Lab7"),
    ("Classroom10", "Lab8"),
    ("Classroom11", "Lab9"),
    ("Classroom12", "Lab10"),
    ("Classroom13", "Lab11"),
]

SEMINAR_TAG = "SEMINAR HALL"
SEMINAR_RES = "r_seminar_24x32"
CEILING_RES = "r_ceiling_tile"
CEILING = 3.0
SEMINAR_ORIGIN = (0.0, 0.0, 40.0)   # stage backs onto the z = 72 hall end wall


def classroom_to_lab(text):
    for w, d in LAB_SIZES:
        text = text.replace("res://Scene/Rooms/Classroom_%dx%d.tscn" % (w, d),
                            "res://Scene/Rooms/Lab_%dx%d.tscn" % (w, d))
    return text


def lab_to_classroom(text):
    for w, d in LAB_SIZES:
        text = text.replace("res://Scene/Rooms/Lab_%dx%d.tscn" % (w, d),
                            "res://Scene/Rooms/Classroom_%dx%d.tscn" % (w, d))
    return text


def convert_1f_to_labs():
    path = "Scene/1FLevel.tscn"
    text = tscn.read(path)
    if "Rooms/Classroom_" not in text:
        return "already converted"
    before = sum(text.count('[node name="%s"' % old) for old, _ in LAB_RENAMES)
    text = classroom_to_lab(text)
    for old, new in LAB_RENAMES:
        text = text.replace('[node name="%s"' % old, '[node name="%s"' % new)
    tscn.write(path, text)
    return "converted %d rooms" % before


def place_seminar_gf():
    path = "Scene/GFLevel.tscn"
    text = tscn.read(path)
    text = strip_gen(text, SEMINAR_TAG)
    if "SeminarFurniture_24x24.tscn" not in text:
        text = tscn.insert_after_last(
            text, "ext_resource",
            '[ext_resource type="PackedScene" '
            'path="res://Scene/Props/SeminarFurniture_24x24.tscn" id="%s"]\n' % SEMINAR_RES)
    if "res://Assets/ceiling_tile.glb" not in text:
        text = tscn.insert_after_last(
            text, "ext_resource",
            '[ext_resource type="PackedScene" path="res://Assets/ceiling_tile.glb" '
            'id="%s"]\n' % CEILING_RES)

    x, y, z = SEMINAR_ORIGIN
    node = ('[node name="SeminarHall" parent="." unique_id=%d instance=ExtResource("%s")]\n'
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %g, %g, %g)\n'
            % (tscn.new_id(), SEMINAR_RES, x, y, z))
    # The hall is a level volume, not a room shell, so it carries no ceiling of
    # its own - lay the same 4 m grid over the seminar footprint.
    node += ('\n[node name="SeminarCeiling" type="Node3D" parent="." unique_id=%d]\n'
             % tscn.new_id())
    i = 0
    for cx in range(2, 24, 4):
        for cz in range(int(z) + 2, int(z) + 32, 4):
            i += 1
            node += ('\n[node name="SemCeil%03d" parent="SeminarCeiling" unique_id=%d '
                     'instance=ExtResource("%s")]\n'
                     'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %g, %g, %g)\n'
                     % (i, tscn.new_id(), CEILING_RES, cx, CEILING, cz))
    text = text.rstrip() + "\n" + gen_block(SEMINAR_TAG, node.rstrip())
    tscn.write(path, text)
    return "placed at %g,%g,%g" % SEMINAR_ORIGIN


# The GEN block markers match build_school.py so the two stay interchangeable.
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


if __name__ == "__main__":
    print("1F:", convert_1f_to_labs())
    print("GF:", place_seminar_gf())
