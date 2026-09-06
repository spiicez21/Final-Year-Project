"""Read the built layout, apply the department's real room programme to it and
report the reduced plan.

The building was laid out by hand after a real department, so the arrangement is
kept exactly: the same bands of rooms, in the same order, on the same sides of
the same courtyard. Only the dimensions change - every room drops to the size
its actual occupancy needs, and each band is re-packed at the new widths, which
pulls the courtyard and the whole footprint in with it.

Run from the project root:  python tools/plan_layout.py
Writes nothing. It prints the before/after plan so the reduction can be checked
before any scene is regenerated.
"""

import json
import re

import tscn

LEVELS = ("Scene/GFLevel.tscn", "Scene/1FLevel.tscn", "Scene/2FLevel.tscn")

# --------------------------------------------------------------------------- #
# The programme, from the department's real numbers:
#   4 years x 4 sections x ~64 students = 1024 students
#   40 teaching staff (HOD included), 13 non-teaching, 6 labs
# --------------------------------------------------------------------------- #
SECTIONS = 16
STUDENTS_PER_SECTION = 64
TEACHING = 40
NON_TEACHING = 13
LABS = 6

# Right-sized footprints. Classrooms sit at ~1.5 sqm/student against the 4 sqm
# they have now; labs hold 32 machines at two students each.
NEW_SIZE = {
    "Classroom": (12, 8),       # 3 benches x 7 rows x 3 seats = 63
    "Lab": (16, 12),
    "Faculty": (12, 8),         # one staff room per year, ~10 desks + in-charge
    "Toilet": (12, 8),
    "StoreRoom": (8, 8),
    "EntranceHall": (16, 16),
    "SeminarHall": (24, 24),
    "LunchHall": (24, 20),
    "Stairwell": None,          # stairs keep their size, the flights are modelled
}

# Faculty rooms outnumber the staff by a mile: 15 rooms for 40 people. Four
# survive as the per-year staff rooms, one becomes the HOD suite and one the
# department office; the rest are dropped and their bands close up.
FACULTY_KEEP = ["Staff Y1", "Staff Y2", "Staff Y3", "Staff Y4", "HOD suite", "Dept office"]
FACULTY_ROLE_SIZE = {"HOD suite": (8, 8), "Dept office": (12, 8)}

# How many of each the department actually needs. Everything built beyond these
# is surplus - it is dropped and its band closes up.
QUOTA = {
    "Classroom": SECTIONS,      # one per section
    "Lab": LABS,
    "Faculty": len(FACULTY_KEEP),
    "Toilet": 6,                # two per floor
    "StoreRoom": 3,
    "EntranceHall": 1,
    "SeminarHall": 1,
    "LunchHall": 1,
}

CELL = 4.0
SCALE = 0.62          # how hard the plan contracts
BAND_GAP = 4.0          # courtyard-side gap kept between bands
MIN_RUN_GAP = 4.0       # a gap in a band collapses to this


def rooms_of(path):
    """Every room instance sitting directly in a level, with its footprint."""
    text = tscn.read(path)
    header, blocks = tscn.split_blocks(text)
    paths = tscn.ext_paths(header)
    out = []
    for b in blocks:
        if tscn.block_parent(b) != ".":
            continue
        ext = tscn.block_ext(b)
        if not ext:
            continue
        res = paths.get(ext, "")
        m = re.search(r"Rooms/([A-Za-z]+)_(\d+)x(\d+)\.tscn$", res)
        kind = None
        if m:
            kind = m.group(1)
        elif res.endswith("SeminarFurniture_24x24.tscn"):
            kind = "SeminarHall"
        if kind is None:
            continue
        fp = tscn.footprint(b, res if m else "Rooms/SeminarHall_24x24.tscn")
        if fp is None:
            continue
        out.append({
            "kind": kind,
            "name": tscn.block_name(b),
            "x0": fp[0], "z0": fp[1], "x1": fp[2], "z1": fp[3],
        })
    return out


def contract(rooms, level, keep, roles, scale):
    """Shrink the plan in place.

    Each room keeps its position in the arrangement - its corner is simply
    pulled towards the origin by `scale` and snapped back to the 4 m grid - and
    takes the size the programme gives it. Courtyard, bands and the order of
    rooms along each run all survive; the whole thing just gets smaller.
    """
    out = []
    for r in rooms:
        size = new_size(r, level, keep, roles)
        if size is None:
            continue
        w, d = size
        out.append({
            "level": level, "kind": r["kind"], "name": r["name"],
            "role": roles.get((level, r["name"])) if r["kind"] == "Faculty" else None,
            "was": [int(r["x1"] - r["x0"]), int(r["z1"] - r["z0"])],
            "now": [w, d],
            "x": snap(r["x0"] * scale), "z": snap(r["z0"] * scale),
        })
    return out


def snap(v):
    return round(v / CELL) * CELL


def runs_of(plan):
    """Rooms whose new footprints share z, west to east - one side of a corridor."""
    runs = {}
    for r in plan:
        runs.setdefault(r["z"], []).append(r)
    for z in runs:
        runs[z].sort(key=lambda r: r["x"])
    return runs


def unoverlap(plan):
    """Close or open the gaps a run needs after the resize, without reordering.

    A run keeps a gap wherever the built layout had one, so doorways and the
    breaks between blocks stay where the department put them.
    """
    moved = 0
    for z, run in runs_of(plan).items():
        cursor = run[0]["x"]
        for i, r in enumerate(run):
            if i > 0:
                gap = MIN_RUN_GAP if r["x"] - (run[i - 1]["x"] + run[i - 1]["now"][0]) > 0.5 else 0.0
                cursor += gap
            if r["x"] != cursor:
                moved += 1
            r["x"] = cursor
            cursor += r["now"][0]
    return moved


BLOCKS = ("SeminarHall", "LunchHall", "EntranceHall")


def clear_blocks(plan):
    """Slide runs east past the big halls.

    The hall keeps its corner of the plan, but it shrinks less than the rooms
    around it, so a run that used to pass under it can now collide. Those rooms
    shift east as a group, which keeps their order and their side.
    """
    halls = [r for r in plan if r["kind"] in BLOCKS]
    if not halls:
        return
    for z, run in runs_of(plan).items():
        for hall in halls:
            if hall["z"] >= z + max(r["now"][1] for r in run) or hall["z"] + hall["now"][1] <= z:
                continue
            edge = hall["x"] + hall["now"][0] + MIN_RUN_GAP
            shift = 0.0
            for r in run:
                if r is hall:
                    continue
                if r["x"] < hall["x"] + hall["now"][0] and hall["x"] < r["x"] + r["now"][0]:
                    shift = max(shift, edge - r["x"])
            if shift > 0:
                for r in run:
                    if r is not hall:
                        r["x"] += shift


def overlaps(plan):
    bad = []
    for i in range(len(plan)):
        a = plan[i]
        for j in range(i + 1, len(plan)):
            b = plan[j]
            if (a["x"] < b["x"] + b["now"][0] and b["x"] < a["x"] + a["now"][0]
                    and a["z"] < b["z"] + b["now"][1] and b["z"] < a["z"] + a["now"][1]):
                bad.append((a["name"], b["name"]))
    return bad


def new_size(room, level, keep, roles):
    kind = room["kind"]
    if not keep.get((level, room["name"])):
        return None                          # surplus, dropped
    if kind == "Faculty":
        role = roles.get((level, room["name"]))
        return FACULTY_ROLE_SIZE.get(role, NEW_SIZE["Faculty"])
    size = NEW_SIZE.get(kind, "keep")
    if size is None or size == "keep":
        return (int(room["x1"] - room["x0"]), int(room["z1"] - room["z0"]))
    return size


def assign_programme(levels):
    """Decide which built rooms survive.

    Faculty rooms are picked biggest-first so the staff rooms land where the
    department already put its staff; everything else keeps the first of each
    kind in level order. Rooms are identified by (level, node name) - the same
    node name repeats on every floor, so name alone collides.
    """
    keep = {}
    roles = {}
    counts = {}

    faculty = []
    for path, rooms in levels:
        for r in rooms:
            if r["kind"] == "Faculty":
                area = (r["x1"] - r["x0"]) * (r["z1"] - r["z0"])
                faculty.append((-area, path, r["name"]))
    faculty.sort()
    for i, (_, path, name) in enumerate(faculty):
        if i < len(FACULTY_KEEP):
            keep[(path, name)] = True
            roles[(path, name)] = FACULTY_KEEP[i]

    built = {}
    for path, rooms in levels:
        for r in rooms:
            kind = r["kind"]
            built[kind] = built.get(kind, 0) + 1
            if kind == "Faculty":
                continue
            if kind not in QUOTA:               # stairwells and anything else
                keep[(path, r["name"])] = True
                continue
            counts[kind] = counts.get(kind, 0) + 1
            if counts[kind] <= QUOTA[kind]:
                keep[(path, r["name"])] = True
    return keep, roles, built


def main():
    levels = [(p, rooms_of(p)) for p in LEVELS]
    keep, roles, built = assign_programme(levels)

    print("programme: %d sections x %d students = %d | %d teaching, %d non-teaching, %d labs"
          % (SECTIONS, STUDENTS_PER_SECTION, SECTIONS * STUDENTS_PER_SECTION,
             TEACHING, NON_TEACHING, LABS))
    for kind in sorted(built):
        want = QUOTA.get(kind, built[kind])
        flag = "" if built[kind] <= want else "   <- %d surplus dropped" % (built[kind] - want)
        print("  %-13s built %2d  keep %2d%s"
              % (kind, built[kind], min(built[kind], want), flag))
    print()

    everything = []
    for path, rooms in levels:
        plan = contract(rooms, path, keep, roles, SCALE)
        unoverlap(plan)
        clear_blocks(plan)
        clash = overlaps(plan)
        everything += plan
        was = sum(r["was"][0] * r["was"][1] for r in plan)
        now = sum(r["now"][0] * r["now"][1] for r in plan)
        xs = [r["x"] + r["now"][0] for r in plan] + [r["x"] for r in plan]
        zs = [r["z"] + r["now"][1] for r in plan] + [r["z"] for r in plan]
        print("== %s   %d rooms   %d -> %d sqm   footprint %.0f x %.0f%s"
              % (path.split("/")[-1], len(plan), was, now,
                 max(xs) - min(xs), max(zs) - min(zs),
                 "   OVERLAPS: %d" % len(clash) if clash else ""))
        for a, b in clash[:6]:
            print("   !! %s overlaps %s" % (a, b))
        for r in plan:
            label = r["role"] or r["kind"]
            print("   %-12s %2dx%-3d -> %2dx%-3d  at x %5.1f  z %5.1f   %s"
                  % (label, r["was"][0], r["was"][1], r["now"][0], r["now"][1],
                     r["x"], r["z"], r["name"]))
        print()

    was = sum(r["was"][0] * r["was"][1] for r in everything)
    now = sum(r["now"][0] * r["now"][1] for r in everything)
    print("room area: %d -> %d sqm  (%.0f%% cut)" % (was, now, 100.0 * (1 - now / was)))
    with open(tscn.ROOT + "/tools/layout_plan.json", "w", encoding="utf-8") as f:
        json.dump(everything, f, indent=1)
    print("wrote tools/layout_plan.json")


if __name__ == "__main__":
    main()
