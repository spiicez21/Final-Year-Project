"""Small helpers for reading/writing the hand-authored .tscn level files.

The levels are plain text scenes: a header block of [ext_resource]/[sub_resource]
entries followed by [node] blocks. Everything here works on that text form so the
generated output stays diff-friendly with what the Godot editor writes.
"""

import re
import random

ROOT = "D:/cut-adi-mame"

# Rooms are named <Kind>_<width>x<depth>, footprint is [0..w] x [0..d] in local space.
ROOM_SIZE_RE = re.compile(r"_(\d+)x(\d+)\.tscn$")

NODE_RE = re.compile(r"^\[node ", re.M)


def new_id():
    return random.randint(1, 2147483647)


def read(path):
    with open(f"{ROOT}/{path}", "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    with open(f"{ROOT}/{path}", "w", encoding="utf-8") as f:
        f.write(text)


def split_blocks(text):
    """Return (header, [node_block, ...]). Each node block keeps its trailing blank line."""
    first = NODE_RE.search(text)
    header, body = text[: first.start()], text[first.start():]
    starts = [m.start() for m in NODE_RE.finditer(body)] + [len(body)]
    blocks = [body[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    return header, blocks


def block_name(block):
    m = re.search(r'\[node name="([^"]*)"', block)
    return m.group(1) if m else ""


def block_parent(block):
    m = re.search(r'parent="([^"]*)"', block)
    return m.group(1) if m else None


def block_ext(block):
    m = re.search(r'instance=ExtResource\("([^"]*)"\)', block)
    return m.group(1) if m else None


def block_transform(block):
    m = re.search(r"^transform = Transform3D\(([^)]*)\)", block, re.M)
    if not m:
        return [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    return [float(v) for v in m.group(1).split(",")]


def insert_after_last(header, tag, text):
    """Insert text after the last [tag] entry. Godot needs all ext_resource
    declarations before the sub_resource ones, and both before any node."""
    last = None
    for m in re.finditer(r"^\[" + tag + r"\b", header, re.M):
        last = m
    if last is None:
        return header.rstrip() + "\n\n" + text
    nxt = re.compile(r"^\[", re.M).search(header, last.end())
    at = nxt.start() if nxt else len(header)
    return header[:at] + text + header[at:]


def ext_paths(header):
    """id -> res:// path for every ext_resource in the header."""
    out = {}
    for m in re.finditer(r'\[ext_resource[^\]]*path="([^"]*)"[^\]]*id="([^"]*)"\]', header):
        out[m.group(2)] = m.group(1)
    return out


def footprint(block, path):
    """World-space (x0, z0, x1, z1) of a top-level node, or None if unknown."""
    t = block_transform(block)
    bx, bz = t[9], t[11]
    if path.endswith("/Floor.glb"):
        return (bx - 2.0, bz - 2.0, bx + 2.0, bz + 2.0)
    m = ROOM_SIZE_RE.search(path)
    if not m:
        return None
    w, d = float(m.group(1)), float(m.group(2))
    # Columns of the basis: local +X maps to (t[0], t[2]), local +Z to (t[6], t[8]).
    corners = []
    for lx, lz in ((0, 0), (w, 0), (0, d), (w, d)):
        corners.append((bx + t[0] * lx + t[6] * lz, bz + t[2] * lx + t[8] * lz))
    xs = [c[0] for c in corners]
    zs = [c[1] for c in corners]
    return (min(xs), min(zs), max(xs), max(zs))


def occupied_cells(text, cell=4.0):
    """Set of (col, row) 4m grid cells covered by floor of a level scene."""
    header, blocks = split_blocks(text)
    paths = ext_paths(header)
    cells = set()
    for b in blocks:
        if block_parent(b) != ".":
            continue
        ext = block_ext(b)
        if not ext or ext not in paths:
            continue
        fp = footprint(b, paths[ext])
        if not fp:
            continue
        x0, z0, x1, z1 = fp
        c0, r0 = int(round(x0 / cell)), int(round(z0 / cell))
        c1, r1 = int(round(x1 / cell)), int(round(z1 / cell))
        for c in range(c0, c1):
            for r in range(r0, r1):
                cells.add((c, r))
    return cells


def merge_rects(cells):
    """Greedy maximal-rectangle merge of grid cells -> [(c0, r0, c1, r1), ...] (exclusive end)."""
    todo = set(cells)
    rects = []
    while todo:
        c0, r0 = min(todo, key=lambda p: (p[1], p[0]))
        c1 = c0
        while (c1 + 1, r0) in todo:
            c1 += 1
        r1 = r0
        while all((c, r1 + 1) in todo for c in range(c0, c1 + 1)):
            r1 += 1
        for c in range(c0, c1 + 1):
            for r in range(r0, r1 + 1):
                todo.discard((c, r))
        rects.append((c0, r0, c1 + 1, r1 + 1))
    return rects
