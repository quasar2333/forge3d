"""Blockout -> slots -> parts. `f3d blockout` turns a blockout script into a plan and one work order per slot;
`f3d part` builds one part inside its slot, among its neighbours, and checks it there."""
import json
from contextlib import nullcontext
from itertools import product
import math
import os
import re
import shlex
import shutil
import tempfile
import time

import bpy
import cv2
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import refs as refmod
from . import runner, sheet
from .kit import core, ink, partkit, refkit, staging

BG = refmod.GRID_BG
SLOT_COLORS = ["#e4572e", "#2e86de", "#27ae60", "#f4b400", "#8e44ad", "#16a085", "#d35492", "#7f8c2d", "#c0392b",
               "#3d5a80", "#e58e26", "#5f27cd", "#0abde3", "#b33771", "#6ab04c", "#778ca3", "#cd6133", "#218c74",
               "#f78fb3", "#574b90"]
HIGHLIGHT = (1.0, 0.42, 0.04, 1.0)
GHOST = (0.80, 0.79, 0.77, 1.0)


class PartsError(Exception):
    pass


def _rel(path):
    try:
        r = os.path.relpath(path)
        return r if not r.startswith("..") else path
    except ValueError:
        return path


def _parts_command(action, parts_dir, name=None):
    args = ["python3", os.environ.get("F3D_LAUNCHER", "<skill>/scripts/f3d.py"), action]
    if name:
        args.append(name)
    if _rel(parts_dir) != "parts":
        args.extend(["--parts", _rel(parts_dir)])
    return shlex.join(args)


def _skill_file(*rel):
    return os.path.join(os.environ.get("F3D_SKILL_DIR", "<skill>"), *rel)


# ---------------------------------------------------------------- reference crops
def _packs(ref_dir):
    """[(label, Ref)]: the whole-picture pack, then its layer packs."""
    out = [("whole picture", refkit.Ref(ref_dir))]
    lj = os.path.join(ref_dir, "layers.json")
    if os.path.isfile(lj):
        with open(lj) as f:
            for rel in json.load(f).get("layers", []):
                d = os.path.normpath(os.path.join(ref_dir, rel))
                if os.path.isfile(os.path.join(d, "ref.json")):
                    out.append(("layer " + os.path.basename(d).split("_", 1)[-1], refkit.Ref(d)))
    return out


def _crop_box(view, box):
    u0, v0, u1, v1 = box
    m = min(max(0.07, 0.3 * max(u1 - u0, v1 - v0)), 0.16)     # generous: views draw a part wider than its slot
    f = view.frame
    return (max(f[0], u0 - m), max(f[1], v0 - m), min(f[2], u1 + m), min(f[3], v1 + m))


def _px_box(view, box):
    h, w = view.image.shape[:2]
    x0, x1 = int(math.floor(view.x_of(box[0]))), int(math.ceil(view.x_of(box[2])))
    y0, y1 = int(math.floor(view.y_of(box[3]))), int(math.ceil(view.y_of(box[1])))
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def _tile(view, box, title, picture=None, target=520, margin=38, grid=True):
    """A piece of a reference view, enlarged, with world coordinates along its edges.
    picture: BGR image on the view's pixel grid (default: the view's cutout over grey)."""
    if picture is None:
        picture = getattr(view, "_over_bg", None)
        if picture is None:
            picture = view._over_bg = refmod.over_bg(view.image)
    x0, y0, x1, y1 = _px_box(view, box)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = picture[y0:y1, x0:x1]
    zoom = min(6.0, target / float(max(crop.shape[:2])))
    big = cv2.resize(crop, (max(2, int(round(crop.shape[1] * zoom))), max(2, int(round(crop.shape[0] * zoom)))),
                     interpolation=cv2.INTER_LANCZOS4 if zoom > 1 else cv2.INTER_AREA)
    H, W = big.shape[:2]
    canvas = np.full((H + 2 * margin, W + 2 * margin, 3), BG, np.uint8)
    canvas[margin:margin + H, margin:margin + W] = big

    def to_px(u, v):
        return (int(round(margin + (view.x_of(u) - x0) * zoom)), int(round(margin + (view.y_of(v) - y0) * zoom)))
    if grid:
        step = refmod._nice_step(max(box[2] - box[0], box[3] - box[1]), 5)
        k = math.ceil(box[0] / step - 1e-9)
        while k * step <= box[2] + 1e-9:
            x, _ = to_px(k * step, 0)
            if margin <= x <= W + margin:
                cv2.line(canvas, (x, margin), (x, H + margin), (236, 236, 236), 1, cv2.LINE_AA)
                refmod._put(canvas, refmod._fmt(k * step), (x - 12, H + margin + 16), 0.42)
            k += 1
        k = math.ceil(box[1] / step - 1e-9)
        while k * step <= box[3] + 1e-9:
            _, y = to_px(0, k * step)
            if margin <= y <= H + margin:
                cv2.line(canvas, (margin, y), (W + margin, y), (236, 236, 236), 1, cv2.LINE_AA)
                refmod._put(canvas, refmod._fmt(k * step), (2, y + 5), 0.42)
            k += 1
    refmod._put(canvas, title, (margin, 16), 0.46, (120, 255, 255), 1)
    return canvas


def _stack_rows(rows, gap=10):
    rows = [r for r in rows if r is not None]
    w = max(r.shape[1] for r in rows)
    out = np.full((sum(r.shape[0] for r in rows) + gap * (len(rows) - 1), w, 3), BG, np.uint8)
    y = 0
    for r in rows:
        out[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def _ref_picture(packs, slot, path):
    """Reference crops around a slot: one row per pack that shows something there, one tile per view."""
    rows, used = [], []
    for label, R in packs:
        tiles = []
        for v in R:
            box = _crop_box(v, slot.box(v.name))
            if label != "whole picture":
                x0, y0, x1, y1 = _px_box(v, slot.box(v.name))
                if x1 <= x0 or y1 <= y0 or v._full_mask()[y0:y1, x0:x1].mean() < 0.04:
                    continue
            t = _tile(v, box, "%s, %s view   right = %s  up = %s" % (label, v.name, v.rec["u_axis"], v.rec["v_axis"]))
            if t is not None:
                tiles.append(t)
        if tiles:
            rows.append(refmod.hstack(tiles))
            used.append((label, R))
    cv2.imwrite(path, _stack_rows(rows))
    return used


def _regions_text(R, slot, limit=8):
    lines = []
    for v in R:
        b = slot.box(v.name)
        hits = []
        for r in v.regions.values():
            rb = r["box"]
            ix = min(b[2], rb[2]) - max(b[0], rb[0])
            iy = min(b[3], rb[3]) - max(b[1], rb[1])
            if ix > 0 and iy > 0 and ix * iy > 0.35 * (rb[2] - rb[0]) * (rb[3] - rb[1]):
                hits.append(r)
        hits.sort(key=lambda r: -r["share"])
        if hits:
            lines.append("  - %s: %s" % (v.name, ", ".join("%d %s" % (r["id"], r["color"]) for r in hits[:limit])))
    return lines


# ---------------------------------------------------------------- renders
def _workbench_tiles(colors, shots, tile):
    """Flat-lit clay renders with one colour per object. colors: {object: rgba}; shots: [(az, el, focus objects,
    margin)]. -> [BGR tiles]."""
    saved = {o: tuple(o.color) for o in colors}
    tmp = tempfile.mkdtemp(prefix="f3d_slot_")
    out = []
    try:
        for o, c in colors.items():
            o.color = c
        with staging.evaluation_setup("clay") as sc:
            sc.display.shading.color_type = "OBJECT"
            for i, (az, el, focus, margin) in enumerate(shots):
                pts = staging._sample_points(focus, 6000)
                cam = staging.temp_camera(az, el, pts, ortho=True, margin=margin)
                p = os.path.join(tmp, "t%02d.png" % i)
                staging.render_tile(cam, p, (tile, tile))
                staging.remove_camera(cam)
                out.append(sheet.composite(p, BG)[0])
    finally:
        for o, c in saved.items():
            o.color = c
        shutil.rmtree(tmp)
    return out


def _label(img, text):
    img = img.copy()
    refmod._put(img, text, (6, 16), 0.44, (120, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (110, 108, 104), 1)
    return img


def _slot_shots(slot, objs, everything):
    cx, cy = slot.center[0], slot.center[1]
    fb = ("back", 180) if cy > 0.04 else ("front", 0)
    side = ("left", -90) if cx < -0.03 else ("right", 90)
    q = (140 if cx >= 0 else -140) if cy > 0.04 else (40 if cx >= 0 else -40)
    return [("whole model, front", 0, 0, everything, 1.1), ("whole model, az %d" % q, q, 12, everything, 1.1),
            ("close: %s" % fb[0], fb[1], 0, objs, 2.2), ("close: %s" % side[0], side[1], 0, objs, 2.2),
            ("close: az %d" % q, q, 15, objs, 2.2)]


def _slot_picture(groups, name, slot, path, tile=352):
    everything = [o for objs in groups.values() for o in objs]
    colors = {o: (HIGHLIGHT if n == name else GHOST) for n, objs in groups.items() for o in objs}
    shots = _slot_shots(slot, groups[name], everything)
    tiles = _workbench_tiles(colors, [(az, el, f, m) for _t, az, el, f, m in shots], tile)
    cv2.imwrite(path, refmod.hstack([_label(t, s[0]) for t, s in zip(tiles, shots)], gap=4))


# ---------------------------------------------------------------- blockout
def _waves(decl):
    wave = {}

    def depth(n, stack):
        if n in wave:
            return wave[n]
        if n in stack:
            raise PartsError("BLOCKOUT: slots sit on each other in a circle: %s" % " -> ".join(stack + (n,)))
        wave[n] = 1 + max([depth(m, stack + (n,)) for m in decl[n]["on"]], default=0)
        return wave[n]
    for n in decl:
        depth(n, ())
    return wave


def blockout(script, ref_dir, parts_dir):
    """Run a blockout script and write the plan, the overview and a work order per slot. -> report dict."""
    t0 = time.time()
    runner.load_source(script, ref=ref_dir)
    decl = partkit.declared()
    if not decl:
        raise PartsError("BLOCKOUT: the script declares no slots. Give every placeholder shape to a slot:\n"
                         "    head = k.sphere(radii=(0.2, 0.18, 0.19), at=(0, 0, 0.62))\n"
                         "    k.slot(\"head\", head, note=\"what this part is and what matters about it\")")
    groups, loose = {}, []
    for o in core.scene_objects():
        n = o.get("f3d_slot")
        if n:
            groups.setdefault(n, []).append(o)
        else:
            loose.append(o.name)
    for n, rec in decl.items():
        if n not in groups:
            raise PartsError("BLOCKOUT: slot %r has no visible shape left (was it deleted or joined into another?)" % n)
        for m in rec["on"]:
            if m not in decl:
                raise PartsError("BLOCKOUT: slot %r sits on %r, which is not a slot (slots: %s)" % (n, m, ", ".join(decl)))
    wave = _waves(decl)
    order = sorted(decl, key=lambda n: (wave[n], list(decl).index(n)))
    os.makedirs(parts_dir, exist_ok=True)
    from .kit import materials
    plan = dict(version=1, ref=os.path.relpath(ref_dir, parts_dir) if ref_dir else None,
                script=os.path.relpath(script, parts_dir), order=order, slots={}, packs=[])
    from .bb import model as bbmodel
    if bbmodel.PLAN:
        plan["bb"] = dict(bbmodel.PLAN)             # a Blockbench cuboid model: parts are built with bb.part()
    for i, n in enumerate(order):
        col = SLOT_COLORS[i % len(SLOT_COLORS)]
        m = materials.mat(col, "matte", name="slot_" + n)
        for o in groups[n]:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
            if o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(m)
        bpy.context.view_layer.update()
        bb = core.bbox(groups[n])
        plan["slots"][n] = dict(on=decl[n]["on"], note=decl[n]["note"], wave=wave[n], color=col,
                                lo=[round(float(v), 4) for v in bb.min], hi=[round(float(v), 4) for v in bb.max],
                                objects=[o.name for o in groups[n]])
        if "review" in decl[n]:
            plan["slots"][n]["review"] = decl[n]["review"]
    rep = dict(slots=len(order), loose=loose, waves={}, compare=None, stale=[], time=0.0)
    for n in order:
        rep["waves"].setdefault(wave[n], []).append(n)
    # pictures of the whole blockout
    over = os.path.join(parts_dir, "overview.png")
    sheet.make_sheet(over, views="front,az40el10,right,az140el10,back,az-40el10", tile=400, cols=3, title="blockout")
    img = cv2.imread(over)
    rows = int(math.ceil(len(order) / 6.0))
    legend = np.full((rows * 26 + 8, img.shape[1], 3), BG, np.uint8)
    for i, n in enumerate(order):
        x, y = 10 + (i % 6) * (img.shape[1] // 6), 22 + (i // 6) * 26
        cv2.rectangle(legend, (x, y - 13), (x + 16, y + 3), refkit.hex_to_bgr(plan["slots"][n]["color"]), -1)
        refmod._put(legend, "%s (wave %d)" % (n, wave[n]), (x + 22, y), 0.42, (255, 255, 255), 1)
    cv2.imwrite(over, np.vstack([img, legend]))
    rep["overview"] = over
    if ref_dir:
        from . import compare
        rep["compare"] = compare.run(ref_dir, os.path.join(parts_dir, "blockout_compare.png"))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(parts_dir, partkit.BLOCKOUT), check_existing=False,
                                compress=True, copy=True)
    with open(os.path.join(parts_dir, partkit.PLAN), "w") as f:
        json.dump(plan, f, indent=1)
    # a work order per slot
    packs = _packs(ref_dir) if ref_dir else []
    plan["packs"] = [os.path.relpath(R.dir, parts_dir) for _l, R in packs]
    with open(os.path.join(parts_dir, partkit.PLAN), "w") as f:
        json.dump(plan, f, indent=1)
    rep["layers"] = len(packs) - 1 if packs else 0
    for n in order:
        d = os.path.join(parts_dir, n)
        os.makedirs(d, exist_ok=True)
        slot = partkit.Slot(parts_dir, plan, n)
        used = _ref_picture(packs, slot, os.path.join(d, "ref.png")) if packs else []
        _slot_picture(groups, n, slot, os.path.join(d, "slot.png"))
        with open(os.path.join(d, "brief.md"), "w") as f:
            f.write(_brief(parts_dir, plan, slot, packs, used))
        old = os.path.join(d, "report.json")
        if os.path.isfile(old) and os.path.isfile(partkit.part_file(parts_dir, n)):
            with open(old) as f:
                if partkit._moved(json.load(f).get("slot"), plan["slots"][n]):
                    rep["stale"].append(n)
    rep["time"] = time.time() - t0
    rep["plan"] = plan
    return rep


def _brief_cuboid(L, plan, slot, packs, d, n):
    cfg = plan["bb"]
    s = cfg["height"] / float(packs[0][1].height if packs else 1.0)
    lo, hi = [v * s for v in slot.lo], [v * s for v in slot.hi]
    L += ["", "## A Blockbench cuboid part", "",
          "The model is a Blockbench cuboid model (format `%s`): boxes on bones with a pixel-art texture, %g px tall "
          "(16 px = one Minecraft block), %d texel(s) per pixel. You build this part's bones and cubes." % (
              cfg["format"], cfg["height"], cfg["density"]),
          "",
          "- the slot in pixels: x %.1f .. %.1f,  y %.1f .. %.1f,  z %.1f .. %.1f" % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]),
          "- a point (u, v) read off the front picture is x = u * %g, z = v * %g pixels; the side picture gives "
          "y = u * %g" % (s, s, s),
          "", "## What to do", "",
          "1. Read `%s`, the section \"Building a cuboid part\"." % _skill_file("references", "blockbench.md"),
          "2. Write `%s/part.py`: `from forge3d import bb, kit as k`, then `m = bb.part()` and your bones and cubes." % d,
          "3. Build and check it: `%s` (5 to 30 s: wait for it in one call), then open `%s/check.png`. "
          "Repeat until it is right." % (_parts_command("part", slot.dir, n), d),
          "4. Reply with three lines: what you built, what the check says, what is still wrong or simplified.", ""]
    return "\n".join(L)


def _brief(parts_dir, plan, slot, packs, used):
    n = slot.name
    rec = plan["slots"][n]
    d = _rel(os.path.join(parts_dir, n))
    L = ["# Part `%s`" % n, "",
         "You build this one part of a larger model. Other agents build the other parts at the same time; a blockout "
         "(rough placeholder shapes) fixes where every part goes, so that the parts fit when they are put together.",
         "", "## What it is", "", rec["note"] or "(the lead wrote no note: go by the pictures)", "",
         "## Its slot", "",
         "World units are metres; the whole subject is %s tall, stands on z = 0, faces -Y; +X is the viewer's right "
         "in the front view." % refmod._fmt(packs[0][1].height if packs else 1.0), "",
         "- x %.3f .. %.3f,  y %.3f .. %.3f,  z %.3f .. %.3f   (size %.3f x %.3f x %.3f)" % (
             slot.lo[0], slot.hi[0], slot.lo[1], slot.hi[1], slot.lo[2], slot.hi[2], *slot.size)]
    if packs:
        L.append("- in the pictures: " + ";  ".join(
            "%s view u %.3f .. %.3f, v %.3f .. %.3f" % ((v.name,) + tuple(slot.box(v.name)[i] for i in (0, 2, 1, 3)))
            for v in packs[0][1]))
    L.append("- sits on: " + (", ".join("`%s`" % m for m in rec["on"]) or "nothing (a base part)"))
    near = []
    for m in plan["order"]:
        if m == n:
            continue
        o = plan["slots"][m]
        if all(o["lo"][i] <= slot.hi[i] + 0.02 and o["hi"][i] >= slot.lo[i] - 0.02 for i in range(3)):
            near.append(m)
    L.append("- next to it: " + (", ".join("`%s`" % m for m in near) or "nothing"))
    L += ["", "The slot is where the part goes and roughly how big it is; its exact shape is yours to get right "
          "from the pictures. Stay inside the slot (a little over is fine) and touch what you sit on.", "",
          "## Look at these first", "",
          "- `%s/slot.png`: the blockout with this part in orange, whole and close up." % d]
    if packs:
        L.append("- `%s/ref.png`: the reference around this part, enlarged, with coordinates along the edges%s." % (
            d, "; one row per picture set: %s" % ", ".join(lab for lab, _R in used) if len(used) > 1 else ""))
        L += ["", "## Reference packs for your script", "",
              "- `k.ref()`: the whole picture (`%s`)." % _rel(packs[0][1].dir)]
        for lab, R in packs[1:]:
            name = lab.split(" ", 1)[1]
            L.append("- `k.layer_ref(\"%s\")`: the %s layer alone, complete where the whole picture hides it%s." % (
                name, name, "" if any(R is u for _l, u in used) else " (shows nothing of this part)"))
        reg = _regions_text(packs[0][1], slot)
        if reg:
            L += ["", "Numbered colour regions of the whole picture in this area (for `mask(regions=[...])`; the "
                  "numbers are drawn in `%s/<view>_marks.png`):" % _rel(packs[0][1].dir)] + reg
    if plan.get("bb"):
        return _brief_cuboid(L, plan, slot, packs, d, n)
    L += ["", "## What to do", "",
          "1. Read `%s` (how to build a part, the rules, what done means)." % _skill_file("references", "parts.md"),
          "2. Write `%s/part.py`." % d,
          "3. Build and check it: `%s` (wait for it to finish). Open `%s/check.png` for the visible detail and "
          "`%s/probe.png` for thickness and joints. Repeat until it is right." % (_parts_command("part", parts_dir, n), d, d),
          "4. Reply with three lines: what you built, what the check says, what is still wrong or simplified.", ""]
    if review_focus(plan, n) == "face":
        L += ["Open the original face crop at native resolution and read the face section in the part manual.",
              "Measure the visible cheek and jaw edges. After building, inspect `%s/portrait_form.png` for their "
              "silhouette and cheek planes, then `%s/portrait.png` for the expression with the available hair." % (d, d)]
    return "\n".join(L)


def refresh_briefs(parts_dir, plan):
    """Layer packs made after the blockout: write the reference crops and briefs again. -> True if it did."""
    if not plan.get("ref"):
        return False
    ref_dir = os.path.normpath(os.path.join(parts_dir, plan["ref"]))
    packs = _packs(ref_dir)
    now = [os.path.relpath(R.dir, parts_dir) for _l, R in packs]
    if now == plan.get("packs", now):
        return False
    for n in plan["order"]:
        d = os.path.join(parts_dir, n)
        slot = partkit.Slot(parts_dir, plan, n)
        used = _ref_picture(packs, slot, os.path.join(d, "ref.png"))
        with open(os.path.join(d, "brief.md"), "w") as f:
            f.write(_brief(parts_dir, plan, slot, packs, used))
    plan["packs"] = now
    with open(os.path.join(parts_dir, partkit.PLAN), "w") as f:
        json.dump(plan, f, indent=1)
    return True


# ---------------------------------------------------------------- one part
def _bvh(objs, split_pairs=False):
    dg = core.depsgraph()
    out = []
    for o in objs:
        if o.type != "MESH":
            continue
        v, t = core.world_tris(o, dg)
        if not len(t):
            continue
        if split_pairs and v[:, 0].min() < 0 < v[:, 0].max():
            x = v[:, 0][t]
            left, right = x.max(axis=1) < 0, x.min(axis=1) > 0
            if left.any() and right.any() and (left | right).all():
                # A single mesh can hold both boots. Keep each side's vertices for distance sampling.
                for faces in (t[left], t[right]):
                    indices, local = np.unique(faces.ravel(), return_inverse=True)
                    verts = v[indices]
                    tree = BVHTree.FromPolygons(verts.tolist(), local.reshape(-1, 3).tolist(), all_triangles=True)
                    out.append((verts, tree))
                continue
        out.append((v, BVHTree.FromPolygons(v.tolist(), t.tolist(), all_triangles=True)))
    return out


def _gap(mine, others, reach=0.25):
    """Smallest distance between two sets of meshes (0 when they cut through each other)."""
    best = None
    for va, ta in mine:
        for _vb, tb in others:
            if ta.overlap(tb):
                return 0.0
            pts = va if len(va) <= 1500 else va[np.linspace(0, len(va) - 1, 1500).astype(int)]
            for p in pts:
                hit = tb.find_nearest(Vector(p.tolist()), reach if best is None else best)
                if hit[0] is not None and (best is None or hit[3] < best):
                    best = hit[3]
    return best


def _crop_camera(view, box):
    u0, v0, u1, v1 = box
    centre = Vector(view.u_vec) * ((u0 + u1) / 2.0) + Vector(view.v_vec) * ((v0 + v1) / 2.0)
    d = staging.direction(view.az, view.el)
    data = bpy.data.cameras.new("f3d_part_cam")
    data.type = "ORTHO"
    data.sensor_fit = "HORIZONTAL"
    data.ortho_scale = u1 - u0
    data.clip_start = 0.001
    data.clip_end = 200.0
    cam = bpy.data.objects.new("f3d_part_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    cam["f3d_helper"] = True
    cam.matrix_world = staging.look_at_matrix(centre + d * 20.0, centre, 0.0, view.az)
    return cam


def _face_spans(mask, centre, width, spans):
    """Measure the continuous frontal surface crossing each supplied landmark, excluding detached ears or trim."""
    tile = mask.shape[0]
    mid = (tile - 1) / 2
    scale = width / tile
    rows = []
    for z, left, right in spans:
        row = round(mid - (z - centre[2]) / scale)
        column = round(mid + ((left + right) * .5 - centre[0]) / scale)
        found = None
        if 0 <= row < tile and 0 <= column < tile:
            covered = mask[row] > .5
            if covered[column]:
                gaps = np.flatnonzero(~covered)
                before, after = gaps[gaps < column], gaps[gaps > column]
                if before.size and after.size:
                    found = tuple(centre[0] + (x - mid) * scale for x in (before[-1] + .5, after[0] - .5))
        rows.append((z, left, right, found))
    return rows


def portrait_picture(slot, R, path, tile=448, studio=False, form=False, spans=()):
    """Compare the face at a fixed world scale, including every finished neighbour visible at that angle."""
    if tile < 32:
        raise PartsError("PORTRAIT: use a tile of at least 32 pixels.")
    if any(not all(math.isfinite(n) for n in (z, lo, hi)) or hi <= lo for z, lo, hi in spans):
        raise PartsError("PORTRAIT: each --span needs a height and two finite X coordinates, left < right.")
    centre = np.array(slot.center)
    width = max(*slot.size, .0001) * 1.28
    corners = list(product(*zip(slot.lo, slot.hi)))
    side = R.side if R is not None else None
    views = [("front", 0, 0), ("profile", side.az if side else 90, 0),
             ("three-quarter +35", 35, 3), ("three-quarter -35", -35, 3)]
    objects = list(bpy.context.scene.objects)
    owners = {o.name for o in objects if o.type in core.GEOM_TYPES and o.get("f3d_part") == slot.name}
    if form and not owners:
        raise PartsError("PORTRAIT: the saved scene has no geometry tagged with the chosen face slot.")
    hidden = [o for o in objects if not o.hide_render and (o.get("f3d_placeholder") or
              (form and o.type in core.GEOM_TYPES and o.name not in owners))]
    for o in hidden:
        o.hide_render = True
    renders, masks = {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix="f3d_portrait_") as tmp, (
                nullcontext() if studio and not form else staging.evaluation_setup("clay" if form else "color")):
            for name, az, el in views:
                cam = staging.temp_camera(az, el, corners, ortho_scale=width)
                try:
                    file = staging.render_tile(cam, os.path.join(tmp, name + ".png"), (tile, tile))
                    img, masks[name] = sheet.composite(file, BG)
                    renders[name] = _label(img, ("bare form, " if form else "") + name)
                finally:
                    staging.remove_camera(cam)
    finally:
        for o in hidden:
            o.hide_render = False
    measured = _face_spans(masks["front"], centre, width, spans)
    for z, left, right, found in measured:
        if found is None:
            print("SPAN z=%g: no complete face span at this position in the portrait." % z)
        else:
            actual, expected = found[1] - found[0], right - left
            print("SPAN z=%.4f reference=%.4f mesh=%.4f difference=%+.1f%% (one pixel %.5f world units)" %
                  (z, expected, actual, 100 * (actual / expected - 1), width / tile))
    rows = []
    for v, direct, turn in [(R.views.get("front") if R is not None else None, "front", "three-quarter +35"),
                            (side, "profile", "three-quarter -35")]:
        tiles = []
        if v is not None:
            u, z = np.dot(centre, v.u_vec), np.dot(centre, v.v_vec)
            scale = (v.x_of(u + width) - v.x_of(u)) / tile
            matrix = np.array([[scale, 0, v.x_of(u) - scale * (tile - 1) / 2],
                               [0, scale, v.y_of(z) - scale * (tile - 1) / 2]])
            crop = cv2.warpAffine(refmod.over_bg(v.image), matrix, (tile, tile),
                                  flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderValue=BG)
            if form:
                edges, _ = cv2.findContours((masks[direct] > .5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(crop, edges, -1, (225, 52, 224), 1, cv2.LINE_AA)
                if direct == "front":
                    for z_value, left, right, _ in measured:
                        y = round((tile - 1) / 2 - (z_value - centre[2]) * tile / width)
                        for end in (left, right):
                            x = round((tile - 1) / 2 + (end - centre[0]) * tile / width)
                            cv2.drawMarker(crop, (x, y), (45, 210, 60), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
            tiles.append(_label(crop, "reference + bare outline" if form else "reference, " + v.name))
        rows.append(refmod.hstack(tiles + [renders[direct], renders[turn]]))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, _stack_rows(rows)):
        raise PartsError("PORTRAIT: could not write " + path)
    return path


def _check_picture(slot, R, mine, ctx, path, tile=440):
    """Reference, isolated part, its outline, and the actual fit from each reference camera; then
    three-quarter views of the part alone and in place."""
    tmp = tempfile.mkdtemp(prefix="f3d_part_")
    views = list(R) if R is not None else []
    boxes = {v.name: _crop_box(v, slot.box(v.name)) for v in views}
    sizes = {}
    for v in views:
        b = boxes[v.name]
        z = tile / max(b[2] - b[0], b[3] - b[1])
        sizes[v.name] = (max(8, int(round((b[2] - b[0]) * z))), max(8, int(round((b[3] - b[1]) * z))))
    shots = {}

    def set_hidden(objs, hide):
        for o in objs:
            o.hide_render = hide
    quarter = [(40 if slot.center[0] >= 0 else -40, 12), (-40 if slot.center[0] >= 0 else 40, 12),
               (140 if slot.center[0] >= 0 else -140, 12)]
    pts_mine = staging._sample_points(mine, 8000)
    inked = False
    if R is not None and R.ink and not slot.plan.get("bb"):     # the assembled model gets line art: show it
        try:
            ink.outline(mine)
            inked = True
        except ValueError:
            pass
    try:
        set_hidden(ctx, True)
        with staging.evaluation_setup("color"):
            for v in views:
                cam = _crop_camera(v, boxes[v.name])
                shots["p_" + v.name] = cv2.imread(staging.render_tile(cam, os.path.join(tmp, "p_%s.png" % v.name),
                                                                    sizes[v.name]), cv2.IMREAD_UNCHANGED)
                staging.remove_camera(cam)
            for i, (az, el) in enumerate(quarter):
                cam = staging.temp_camera(az, el, pts_mine, ortho=False, margin=1.12)
                shots["a%d" % i] = sheet.composite(staging.render_tile(cam, os.path.join(tmp, "a%d.png" % i),
                                                                       (tile, tile)), BG)[0]
                staging.remove_camera(cam)
            set_hidden(ctx, False)
            if ctx:
                for v in views:
                    cam = _crop_camera(v, boxes[v.name])
                    try:
                        p = staging.render_tile(cam, os.path.join(tmp, "fit_%s.png" % v.name), sizes[v.name])
                        shots["fit_" + v.name] = sheet.composite(p, BG)[0]
                    finally:
                        staging.remove_camera(cam)
                for i, (az, el) in enumerate(quarter[:1] + quarter[2:]):
                    cam = staging.temp_camera(az, el, pts_mine, ortho=False, margin=1.9)
                    shots["c%d" % i] = sheet.composite(staging.render_tile(cam, os.path.join(tmp, "c%d.png" % i),
                                                                           (tile, tile)), BG)[0]
                    staging.remove_camera(cam)
        if ctx and views:
            set_hidden(mine, True)
            with staging.evaluation_setup("clay"):
                for v in views:
                    cam = _crop_camera(v, boxes[v.name])
                    shots["g_" + v.name] = cv2.imread(staging.render_tile(cam, os.path.join(tmp, "g_%s.png" % v.name),
                                                                        sizes[v.name]), cv2.IMREAD_UNCHANGED)
                    staging.remove_camera(cam)
    finally:
        set_hidden(ctx, False)
        set_hidden(mine, False)
        if inked:
            ink.clear()
        shutil.rmtree(tmp)
    rows = []
    for v in views:
        b = boxes[v.name]
        w, h = sizes[v.name]
        ref = _tile(v, b, "reference, %s" % v.name, target=tile, margin=0, grid=False)
        if ref is None:
            continue
        ref = cv2.resize(ref, (w, h))
        part = shots["p_" + v.name]
        a = part[:, :, 3:4].astype(np.float32) / 255.0
        back = np.full((h, w, 3), BG, np.float32)
        g = shots.get("g_" + v.name)
        if g is not None:
            ga = g[:, :, 3:4].astype(np.float32) / 255.0 * 0.55
            back = g[:, :, :3].astype(np.float32) * ga + back * (1 - ga)
        mid = (part[:, :, :3].astype(np.float32) * a + back * (1 - a)).astype(np.uint8)
        over = (ref.astype(np.float32) * 0.75 + 255 * 0.25).astype(np.uint8)
        cs, _h = cv2.findContours((part[:, :, 3] > 100).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(over, cs, -1, (255, 0, 255), 2, cv2.LINE_AA)
        row = [_label(ref, "reference, %s view" % v.name), _label(mid, "part in isolation"),
               _label(over, "isolated outline (magenta)")]
        if ctx:
            row.append(_label(shots["fit_" + v.name], "in place"))
        rows.append(refmod.hstack(row, gap=4))
    last = [_label(shots["a%d" % i], "part alone, az %d" % quarter[i][0]) for i in range(len(quarter))]
    last += [_label(shots[k], "in place") for k in ("c0", "c1") if k in shots]
    rows.append(refmod.hstack(last, gap=4))
    cv2.imwrite(path, _stack_rows(rows, gap=6))
    return path


def _file_stamp(path):
    try:
        stat = os.stat(path)
        return [stat.st_mtime_ns, stat.st_size]
    except FileNotFoundError:
        return None


def _picture_context(parts_dir, plan, name):
    """Everything except the owner's blend can change while that worker is building it."""
    paths = [partkit.PLAN, partkit.BLOCKOUT]
    paths += [os.path.join(n, partkit.PART) for n in plan["order"] if n != name]
    if plan.get("ref"):
        ref = plan["ref"]
        paths.extend(os.path.join(ref, f) for f in os.listdir(os.path.join(parts_dir, ref))
                     if f == "ref.json" or f.endswith(".png"))
    return {p: _file_stamp(os.path.join(parts_dir, p)) for p in paths}


def _picture_stamps(d, names):
    return {p: _file_stamp(os.path.join(d, p)) for p in names}


def part(parts_dir, name, render=True):
    """Build parts/<name>/part.py inside its slot, among its neighbours; check it; save parts/<name>/part.blend.
    -> report dict."""
    t0 = time.time()
    plan = partkit.load_plan(parts_dir)
    if name not in plan["slots"]:
        raise PartsError("PART: the blockout has no slot %r (slots: %s)" % (name, ", ".join(plan["order"])))
    d = os.path.join(parts_dir, name)
    script = os.path.join(d, "part.py")
    if not os.path.isfile(script):
        raise PartsError("PART: no script at %s. Read %s and write it." % (_rel(script), _rel(os.path.join(d, "brief.md"))))
    ref_dir = os.path.normpath(os.path.join(parts_dir, plan["ref"])) if plan.get("ref") else None
    refreshed = refresh_briefs(parts_dir, plan)
    context_at_start = _picture_context(parts_dir, plan, name) if render else None
    core.reset()
    refkit.set_default(ref_dir)
    built = {}
    ctx = []
    for n in plan["order"]:
        if n == name:
            continue
        objs, done = partkit.load_slot(parts_dir, plan, n)
        built[n] = done
        for o in objs:
            o["f3d_context"] = n
            if not done:
                o["f3d_placeholder"] = True
            if not done and o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(partkit.placeholder_material())
            if o.type in core.GEOM_TYPES:
                ctx.append(o)
    partkit.set_active(parts_dir, name)
    try:
        runner.run_script(script, fresh=False)
    finally:
        partkit.set_active(None, None)
    from .bb import model as bbmodel
    cuboid = bbmodel.PART["model"]
    bbmodel.PART["model"] = None
    if plan.get("bb") and (cuboid is None or not cuboid.cubes):
        raise PartsError("PART %s: this is a Blockbench cuboid model, so the part is built with `m = bb.part()` and "
                         "m.bone / m.cube / m.beam; the script made no cubes. See \"Building a cuboid part\" in %s." % (
                             name, _skill_file("references", "blockbench.md")))
    if cuboid is not None and cuboid.cubes:
        import pickle
        from .bb import to_blender
        bbfile = cuboid.save(os.path.join(d, "part.bbmodel"), texture=False)
        to_blender.load(bbfile)
        with open(os.path.join(d, bbmodel.PART_FILE), "wb") as f:
            pickle.dump(cuboid, f)
    mine_all = [o for o in bpy.context.scene.objects
                if not o.get("f3d_context") and not o.get("f3d_helper") and o.type not in ("CAMERA", "LIGHT")]
    mine = core.scene_objects()
    if not mine:
        raise PartsError("PART %s: the script ran but left no geometry in the scene (objects must stay visible; do "
                         "not delete or hide what you built)." % name)
    slot = partkit.Slot(parts_dir, plan, name)
    issues = []
    bb = core.bbox(mine)
    lo, hi = np.array(bb.min), np.array(bb.max)
    slo, shi = np.array(slot.lo), np.array(slot.hi)
    pad = np.maximum(np.maximum(0.03, 0.2 * (shi - slo)), 0.06 * float((shi - slo).max()))   # a slot is rough
    over = []
    for i, ax in enumerate("XYZ"):
        if lo[i] < slo[i] - pad[i]:
            over.append("%.3f beyond the slot toward -%s" % (slo[i] - lo[i], ax))
        if hi[i] > shi[i] + pad[i]:
            over.append("%.3f beyond the slot toward +%s" % (hi[i] - shi[i], ax))
    if over:
        issues.append(("WARN", "outside", "the part leaves its slot: %s. Parts that outgrow their slot run into "
                       "their neighbours; if the reference really needs this, say so in your report." % "; ".join(over)))
    small = ["%s (%.3f of %.3f)" % (ax, hi[i] - lo[i], shi[i] - slo[i]) for i, ax in enumerate("XYZ")
             if hi[i] - lo[i] < 0.55 * (shi[i] - slo[i]) and shi[i] - slo[i] > 0.03]
    if small:
        issues.append(("INFO", "small", "the part is much smaller than its slot along %s; check it against the "
                       "reference crop." % ", ".join(small)))
    if lo[2] < -0.004:
        issues.append(("WARN", "below_ground", "the part reaches z = %.3f, below the ground (z = 0)." % lo[2]))
    mine_bvh = _bvh(mine, split_pairs=True) if slot.on else []
    left = [(v, tree) for v, tree in mine_bvh if v[:, 0].max() < 0]
    right = [(v, tree) for v, tree in mine_bvh if v[:, 0].min() > 0]
    paired = left and right and len(left) + len(right) == len(mine_bvh)
    touch = {}
    side_gaps = {"-X": [], "+X": []} if paired else {}
    for m in slot.on:
        others = _bvh([o for o in ctx if o.get("f3d_context") == m])
        if paired:
            gaps = [_gap(side, others) for side in (left, right)]
            for key, gap in zip(side_gaps, gaps):
                if gap is not None:
                    side_gaps[key].append(gap)
            g = min((gap for gap in gaps if gap is not None), default=None)
        else:
            g = _gap(mine_bvh, others) if others else None
        touch[m] = g
        if g is None or g > 0.006:
            issues.append(("WARN", "gap", "the part does not touch `%s`, which it sits on (%s). Move or extend it "
                           "until they meet or overlap slightly%s." % (
                               m, "gap %.3f" % g if g is not None else "further than 0.25 away",
                               "" if built.get(m) else "; note `%s` is still its blockout shape" % m)))
    side_touch = {key: min(values, default=None) for key, values in side_gaps.items()}
    if any(g is not None and g <= 0.006 for g in side_touch.values()):
        for side, g in side_touch.items():
            if g is None or g > 0.006:
                issues.append(("WARN", "gap", "only one side of the pair touches its supports. Check the %s side "
                               "in place (%s)." % (side, "over 0.25 away" if g is None else "gap %.3f" % g)))
    from .kit import qa
    lint = qa.lint(mine + ctx, verbose=False)
    names = {o.name for o in mine}
    for level, code, msg in lint["issues"]:
        if code in ("below_ground", "above_ground", "hidden") or not any("'%s'" % n in msg for n in names):
            continue
        if code == "floating" and not any(msg.startswith("'%s'" % n) or (", '%s'" % n) in msg.split(" not touching")[0]
                                          for n in names):
            continue
        issues.append((level, code, msg))
    tris = 0
    dg = core.depsgraph()
    for o in mine:
        tris += len(core.world_tris(o, dg)[1])
    diag = float(np.linalg.norm(hi - lo))
    if tris > 200000:
        issues.append(("WARN", "heavy", "the part has %s triangles; over 200,000 is too heavy for one part. Use the "
                       "default res= of k.volume / k.inflate and k.decimate(obj, faces=...) BEFORE painting." % format(tris, ",")))
    elif tris > 60000 and diag < 0.25:
        issues.append(("INFO", "heavy", "%s triangles is a lot for a part this small; k.decimate(obj, faces=15000) "
                       "before painting keeps the model light." % format(tris, ",")))
    for o in mine_all:
        o["f3d_part"] = name
        if o.parent is not None and o.parent not in mine_all:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
    bpy.context.view_layer.update()
    check = portrait = form = None
    if render:
        R = refkit.Ref(ref_dir) if ref_dir else None
        check = _check_picture(slot, R, mine, ctx, os.path.join(d, "check.png"))
        if review_focus(plan, name) == "face":
            portrait = portrait_picture(slot, R, os.path.join(d, "portrait.png"))
            form = portrait_picture(slot, R, os.path.join(d, "portrait_form.png"), form=True)
        fitted = _probe_neighbours(slot, ctx)
        _probe, inside = probe_picture(mine, os.path.join(d, "probe.png"), fitted=fitted)
        if inside > 0.04 and not plan.get("bb"):
            issues.append(("WARN", "open", "%.0f %% of visible pixels in the isolated, uncut views are backfaces "
                           "(red in %s). Check normals, cloth thickness and the fit on the body." % (
                               inside * 100, _rel(os.path.join(d, "probe.png")))))
    bpy.data.libraries.write(partkit.part_file(parts_dir, name), set(mine_all), fake_user=False, compress=True)
    if cuboid is not None and cuboid.cubes:
        flat = [c.name for c in cuboid.cubes if all(not f.painted for f in c.faces.values())]
        if flat:
            issues.append(("WARN", "unpainted", "%d cube(s) have no texture and will be black: %s. Paint them with "
                           ".panel(colour) or .paint(colour)." % (len(flat), ", ".join(flat[:6]))))
    rep = dict(name=name, ok=not any(i[0] == "ERROR" for i in issues), issues=issues, objects=len(mine), tris=tris,
               cubes=len(cuboid.cubes) if cuboid is not None else None,
               bones=len(cuboid.bones) if cuboid is not None else None,
               slot=dict(lo=list(slot.lo), hi=list(slot.hi)), bbox=dict(lo=lo.tolist(), hi=hi.tolist()),
               touch=touch, side_touch=side_touch, built_against=built, check=check, portrait=portrait, form=form,
               time=time.time() - t0, refreshed=refreshed)
    if render and context_at_start == _picture_context(parts_dir, plan, name):
        names = ["check.png", "probe.png"] + (["portrait.png", "portrait_form.png"] if portrait else [])
        rep["render_evidence"] = dict(version=2, context=context_at_start, inside=inside,
                                      fitted=sorted({o["f3d_context"] for o in fitted}),
                                      files=_picture_stamps(d, names + [partkit.PART]))
    with tempfile.NamedTemporaryFile(mode="w", dir=d, delete=False) as f:
        json.dump(rep, f, indent=1, default=float)
    os.replace(f.name, os.path.join(d, "report.json"))
    return rep


def status(parts_dir):
    """[(name, wave, state, note)] for every slot: state is 'finished', 'stale', 'script, not built', 'not started'."""
    plan = partkit.load_plan(parts_dir)
    out = []
    for n in plan["order"]:
        d = os.path.join(parts_dir, n)
        rec = plan["slots"][n]
        state, note = "not started", ""
        if os.path.isfile(partkit.part_file(parts_dir, n)):
            state = "finished"
            rp = os.path.join(d, "report.json")
            if os.path.isfile(rp):
                with open(rp) as f:
                    r = json.load(f)
                if partkit._moved(r.get("slot"), rec):
                    state = "stale"
                    note = "the blockout moved its slot; build it again"
                else:
                    w = [i for i in r.get("issues", []) if i[0] in ("ERROR", "WARN")]
                    note = "%d warning(s): %s" % (len(w), "; ".join(i[1] for i in w)) if w else "clean"
                    newer = [m for m in rec["on"] if os.path.isfile(partkit.part_file(parts_dir, m)) and
                             os.path.getmtime(partkit.part_file(parts_dir, m)) > os.path.getmtime(partkit.part_file(parts_dir, n))]
                    if newer:
                        note += "; built before %s was (re)built: check it still fits" % ", ".join(newer)
        elif os.path.isfile(os.path.join(d, "part.py")):
            state = "script, not built"
        out.append((n, rec["wave"], state, note))
    return out


def ready(parts_dir):
    """Which unbuilt parts can start now. -> ([names whose supports are all built], [(name, [supports it waits for])])"""
    plan = partkit.load_plan(parts_dir)
    done = {n for n in plan["order"] if os.path.isfile(partkit.part_file(parts_dir, n))}
    go, wait = [], []
    for n in plan["order"]:
        if n in done:
            continue
        missing = [m for m in plan["slots"][n]["on"] if m not in done]
        if missing:
            wait.append((n, missing))
        else:
            go.append(n)
    return go, wait


def blame(ref_dir):
    """Which part each difference between the assembled model and the reference outlines belongs to.
    -> [(view, slot, missing, extra)] as fractions of the reference outline's area, worst first; [] when the
    scene was not put together with k.assemble()."""
    from scipy import ndimage
    from .kit import raster
    groups = {}
    for o in core.scene_objects():
        n = o.get("f3d_part")
        if n:
            groups.setdefault(n, []).append(o)
    if len(groups) < 2:
        return []
    R = refkit.Ref(ref_dir)
    names = list(groups)
    tris = {n: raster.scene_tris(groups[n]) for n in names}
    out = []
    for v in R:
        zoom = min(1.0, 360.0 / max(v.size_px))
        sil = {n: raster.silhouette(tris[n], v.rec, zoom) > 0 for n in names}
        h, w = next(iter(sil.values())).shape
        ref = cv2.resize(v._full_mask().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        model = np.zeros((h, w), bool)
        for s in sil.values():
            model |= s
        dist = np.stack([ndimage.distance_transform_edt(~sil[n]) if sil[n].any() else np.full((h, w), 1e9)
                         for n in names])
        owner = dist.argmin(0)
        area = float(max(ref.sum(), 1))
        for i, n in enumerate(names):
            miss = float((ref & ~model & (owner == i)).sum()) / area
            extra = float((model & ~ref & (owner == i)).sum()) / area
            if miss >= 0.008 or extra >= 0.008:
                out.append((v.name, n, miss, extra))
    out.sort(key=lambda r: -(r[2] + r[3]))
    return out


def format_blame(rows, limit=8):
    if not rows:
        return None
    L = ["PARTS behind the outline differences (share of the reference outline; ask the builder of that part):"]
    for view, n, miss, extra in rows[:limit]:
        bits = []
        if miss >= 0.008:
            bits.append("%.0f%% of the reference is missing next to it (too small, too short or out of place)" % (miss * 100))
        if extra >= 0.008:
            bits.append("%.0f%% sticks out beyond the reference" % (extra * 100))
        L.append("  [%s] %-10s %s" % (view, n, "; ".join(bits)))
    L.append("  a drawing is not a photograph: a part the views draw inconsistently (a bow or fin facing the viewer "
             "in every view, a tail on the same picture side front and back) cannot match all of them. Match the "
             "front, keep it one clean 3D shape, and leave such a difference alone.")
    return "\n".join(L)


# ---------------------------------------------------------------- independent review
MODEL = "_model"                 # parts/_model holds the review of the assembled model
MAX_ROUNDS = 3                   # explicit part reviews, including existing projects
MODEL_ROUNDS = 2
DEFECTS = "defects.md"
STRUCTURAL_ISSUES = {"gap", "open", "floating", "unpainted"}


def review_focus(plan, name):
    """Explicit ownership wins. Common anatomical slot names also work in existing plans."""
    explicit = plan["slots"][name].get("review")
    if explicit:
        return explicit
    words = set(name.split("_"))
    note = plan["slots"][name].get("note", "").lower()
    anatomical = bool(re.search(r"\b(eyes?|brows?|cheeks?|jaw|muzzle|snout|chin|mouth|gaze|irises)\b|眼|下颌|口鼻|五官|脸", note))
    character = any(set(n.split("_")) & {"hair", "fringe", "braids", "eyes", "pelvis", "thighs"} for n in plan["order"])
    if name in ("eyes", "face") or (words & {"head", "face"} and (anatomical or character)
                                    and not words & {"wear", "gear", "plate"}):
        return "face"
    if words & {"hair", "fringe", "braids", "mane"} and not words & {"bow", "clip", "ribbon", "band", "pin"}:
        return "hair"
    if words & {"body", "pelvis", "hips", "legs", "thighs"}:
        return "joint"
    return "assembly"


def _scheduled_reviews(focus, directory):
    if focus in ("face", "hair"):
        return 2
    if focus in ("joint", "detail"):
        return 1
    try:
        with open(os.path.join(directory, "report.json")) as f:
            report = json.load(f)
    except FileNotFoundError:
        return int(os.path.isfile(os.path.join(directory, partkit.PART)))  # imported part without diagnostics
    return int(any(level == "ERROR" or (level == "WARN" and code in STRUCTURAL_ISSUES)
                   for level, code, _ in report.get("issues", [])))


def read_defects(path):
    """-> (verdict 'pass' | 'fix' | None, [defect lines]) of a reviewer's defects.md."""
    if not os.path.isfile(path):
        return None, []
    verdict, items = None, []
    with open(path, errors="replace") as f:
        for line in f:
            t = line.strip()
            match = re.fullmatch(r"VERDICT:\s*(PASS|FIX)", t, re.I)
            if match:
                verdict = match[1].lower()
            else:
                item = re.match(r"(?:[-*]|\d+[.)])\s+(.+)", t)
                if item:
                    items.append(item[1])
    if verdict == "pass":
        return verdict, items
    return ("fix" if verdict == "fix" and items else None), items


def _verdict_time(d):
    """Bind a verdict to the geometry its reviewer saw, even if a build finished while they wrote."""
    written = os.path.getmtime(os.path.join(d, DEFECTS))
    try:
        with open(os.path.join(d, "review_input.json")) as f:
            return min(written, json.load(f)["built_at"])
    except FileNotFoundError:
        return written  # review created by an earlier skill version or written by hand


def _archives(d):
    r = os.path.join(d, "reviews")
    if not os.path.isdir(r):
        return []
    entries = [(int(m[1]), os.path.join(r, f)) for f in os.listdir(r)
               if (m := re.fullmatch(r"round_(\d+)\.md", f))]
    return sorted(entries)


def _rounds(d):
    """Reviews made so far: the filed ones plus the current verdict."""
    filed = sum(read_defects(p)[0] is not None for _, p in _archives(d))
    return filed + int(read_defects(os.path.join(d, DEFECTS))[0] is not None)


def _review_record(d, limit):
    path = os.path.join(d, DEFECTS)
    verdict, items = read_defects(path)
    if verdict:
        return path, verdict, items, _verdict_time(d)
    # Earlier versions archived the last verdict even on a read-only rerender.
    if _rounds(d) >= limit:
        for _, archive in reversed(_archives(d)):
            verdict, items = read_defects(archive)
            if verdict:
                return archive, verdict, items, os.path.getmtime(archive)
    return path, None, [], 0


def stages(parts_dir):
    """What every part needs next. -> [dict(name, stage, round, defects, waits)], stage one of:
    build, review (a fresh agent judges it), fix (a fresh agent repairs the listed defects), done."""
    plan = partkit.load_plan(parts_dir)
    states = {n: state for n, _w, state, _note in status(parts_dir)}
    built = {n for n, state in states.items() if state == "finished"}
    model_file, mv, mitems, mtime_model = _review_record(os.path.join(parts_dir, MODEL), MODEL_ROUNDS)
    out = []
    for n in plan["order"]:
        d = os.path.join(parts_dir, n)
        focus = review_focus(plan, n)
        rec = dict(name=n, stage="build", round=_rounds(d), defects=[], stale=states[n] == "stale",
                   focus=focus, limit=_scheduled_reviews(focus, d),
                   waits=[m for m in plan["slots"][n]["on"] if m not in built])
        if n in built:
            pt = os.path.getmtime(partkit.part_file(parts_dir, n))
            df, verdict, items, reviewed = _review_record(d, MAX_ROUNDS)
            fresh = verdict is not None and reviewed >= pt
            mine = [x for x in mitems if x.lower().startswith(n + ":")] if mv == "fix" else []
            if mine and mtime_model >= pt:
                rec.update(stage="fix", defects=mine, why="model", report=model_file)
            elif mine and not fresh:
                rec.update(stage="done", unverified=True)  # verify the assembly repair on the assembly
            elif not fresh:                         # built or repaired since its last review
                rec["stage"] = "review" if rec["round"] < rec["limit"] else "done"
                if verdict == "fix":
                    rec["previous"] = items
                rec["unverified"] = rec["stage"] == "done" and rec["round"] > 0
            elif verdict == "fix":                  # every review that finds defects is followed by a repair
                rec.update(stage="fix", defects=items, why="part", report=df)
            else:
                rec["stage"] = "done"
        out.append(rec)
    return out


def _load_all(parts_dir, name=None):
    """The scene with every slot (finished parts where they exist). -> (plan, ref_dir, mine, ctx)."""
    plan = partkit.load_plan(parts_dir)
    ref_dir = os.path.normpath(os.path.join(parts_dir, plan["ref"])) if plan.get("ref") else None
    core.reset()
    refkit.set_default(ref_dir)
    mine, ctx = [], []
    for n in plan["order"]:
        objs, done = partkit.load_slot(parts_dir, plan, n)
        for o in objs:
            if o.type not in core.GEOM_TYPES:
                continue
            if name is None or n == name:
                o["f3d_part"] = n
                mine.append(o)
            else:
                o["f3d_context"] = n
                ctx.append(o)
            if not done:
                o["f3d_placeholder"] = True
            if not done and o.type == "MESH":
                o.data.materials.clear()
                o.data.materials.append(partkit.placeholder_material())
    bpy.context.view_layer.update()
    return plan, ref_dir, mine, ctx


def _inside_material():
    m = bpy.data.materials.new("f3d_inside")
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    lw = nt.nodes.new("ShaderNodeLayerWeight")          # lit like a headlamp, so it reads from below as well
    ramp = nt.nodes.new("ShaderNodeMix")
    ramp.data_type = "RGBA"
    ramp.inputs["A"].default_value = (0.78, 0.78, 0.76, 1.0)
    ramp.inputs["B"].default_value = (0.16, 0.16, 0.17, 1.0)
    nt.links.new(lw.outputs["Facing"], ramp.inputs["Factor"])
    front = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(ramp.outputs["Result"], front.inputs["Color"])
    back = nt.nodes.new("ShaderNodeEmission")
    back.inputs["Color"].default_value = (0.9, 0.05, 0.05, 1.0)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(front.outputs[0], mix.inputs[1])
    nt.links.new(back.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = False
    return m


def _probe_neighbours(slot, ctx):
    attached = set(slot.on)
    attached.update(n for n, rec in slot.plan["slots"].items() if slot.name in rec["on"])
    return [o for o in ctx if o.get("f3d_context") in attached and not o.get("f3d_placeholder")]


def probe_picture(objs, path, tile=420, fitted=()):
    """Expose backfaces in isolation, then show joints in colour with their finished neighbours.
    The red share measures only the isolated, uncut views."""
    pts = staging._sample_points(objs, 12000)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2.0
    rows, red, seen = [], 0, 0
    vl = bpy.context.view_layer
    old = vl.material_override
    mat = _inside_material()
    keep = {o.name for o in objs}
    others = [o for o in bpy.context.scene.objects if o.type in core.GEOM_TYPES and o.name not in keep and not o.hide_render]
    for o in others:
        o.hide_render = True
    try:
        with tempfile.TemporaryDirectory(prefix="f3d_probe_") as tmp, staging.evaluation_setup("color"):
            p = os.path.join(tmp, "tile.png")
            for in_place in ([False, True] if fitted else [False]):
                vl.material_override = None if in_place else mat
                if in_place:
                    for o in fitted:
                        if o in others:
                            o.hide_render = False
                shots = []
                for label, az, el, cut in (("below, front", 0, -50, False), ("below, back", 180, -50, False),
                                           ("above", 30, 60, False), ("front cut", 0, 0, True),
                                           ("side cut", 90, 0, True)):
                    cam = staging.temp_camera(az, el, pts, ortho=True, margin=1.1)
                    try:
                        if cut:
                            d = staging.direction(az, el)
                            cam.data.clip_start = max(0.001, float((np.array(cam.matrix_world.translation) - centre) @ np.array(d)))
                        staging.render_tile(cam, p, (tile, tile))
                    finally:
                        staging.remove_camera(cam)
                    if not cut and not in_place:
                        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                        a = img[:, :, 3] > 100
                        red += int((a & (img[:, :, 2] > 150) & (img[:, :, 1] < 90)).sum())
                        seen += int(a.sum())
                    shots.append(_label(sheet.composite(p, BG)[0], ("fitted: " if in_place else "surface: ") + label))
                rows.append(refmod.hstack(shots, gap=4))
    finally:
        vl.material_override = old
        bpy.data.materials.remove(mat)
        for o in others:
            o.hide_render = False
    cv2.imwrite(path, _stack_rows(rows, gap=4))
    return path, (red / seen if seen else 0.0)


def _review_pictures(parts_dir, name, d, tile):
    plan, ref_dir, mine, ctx = _load_all(parts_dir, name)
    pics = []
    R = refkit.Ref(ref_dir) if ref_dir else None
    if name is not None:
        pics.append(_check_picture(partkit.Slot(parts_dir, plan, name), R, mine, ctx, os.path.join(d, "check.png"),
                                   tile=tile or 520))
    else:
        if R is not None and R.ink and not plan.get("bb"):
            try:
                ink.outline(mine)
            except ValueError:
                pass
        p, _i = sheet.make_sheet(os.path.join(d, "sheet.png"), views="front,right,back,az40el10,az-40el10,az140el10,az-140el10,az20el40",
                                 tile=tile or 560, cols=4, title="assembled model")
        pics.append(p)
        if R is not None:
            from . import compare
            pics.append(compare.run(ref_dir, os.path.join(d, "compare.png"))["image"])
    for target in ([name] if name else plan["order"]):
        if review_focus(plan, target) == "face":
            portrait = "portrait.png" if name else "portrait_" + target + ".png"
            slot = partkit.Slot(parts_dir, plan, target)
            pics.append(portrait_picture(slot, R, os.path.join(d, portrait), tile or 448))
            form = os.path.splitext(portrait)[0] + "_form.png"
            pics.append(portrait_picture(slot, R, os.path.join(d, form), tile or 448, form=True))
    if name is None:
        ink.clear()
    fitted = _probe_neighbours(partkit.Slot(parts_dir, plan, name), ctx) if name else []
    probe, inside = probe_picture(mine, os.path.join(d, "probe.png"), tile=tile or 420, fitted=fitted)
    pics.append(probe)
    return pics, inside, sorted({o["f3d_context"] for o in fitted})


def review(parts_dir, name=None, tile=None):
    """Bind an independent verdict to a build. Reuse builder renders only with identical inputs and outputs."""
    if tile is not None and tile < 32:
        raise PartsError("REVIEW: --tile is the tile width in pixels (at least 32).")
    plan = partkit.load_plan(parts_dir)
    if name is not None and name not in plan["slots"]:
        raise PartsError("REVIEW: unknown slot %r" % name)
    targets = [name] if name else plan["order"]
    missing = [n for n in targets if not os.path.isfile(partkit.part_file(parts_dir, n))]
    if missing:
        raise PartsError("REVIEW: build the remaining parts first: " + ", ".join(missing))
    built_at = max(os.path.getmtime(partkit.part_file(parts_dir, n)) for n in targets)
    ref_dir = os.path.normpath(os.path.join(parts_dir, plan["ref"])) if plan.get("ref") else None
    d = os.path.join(parts_dir, name or MODEL)
    os.makedirs(os.path.join(d, "reviews"), exist_ok=True)
    df = os.path.join(d, DEFECTS)
    evidence = {}
    if name and tile is None and os.path.isfile(os.path.join(d, "report.json")):
        with open(os.path.join(d, "report.json")) as f:
            evidence = json.load(f).get("render_evidence", {})
    reuse = (evidence.get("version") == 2 and evidence["context"] == _picture_context(parts_dir, plan, name)
             and evidence["files"] == _picture_stamps(d, evidence["files"]))
    if reuse:
        pics = [os.path.join(d, p) for p in evidence["files"] if p != partkit.PART]
        inside, fitted = evidence["inside"], evidence["fitted"]
    else:
        pics, inside, fitted = _review_pictures(parts_dir, name, d, tile)
    verdict, _ = read_defects(df)
    current = verdict is not None and _verdict_time(d) >= built_at
    count = _rounds(d)
    # At the cap reviewers may still reopen the pictures to verify a final repair.
    # Preserve its last verdict, including older projects with off-by-one archive filenames.
    at_limit = count >= (MAX_ROUNDS if name else MODEL_ROUNDS)
    if not current and not at_limit:
        if verdict is not None:
            # Old workspaces numbered their first archive round_2. Append without overwriting it.
            folder = os.path.join(d, "reviews")
            number = max((i for i, _ in _archives(d)), default=0) + 1
            os.replace(df, os.path.join(folder, "round_%d.md" % number))
        with open(os.path.join(d, "review_input.json"), "w") as f:
            json.dump({"built_at": built_at}, f)
    return dict(dir=d, pictures=pics, round=count + int(not current and not at_limit), current=current,
                at_limit=at_limit, inside=inside, ref=ref_dir,
                fitted=fitted, reused=reuse,
                parts=list(plan["order"]), note=plan["slots"][name]["note"] if name else "")
