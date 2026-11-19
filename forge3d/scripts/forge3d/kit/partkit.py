"""Parts over a blockout: the model is first laid out as rough placeholder shapes, every placeholder is a slot,
each slot is filled by a part built on its own (by its own agent, in its own script), and the finished parts
are put back together. The slots are what keeps separately built parts fitting each other."""
import json
import os
import re

import bpy
import numpy as np

from . import core

PLAN = "plan.json"
BLOCKOUT = "blockout.blend"
PART = "part.blend"
_state = {"declared": {}, "active": None}       # active: dict(dir=<parts folder>, name=<slot being built>)
_VIEW_AXES = {"front": ((1, 0, 0), (0, 0, 1)), "back": ((-1, 0, 0), (0, 0, 1)), "right": ((0, 1, 0), (0, 0, 1)),
              "left": ((0, -1, 0), (0, 0, 1)), "top": ((1, 0, 0), (0, 1, 0)), "bottom": ((1, 0, 0), (0, -1, 0))}


def reset():
    _state["declared"] = {}


def declared():
    return _state["declared"]


def set_active(parts_dir, name):
    _state["active"] = dict(dir=os.path.abspath(parts_dir), name=name) if parts_dir else None


def load_plan(parts_dir):
    path = os.path.join(parts_dir, PLAN)
    if not os.path.isfile(path):
        raise FileNotFoundError("no blockout plan at %s (make one with: f3d blockout blockout.py --ref refs)" % path)
    with open(path) as f:
        return json.load(f)


def find_parts_dir(path="parts"):
    p = core.in_path(path)
    if os.path.isfile(os.path.join(p, PLAN)):
        return p
    q = os.path.abspath(path)
    if os.path.isfile(os.path.join(q, PLAN)):
        return q
    raise FileNotFoundError("no blockout plan in %s (make one with: f3d blockout blockout.py --ref refs)" % p)


def append_objects(blend, names=None):
    """Objects of a .blend file brought into the scene (all of them, or the named ones)."""
    with bpy.data.libraries.load(blend) as (src, dst):
        dst.objects = [n for n in src.objects if names is None or n in names]
    out = []
    for o in dst.objects:
        if o is not None:
            bpy.context.scene.collection.objects.link(o)
            out.append(o)
    bpy.context.view_layer.update()
    return out


def part_file(parts_dir, name):
    return os.path.join(parts_dir, name, PART)


def placeholder_material():
    m = bpy.data.materials.get("f3d_placeholder")
    if m is None:
        from . import materials
        m = materials.mat("#b9b6b2", "matte", name="f3d_placeholder")
    return m


def load_slot(parts_dir, plan, name, finished=True):
    """One slot's geometry: its finished part when there is one (and finished=True), else its blockout shape.
    -> (objects, is_finished)."""
    f = part_file(parts_dir, name)
    if finished and os.path.isfile(f):
        return append_objects(f), True
    objs = append_objects(os.path.join(parts_dir, BLOCKOUT), set(plan["slots"][name]["objects"]))
    return objs, False


class Slot:
    """The place one part is built into. Sizes are world units; the model faces -Y, +X is the viewer's right.

    name, note      what the part is (the note the lead wrote for it)
    lo, hi          opposite corners (x, y, z) of the slot's bounding box;  center, size  derived from them
    on              names of the slots this part sits on
    below           the objects it sits on (finished parts where they exist, blockout shapes otherwise)
    neighbours      {slot name: [objects]} of every other slot, for k.seat / k.stick / measuring
    box(view)       the slot's rectangle (u0, v0, u1, v1) in a reference view, e.g. F.mask(box=S.box("front"))
    shape()         a copy of the blockout shape as real geometry, when it is a good start for the part
    """

    def __init__(self, parts_dir, plan, name):
        if name not in plan["slots"]:
            raise KeyError("the blockout has no slot %r (it has: %s)" % (name, ", ".join(plan["order"])))
        rec = plan["slots"][name]
        self.dir, self.plan, self.name = parts_dir, plan, name
        self.note = rec.get("note", "")
        self.on = list(rec.get("on", []))
        self.lo, self.hi = tuple(rec["lo"]), tuple(rec["hi"])

    def __repr__(self):
        return "<Slot %s x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f on %s>" % (
            self.name, self.lo[0], self.hi[0], self.lo[1], self.hi[1], self.lo[2], self.hi[2], self.on or "-")

    @property
    def center(self):
        return tuple((a + b) / 2.0 for a, b in zip(self.lo, self.hi))

    @property
    def size(self):
        return tuple(b - a for a, b in zip(self.lo, self.hi))

    def box(self, view="front", pad=0.0):
        if view not in _VIEW_AXES:
            raise ValueError("view must be one of %s" % ", ".join(_VIEW_AXES))
        ua, va = (np.array(a, float) for a in _VIEW_AXES[view])
        c = np.array([[x, y, z] for x in (self.lo[0], self.hi[0]) for y in (self.lo[1], self.hi[1])
                      for z in (self.lo[2], self.hi[2])])
        u, v = c @ ua, c @ va
        return (float(u.min() - pad), float(v.min() - pad), float(u.max() + pad), float(v.max() + pad))

    @property
    def neighbours(self):
        out = {}
        for o in bpy.context.scene.objects:
            n = o.get("f3d_context")
            if n and o.type in core.GEOM_TYPES:
                out.setdefault(n, []).append(o)
        return out

    @property
    def below(self):
        nb = self.neighbours
        return [o for n in self.on for o in nb.get(n, [])]

    def shape(self, name=None):
        objs = append_objects(os.path.join(self.dir, BLOCKOUT), set(self.plan["slots"][self.name]["objects"]))
        for i, o in enumerate(objs):
            for key in ("f3d_slot", "f3d_context"):
                if key in o:
                    del o[key]
            o.name = (name or self.name) + ("" if i == 0 else ".%03d" % i)
            o.data.materials.clear()
        return objs[0] if len(objs) == 1 else objs


def slot(name=None, objs=None, on=None, note="", review=None):
    """Two uses.

    In a blockout script, declare a slot: `k.slot("hair", shape, on="head", note="long wavy hair, ...")`.
    A blockout is the whole model as a handful of rough, unpainted shapes in the right place at the right size
    (spheres, boxes, k.volume / k.inflate of picture masks). Every shape (or group of shapes) is given to one
    slot; `f3d blockout` then writes a work order per slot, and each slot is filled by a part built on its own.
        name: lowercase word (letters, digits, _) such as head, hair, skirt, apron, bow.
        objs: the placeholder shape(s) of this part; for a mirrored pair give both.
        on:   the slot(s) this part sits on or hangs from (hair on head, apron on skirt). Parts are built in
              that order, so a part can lean on the finished part below it.
        note: what the part is and what matters about it: materials and colours, the features it carries, and
              where the drawing cheats (a bow drawn facing the viewer in every view sits sideways in 3D). This
              is all the builder of the part will know beyond the pictures, so say it here.
        review: optional review focus: "face", "hair", "joint", "detail" (an independent visual check)
              or "assembly". Omit for automatic selection from the slot name and build report.

    In a part script (run by `f3d part <name>`), `S = k.slot()` returns the slot being built: S.lo, S.hi,
    S.center, S.size, S.note, S.below (what it sits on), S.neighbours, S.box("front"), S.shape().
    `k.slot("head")` there returns another slot's record (its box and note, to measure against).
    """
    if objs is None:
        act = _state["active"]
        if act is None:
            raise RuntimeError("k.slot() without objects reads the slot of the part being built, which only exists "
                               "under `f3d part <name>`. In a blockout script pass the shape: k.slot(\"head\", shape)")
        return Slot(act["dir"], load_plan(act["dir"]), name or act["name"])
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
        raise ValueError("slot names are lowercase words such as head, hair, skirt_trim (got %r)" % (name,))
    if review is not None and review not in ("face", "hair", "joint", "detail", "assembly"):
        raise ValueError("slot review must be face, hair, joint, detail or assembly")
    items = core.expand(objs)
    if not items:
        raise ValueError("slot %r: give it its placeholder shape(s)" % name)
    for o in items:
        other = o.get("f3d_slot")
        if other and other != name:
            raise ValueError("'%s' is already in slot %r; a shape belongs to one slot" % (o.name, other))
        o["f3d_slot"] = name
    rec = _state["declared"].setdefault(name, dict(on=[], note=""))
    if review is not None:
        rec["review"] = review
    for n in ([on] if isinstance(on, str) else list(on or [])):
        if n == name:
            raise ValueError("slot %r cannot sit on itself" % name)
        if n not in rec["on"]:
            rec["on"].append(n)
    if note:
        rec["note"] = (rec["note"] + " " + note).strip()
    return objs


def assemble(parts="parts", only=None, verbose=True):
    """Put the model together from its parts: every slot of the blockout is filled with its finished part
    (`parts/<name>/part.blend`, saved by `f3d part <name>`); a slot whose part is not made yet shows its grey
    blockout shape, so the model is always whole. Use it in the final build script:

        from forge3d import kit as k
        parts = k.assemble()                 # {"head": [objects], "hair": [...], ...}
        k.export("out/name.glb")

    and run `f3d build model.py --ref refs` as usual (sheet, comparison, eye count).
    parts: the folder `f3d blockout` wrote.  only: build just these slots (a list of names).
    """
    d = find_parts_dir(parts)
    plan = load_plan(d)
    out, missing, stale = {}, [], []
    for name in plan["order"]:
        if only is not None and name not in only:
            continue
        objs, done = load_slot(d, plan, name)
        for o in objs:
            for key in ("f3d_slot", "f3d_context"):
                if key in o:
                    del o[key]
            o["f3d_part"] = name
            if not done:
                o["f3d_placeholder"] = True
                if o.type == "MESH":
                    o.data.materials.clear()
                    o.data.materials.append(placeholder_material())
        out[name] = objs
        if not done:
            missing.append(name)
        else:
            rep = os.path.join(d, name, "report.json")
            if os.path.isfile(rep):
                with open(rep) as f:
                    r = json.load(f)
                if _moved(r.get("slot"), plan["slots"][name]):
                    stale.append(name)
    early = []
    for name in out:
        rep = os.path.join(d, name, "report.json")
        if name in missing or not os.path.isfile(rep):
            continue
        with open(rep) as f:
            r = json.load(f)
        late = [m for m, done in (r.get("built_against") or {}).items()
                if m in plan["slots"][name]["on"] and m not in missing and
                (not done or os.path.getmtime(part_file(d, m)) > os.path.getmtime(part_file(d, name)))]
        if late:
            early.append("%s (before %s)" % (name, ", ".join(late)))
    buried = _buried(out, missing)
    if verbose:
        if early:
            print("[f3d] assemble: supports have changed. Inspect these joints on the assembled model and "
                  "refit any visible gap or interference: " + "; ".join(early))
        for name, inside, share in buried:
            print("[f3d] assemble: %.0f%% of `%s` is inside `%s` and cannot be seen. Build it again on the finished "
                  "`%s` (k.seat or k.drape onto S.below)." % (share * 100, name, inside, inside))
        print("[f3d] assemble: %d part(s) finished%s" % (
            len(out) - len(missing), "" if not missing else "; still blockout shapes (grey): " + ", ".join(missing)))
        if stale:
            print("[f3d] assemble: the blockout moved these slots after their parts were built, build them again "
                  "(f3d part <name>): " + ", ".join(stale))
    return out


def _moved(old, rec, tol=0.012):
    if not old:
        return False
    a = np.array(list(old["lo"]) + list(old["hi"]))
    b = np.array(list(rec["lo"]) + list(rec["hi"]))
    return bool(np.abs(a - b).max() > tol)


def _buried(parts, skip=()):
    """[(part, inside which part, share)] for parts that lie mostly inside another one."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from .qa import _inside
    dg = core.depsgraph()
    geo = {}
    for name, objs in parts.items():
        chunks = [core.world_tris(o, dg) for o in objs if o.type == "MESH"]
        chunks = [(v, t) for v, t in chunks if len(t)]
        if chunks and name not in skip:
            geo[name] = chunks
    out = []
    for name, chunks in geo.items():
        pts = np.concatenate([v for v, _t in chunks])
        if len(pts) > 240:
            pts = pts[np.linspace(0, len(pts) - 1, 240).astype(int)]
        lo, hi = pts.min(0), pts.max(0)
        for other, och in geo.items():
            if other == name:
                continue
            inside = np.zeros(len(pts), bool)
            for v, t in och:
                if np.any(v.min(0) > hi) or np.any(v.max(0) < lo):
                    continue
                tree = BVHTree.FromPolygons(v.tolist(), t.tolist(), all_triangles=True)
                far = float(np.linalg.norm(v.max(0) - v.min(0))) * 4 + 1.0
                inside |= np.array([_inside(tree, p.tolist(), far) for p in pts])
            share = float(inside.mean())
            if share > 0.6:
                out.append((name, other, share))
    return out
