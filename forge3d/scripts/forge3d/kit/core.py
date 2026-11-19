"""Scene, object, bounding-box and placement helpers. Needs only bpy + numpy."""
import math
import re
from contextlib import contextmanager

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

_state = {"collection": None}
GEOM_TYPES = {"MESH", "CURVE", "FONT", "SURFACE", "META"}

NAMED_COLORS = {
    "white": "#ffffff", "black": "#000000", "red": "#d62828", "green": "#2a9d4b", "blue": "#2563eb",
    "yellow": "#f6c90e", "orange": "#f77f00", "purple": "#7b2cbf", "pink": "#ff8fab", "brown": "#7f5539",
    "gray": "#808080", "grey": "#808080", "cyan": "#22d3ee", "magenta": "#d946ef", "navy": "#1d3557",
    "teal": "#14b8a6", "beige": "#e8d8b8", "cream": "#fff3d6", "gold": "#ffc355", "silver": "#c8ccd0",
}


# ---------------------------------------------------------------- paths
def script_dir():
    """Folder of the running build script (the working directory when there is none)."""
    import os
    import sys
    a = sys.argv[0] if sys.argv else ""
    return os.path.dirname(os.path.abspath(a)) if a and a.endswith(".py") and os.path.isfile(a) else os.getcwd()


def in_path(path):
    """Resolve an input file: relative paths are looked up next to the build script first, then in the cwd."""
    import os
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    cand = os.path.join(script_dir(), path)
    return cand if os.path.exists(cand) else os.path.abspath(path)


def out_path(path):
    """Resolve an output file: relative paths are written next to the build script."""
    import os
    path = os.path.expanduser(path)
    path = path if os.path.isabs(path) else os.path.join(script_dir(), path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------- colours
def srgb_to_linear(c):
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c):
    c = max(0.0, float(c))
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


def rgba(color, alpha=1.0):
    """Colour -> linear RGBA. Hex strings and names are sRGB; float tuples are taken as linear."""
    if isinstance(color, str):
        s = NAMED_COLORS.get(color.strip().lower(), color).strip().lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) not in (6, 8) or re.fullmatch(r"[0-9a-fA-F]+", s) is None:
            raise ValueError("bad colour %r: use '#rrggbb', a colour name, or a linear (r, g, b) tuple" % (color,))
        vals = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
        out = [srgb_to_linear(v) for v in vals[:3]]
        return (out[0], out[1], out[2], vals[3] if len(vals) == 4 else alpha)
    c = tuple(float(v) for v in color)
    if max(c[:3]) > 1.0001:
        c = tuple(srgb_to_linear(v / 255.0) for v in c[:3]) + c[3:]
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else alpha)


def hex_of(color):
    r, g, b, _ = rgba(color)
    return "#%02x%02x%02x" % tuple(int(round(255 * min(1.0, linear_to_srgb(v)))) for v in (r, g, b))


# ---------------------------------------------------------------- scene
def reset():
    """Start from an empty metric scene. (In a live GUI session this only deletes objects.)"""
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    else:
        for o in list(bpy.data.objects):
            bpy.data.objects.remove(o, do_unlink=True)
        for coll in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
            for d in list(coll):
                if d.users == 0:
                    coll.remove(d)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1.0
    _state["collection"] = None
    from . import ironing, partkit
    ironing.clear()
    partkit.reset()
    return sc


@contextmanager
def collection(name):
    """`with k.collection("Wheels"):` puts objects created inside into that collection."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    prev = _state["collection"]
    _state["collection"] = coll
    try:
        yield coll
    finally:
        _state["collection"] = prev


def link(obj):
    (_state["collection"] or bpy.context.scene.collection).objects.link(obj)
    return obj


def depsgraph():
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def get(obj):
    """Accept an object or its name."""
    if isinstance(obj, str):
        o = bpy.data.objects.get(obj)
        if o is None:
            raise KeyError("no object named %r (have: %s)" % (obj, ", ".join(sorted(x.name for x in bpy.data.objects)[:40])))
        return o
    return obj


def descendants(obj):
    out = []
    for ch in obj.children:
        out.append(ch)
        out.extend(descendants(ch))
    return out


def expand(objs=None, geometry_only=True):
    """Objects (or names) -> flat unique list including all descendants. None -> whole scene."""
    if objs is None:
        items = list(bpy.context.scene.objects)
    else:
        if isinstance(objs, (bpy.types.Object, str)):
            objs = [objs]

        def flat(seq):                                   # [head, [bow_l, bow_r]] is as good as a flat list
            for o in seq:
                if isinstance(o, (list, tuple, set)):
                    yield from flat(o)
                elif o is not None:
                    yield o
        items = []
        for o in flat(objs):
            o = get(o)
            items.append(o)
            items.extend(descendants(o))
    seen, out = set(), []
    for o in items:
        if o.name in seen:
            continue
        seen.add(o.name)
        if geometry_only and o.type not in GEOM_TYPES:
            continue
        out.append(o)
    return out


def roots(objs):
    """Members of objs whose parent is not also in objs (move these and the rest follows)."""
    objs = [get(o) for o in ([objs] if isinstance(objs, (bpy.types.Object, str)) else objs)]
    names = {o.name for o in objs}
    out = []
    for o in objs:
        p = o.parent
        while p is not None and p.name not in names:
            p = p.parent
        if p is None:
            out.append(o)
    return out


def scene_objects(include_helpers=False, ink=False):
    """Model geometry in the scene, skipping stage helpers (ground, backdrop) made by k.stage(), while one part
    is being built the neighbouring parts shown for context, and the line art made by k.outline (ink=True
    includes it: it is part of what the model looks like, not of what it is made of)."""
    return [o for o in bpy.context.scene.objects
            if o.type in GEOM_TYPES and (include_helpers or not (o.get("f3d_helper") or o.get("f3d_context")))
            and (ink or not o.get("f3d_ink")) and not o.hide_render]


# ---------------------------------------------------------------- geometry queries
def world_verts(obj, dg=None):
    """World-space vertices (N, 3) of the evaluated object (modifiers applied)."""
    obj = get(obj)
    if obj.type not in GEOM_TYPES:
        return np.zeros((0, 3), np.float32)
    dg = dg or depsgraph()
    ev = obj.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return np.zeros((0, 3), np.float32)
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    if n:
        me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    m = np.array(ev.matrix_world, dtype=np.float64)
    return (co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)


def world_tris(obj, dg=None):
    """(verts (N,3), tris (M,3)) of the evaluated object in world space."""
    obj = get(obj)
    dg = dg or depsgraph()
    ev = obj.evaluated_get(dg)
    try:
        me = ev.to_mesh()
    except RuntimeError:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    me.calc_loop_triangles()
    n, t = len(me.vertices), len(me.loop_triangles)
    co = np.empty(n * 3, np.float32)
    tri = np.empty(t * 3, np.int32)
    if n:
        me.vertices.foreach_get("co", co)
    if t:
        me.loop_triangles.foreach_get("vertices", tri)
    ev.to_mesh_clear()
    m = np.array(ev.matrix_world, dtype=np.float64)
    v = (co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]).astype(np.float32)
    return v, tri.reshape(-1, 3)


class BBox:
    """Axis-aligned world bounding box. `.min .max .size .center` are Vectors; `.point(anchor)` resolves anchors."""

    def __init__(self, lo, hi):
        self.min, self.max = Vector(lo), Vector(hi)

    @property
    def size(self):
        return self.max - self.min

    @property
    def center(self):
        return (self.min + self.max) / 2

    @property
    def diag(self):
        return self.size.length

    def point(self, spec="center"):
        v, s, c = anchor_vec(spec), self.size, self.center
        return Vector((c.x + v[0] * s.x / 2, c.y + v[1] * s.y / 2, c.z + v[2] * s.z / 2))

    def __repr__(self):
        f = lambda v: "(%.4g, %.4g, %.4g)" % tuple(v)
        return "BBox(min=%s, max=%s, size=%s)" % (f(self.min), f(self.max), f(self.size))


def own(obj):
    """What an object's anchors refer to: its own mesh, or for an empty (group) all of its descendants."""
    obj = get(obj)
    return [obj] if obj.type in GEOM_TYPES else expand(obj)


def bbox(objs=None, children=True):
    """World bounding box of object(s) incl. modifiers (and children unless children=False).
    None -> all model geometry."""
    if objs is None:
        items = scene_objects()
    elif children:
        items = expand(objs)
    else:
        items = [get(o) for o in ([objs] if isinstance(objs, (bpy.types.Object, str)) else objs)]
    dg = depsgraph()
    lo = np.array([np.inf] * 3)
    hi = -lo
    for o in items:
        v = world_verts(o, dg)
        if len(v):
            lo = np.minimum(lo, v.min(0))
            hi = np.maximum(hi, v.max(0))
    if not np.isfinite(lo).all():
        base = get(objs if isinstance(objs, (bpy.types.Object, str)) else objs[0]) if objs else None
        p = base.matrix_world.translation if base is not None else Vector((0, 0, 0))
        return BBox(p, p)
    return BBox(lo, hi)


_ANCHOR_TOKENS = {
    "top": (2, 1), "up": (2, 1), "bottom": (2, -1), "down": (2, -1), "left": (0, -1), "right": (0, 1),
    "front": (1, -1), "back": (1, 1), "+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1),
    "+z": (2, 1), "-z": (2, -1), "center": None, "centre": None, "middle": None, "mid": None,
}


def anchor_vec(spec):
    """'top', 'front-left', '+x', (0, -1, 0.5) ... -> factors in [-1, 1] per axis.

    left/right are the VIEWER's left/right in front view: left = -X, right = +X. front = -Y, back = +Y.
    """
    if spec is None:
        return (0.0, 0.0, 0.0)
    if isinstance(spec, str):
        v = [0.0, 0.0, 0.0]
        tokens = re.findall(r"[+-][xyz]|[a-z]+", spec.lower())
        if not tokens:
            raise ValueError("bad anchor %r" % (spec,))
        for t in tokens:
            if t not in _ANCHOR_TOKENS:
                raise ValueError("bad anchor token %r in %r (use top/bottom/left/right/front/back/center, +x..-z, "
                                 "or an (x, y, z) tuple of -1..1 factors)" % (t, spec))
            hit = _ANCHOR_TOKENS[t]
            if hit:
                v[hit[0]] = float(hit[1])
        return tuple(v)
    v = tuple(float(x) for x in spec)
    if len(v) != 3:
        raise ValueError("anchor tuple must have 3 factors, got %r" % (spec,))
    return v


def raycast(obj, origin, direction, dg=None):
    """Ray against one evaluated object, world space. -> (point, normal) or None."""
    obj = get(obj)
    if obj.type != "MESH":
        return None
    dg = dg or depsgraph()
    ev = obj.evaluated_get(dg)
    mw = ev.matrix_world
    inv = mw.inverted_safe()
    o = inv @ Vector(origin)
    d = (inv.to_3x3() @ Vector(direction))
    if d.length < 1e-12:
        return None
    d.normalize()
    ok, loc, nor, _ = ev.ray_cast(o, d)
    if not ok:
        return None
    n = (inv.to_3x3().transposed() @ nor).normalized()
    return mw @ loc, n


def raycast_all(objs, origin, direction, children=True):
    """Nearest hit over object(s) (incl. children by default). -> (point, normal, obj) or None."""
    dg = depsgraph()
    best = None
    origin = Vector(origin)
    if children:
        items = expand(objs)
    else:
        items = [get(o) for o in ([objs] if isinstance(objs, (bpy.types.Object, str)) else objs)]
    for o in items:
        hit = raycast(o, origin, direction, dg)
        if hit and (best is None or (hit[0] - origin).length < (best[0] - origin).length):
            best = (hit[0], hit[1], o)
    return best


def surface_point(obj, direction=None, at=None):
    """Point on the surface of obj plus the outward normal there.

    at: anchor spec. A face anchor such as 'front', 'top' or (0.3, -1, 0.2) is projected straight along that
        axis, so (0.3, -1, 0.2) is the spot seen 30 % right of centre and 20 % above the middle in front view.
        Edge / corner anchors ('top-front', (1, -1, 1)) are projected toward the object's centre.
    direction: alternatively a world direction from the object's centre toward the spot, e.g. (0.3, -1, 0.4).
    Uses the object's own mesh (children are ignored unless obj is an empty).
    """
    items = own(obj)
    bb = bbox(items, children=False)
    c = bb.center
    pad = bb.diag * 0.05 + 1e-4
    rays = []
    if direction is not None:
        d = Vector(direction)
        if d.length < 1e-9:
            raise ValueError("direction must be non-zero")
        d.normalize()
        rays.append((c + d * (bb.diag * 1.5 + 1e-4), d))
        fallback = c + d * 0.5 * bb.diag
    else:
        v = anchor_vec(at)
        start = bb.point(v)
        fallback = start
        faces = [i for i in range(3) if abs(abs(v[i]) - 1.0) < 1e-6]
        if len(faces) == 1:
            d = Vector((0, 0, 0))
            d[faces[0]] = 1.0 if v[faces[0]] > 0 else -1.0
            rays.append((start + d * pad, d))
        d = start - c
        if d.length < 1e-9:
            if not rays:
                return c.copy(), Vector((0, 0, 1))
        else:
            d.normalize()
            rays.append((start + d * pad, d))
    for start, d in rays:
        hit = raycast_all(items, start, -d, children=False)
        if hit is not None:
            n = hit[1]
            if n.dot(d) < 0:
                n = -n
            return hit[0], n
    return fallback, rays[0][1]


def anchor(obj, spec="center", surface=False):
    """World position of an anchor on obj's bounding box, optionally projected onto the real surface.
    Refers to the object's own mesh; for an empty (group) to all of its children."""
    if surface and any(abs(x) > 1e-9 for x in anchor_vec(spec)):
        return surface_point(obj, at=spec)[0]
    return bbox(own(obj), children=False).point(spec)


# ---------------------------------------------------------------- transforms
def vec3(v, default=None):
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return Vector((v, v, v))
    return Vector(v)


def place(obj, at=None, rot=None, scale=None):
    """Set location (object origin), rotation (XYZ degrees) and/or scale."""
    obj = get(obj)
    if at is not None:
        obj.location = Vector(at)
    if rot is not None:
        obj.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ")
    if scale is not None:
        obj.scale = vec3(scale)
    return obj


def translate(objs, offset):
    """Move object(s) by a world-space offset (children follow their parents)."""
    off = Vector(offset)
    bpy.context.view_layer.update()
    for o in roots(expand(objs, geometry_only=False)):
        m = o.matrix_world.copy()
        m.translation = m.translation + off
        o.matrix_world = m
    bpy.context.view_layer.update()
    return objs


def rotate(objs, rot=(0, 0, 0), about=None):
    """Rotate object(s) by XYZ degrees in world space about a pivot (default: their bbox centre)."""
    pivot = Vector(about) if about is not None else bbox(objs).center
    r = Euler([math.radians(a) for a in rot], "XYZ").to_matrix().to_4x4()
    t = Matrix.Translation(pivot) @ r @ Matrix.Translation(-pivot)
    bpy.context.view_layer.update()
    for o in roots(expand(objs, geometry_only=False)):
        o.matrix_world = t @ o.matrix_world
    bpy.context.view_layer.update()
    return objs


def resize(objs, factor, about=None):
    """Scale object(s) in world space about a pivot (default: bottom centre of their bbox)."""
    pivot = Vector(about) if about is not None else bbox(objs).point("bottom")
    f = vec3(factor)
    s = Matrix.Diagonal((f.x, f.y, f.z, 1.0))
    t = Matrix.Translation(pivot) @ s @ Matrix.Translation(-pivot)
    bpy.context.view_layer.update()
    for o in roots(expand(objs, geometry_only=False)):
        o.matrix_world = t @ o.matrix_world
    bpy.context.view_layer.update()
    return objs


def ground(objs=None, z=0.0):
    """Shift object(s) vertically so their lowest point rests at height z."""
    items = scene_objects() if objs is None else objs
    if not items:
        return objs
    translate(items, (0, 0, z - bbox(items).min.z))
    return objs


def fit_height(objs=None, height=1.0):
    """Uniformly scale the model so it is `height` tall, standing on z=0, centred in X/Y."""
    items = scene_objects() if objs is None else objs
    bb = bbox(items)
    if bb.size.z > 1e-9:
        resize(items, height / bb.size.z, about=bb.point("bottom"))
    bb = bbox(items)
    translate(items, (-bb.center.x, -bb.center.y, -bb.min.z))
    return objs


def parent(child, to, keep_transform=True):
    child, to = get(child), get(to)
    bpy.context.view_layer.update()
    mw = child.matrix_world.copy()
    child.parent = to
    if keep_transform:
        child.matrix_parent_inverse = to.matrix_world.inverted_safe()
        child.matrix_world = mw
    bpy.context.view_layer.update()
    return child


def group(name, objs, at=None):
    """Parent objects under a new empty (pivot at `at`, default bottom centre of their bbox). Returns the empty."""
    objs = [get(o) for o in objs]
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.05
    link(e)
    e.location = Vector(at) if at is not None else bbox(objs).point("bottom")
    bpy.context.view_layer.update()
    for o in roots(objs):
        parent(o, e)
    return e


def attach(child, to, at="top", child_at=None, offset=(0, 0, 0), embed=0.0, surface=True, parent_to=True):
    """Move `child` so that its `child_at` anchor meets the `at` anchor of `to`.

    at / child_at: anchor specs ('top', 'front-left', (x, y, z) factors). child_at defaults to the opposite of `at`.
    surface: project both anchors onto the real mesh surfaces (matters for round shapes).
    embed: sink the child into the parent by this fraction of the child's size along the joint direction
           (0.1-0.3 gives organic joints a solid overlap; keep 0 for exact stacking).
    offset: extra world-space shift applied afterwards.
    """
    child, to = get(child), get(to)
    av = anchor_vec(at)
    cv = anchor_vec(child_at) if child_at is not None else tuple(-x for x in av)
    target = anchor(to, av, surface=surface)
    source = anchor(child, cv, surface=surface)
    delta = target - source + Vector(offset)
    if embed:
        d = Vector(av)
        if d.length > 1e-9:
            d.normalize()
            size = bbox(own(child), children=False).size
            delta -= d * embed * abs(size.x * d.x + size.y * d.y + size.z * d.z)
    translate(child, delta)
    if parent_to and child.parent is None:
        parent(child, to)
    return child


def stick(child, to, at=None, direction=None, axis="Z", embed=0.0, spin=0.0, child_at=None, parent_to=True):
    """Place `child` on the surface of `to` and orient it along the surface normal.

    at: where on the parent, as bounding-box factors: (0.3, -1, 0.2) = on the front, 30 % right of centre and
        20 % above the middle, exactly as it appears in a front view. Names work too: 'top', 'front-left'.
    direction: alternatively a world direction from the parent's centre toward the spot, e.g. (0.3, -1, 0.4).
    axis: the child's local axis that should point away from the surface ('Z', '-Z', 'Y', '-Y', 'X', '-X').
    child_at: which point of the child touches the surface (default: the face opposite `axis`).
    embed: fraction of the child's extent along the normal to sink into the parent.
    spin: extra rotation (degrees) around the normal.
    """
    child, to = get(child), get(to)
    if at is None and direction is None:
        raise ValueError("stick needs at=(x, y, z) bounding-box factors (or direction=)")
    p, n = surface_point(to, direction=direction, at=at)
    sign = -1.0 if axis.startswith("-") else 1.0
    ax = axis.lstrip("+-").upper()
    local = Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[ax]) * sign
    q = local.rotation_difference(n)
    if spin:
        q = Matrix.Rotation(math.radians(spin), 3, n).to_quaternion() @ q
    bpy.context.view_layer.update()
    if child.parent is not None:
        mw = child.matrix_world.copy()
        child.parent = None
        child.matrix_world = mw
    loc = child.matrix_world.translation.copy()
    child.rotation_euler = q.to_euler("XYZ")
    child.location = loc
    bpy.context.view_layer.update()
    cv = Vector(anchor_vec(child_at)) if child_at is not None else None
    verts = world_verts(child)
    if len(verts) == 0:
        contact = child.matrix_world.translation.copy()
        extent = 0.0
    else:
        nn = np.array(n)
        proj = verts @ nn
        extent = float(proj.max() - proj.min())
        if cv is not None:
            contact = anchor(child, tuple(cv), surface=True)
        else:
            centre = Vector(verts.mean(0).tolist())
            contact = centre + n * (float(proj.min()) - centre.dot(n))
    translate(child, p - contact - n * (embed * extent))
    if parent_to:
        parent(child, to)
    return child


def dup(obj, name=None, linked=False):
    """Copy an object (and its children). linked=True shares mesh data (cheap instances)."""
    obj = get(obj)

    def _copy(o, par):
        c = o.copy()
        if o.data is not None and not linked:
            c.data = o.data.copy()
        link(c)
        if par is not None:
            c.parent = par
            c.matrix_parent_inverse = o.matrix_parent_inverse.copy()
        for ch in o.children:
            _copy(ch, c)
        return c

    c = _copy(obj, None)
    if obj.parent is not None:
        c.parent = obj.parent
        c.matrix_parent_inverse = obj.matrix_parent_inverse.copy()
    if name:
        c.name = name
        if c.data is not None and not linked:
            c.data.name = name
    bpy.context.view_layer.update()
    return c


def delete(objs):
    """Remove object(s) (and their children) from the scene."""
    for o in expand(objs, geometry_only=False):
        bpy.data.objects.remove(o, do_unlink=True)


def describe(objs=None):
    """One line per object: name, size, centre. Handy to print while debugging placement."""
    lines = []
    for o in (scene_objects() if objs is None else expand(objs)):
        bb = bbox(o)
        lines.append("%-28s size=(%.3f, %.3f, %.3f) centre=(%.3f, %.3f, %.3f) z=[%.3f, %.3f]" % (
            o.name, *bb.size, *bb.center, bb.min.z, bb.max.z))
    text = "\n".join(lines)
    print(text)
    return text
