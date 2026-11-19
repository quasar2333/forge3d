"""Shape generators. Every function returns a mesh object.

Common keyword arguments (all optional):
    name   object name
    at     position. Primitives (box, cyl, sphere, capsule, cone, torus, tube, prism): their centre.
           Coordinate-defined shapes (lathe, extrude, loft, sweep, skin, text): an offset added to the
           coordinates you gave, so with at omitted the shape sits exactly where its numbers say.
    base   alternative to `at`: world position of the shape's bottom-centre (no z + h/2 arithmetic)
    rot    XYZ rotation in degrees
    mat    material (from k.mat) or a colour like '#c0392b'
    smooth True (default; smooth shading with edges sharper than 35 degrees kept crisp), False, or an angle
    parent object to parent under (world position is kept)
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon

from . import core

_AXIS_ROT = {
    "Z": Matrix.Identity(4),
    "X": Matrix.Rotation(math.radians(90), 4, "Y"),
    "Y": Matrix.Rotation(math.radians(-90), 4, "X"),
}


# ---------------------------------------------------------------- helpers
def shade(obj, smooth=True, angle=35.0):
    """Smooth shading that keeps edges sharper than `angle` degrees crisp (or flat shading)."""
    obj = core.get(obj)
    me = obj.data
    if not isinstance(me, bpy.types.Mesh):
        return obj
    if smooth is False:
        me.shade_flat() if hasattr(me, "shade_flat") else [setattr(p, "use_smooth", False) for p in me.polygons]
        return obj
    if isinstance(smooth, (int, float)) and not isinstance(smooth, bool):
        angle = float(smooth)
    if hasattr(me, "set_sharp_from_angle"):
        me.shade_smooth()
        me.set_sharp_from_angle(angle=math.radians(angle))
    else:
        for p in me.polygons:
            p.use_smooth = True
        if hasattr(me, "use_auto_smooth"):
            me.use_auto_smooth = True
            me.auto_smooth_angle = math.radians(angle)
    return obj


def bevel_bm(bm, width, segs=3, angle=40.0, profile=0.5):
    """Round every edge of a bmesh sharper than `angle` degrees."""
    if width <= 0:
        return
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    lim = math.radians(angle)
    edges = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > lim]
    if edges:
        bmesh.ops.bevel(bm, geom=edges, offset=width, offset_type="OFFSET", segments=max(1, int(segs)),
                        profile=profile, affect="EDGES", clamp_overlap=True)


def _apply_material(obj, mat):
    if mat is None:
        return
    from . import materials
    materials.assign(obj, mat)


def finish(mesh_or_bm, default_name, name=None, at=None, base=None, rot=None, mat=None, smooth=True,
           parent=None, recenter=False):
    """Turn a bmesh / Mesh into a placed object. Shared tail of every generator."""
    name = name or default_name
    if isinstance(mesh_or_bm, bmesh.types.BMesh):
        me = bpy.data.meshes.new(name)
        bmesh.ops.recalc_face_normals(mesh_or_bm, faces=mesh_or_bm.faces)
        mesh_or_bm.to_mesh(me)
        mesh_or_bm.free()
    else:
        me = mesh_or_bm
        me.name = name
    obj = bpy.data.objects.new(name, me)
    core.link(obj)
    n = len(me.vertices)
    if recenter and n:
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        c = (co.min(0) + co.max(0)) / 2
        me.vertices.foreach_set("co", (co - c).ravel())
        me.update()
        obj.location = Vector(c.tolist())
    if at is not None:
        obj.location = obj.location + Vector(at) if recenter else Vector(at)
    if rot is not None:
        core.place(obj, rot=rot)
    shade(obj, smooth)
    bpy.context.view_layer.update()
    if base is not None:
        bb = core.bbox(obj)
        core.translate(obj, Vector(base) - bb.point("bottom"))
    _apply_material(obj, mat)
    if parent is not None:
        core.parent(obj, parent)
    return obj


def from_pydata(verts, faces, name="Mesh", edges=(), **kw):
    """Mesh object from raw vertex / face lists."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [tuple(e) for e in edges], [tuple(f) for f in faces])
    me.validate()
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    kw.setdefault("recenter", True)
    return finish(bm, name, **kw)


def catmull_rom(points, subdiv):
    """Smooth curve through points (N, D). A repeated point makes a sharp corner. subdiv = samples per span."""
    pts = np.asarray(points, dtype=np.float64)
    if subdiv <= 0 or len(pts) < 3:
        return pts
    runs, start = [], 0
    for i in range(1, len(pts)):
        if np.allclose(pts[i], pts[i - 1]):
            runs.append(pts[start:i])
            start = i
    runs.append(pts[start:])
    out = []
    for run in runs:
        if len(run) < 3:
            seg = run
        else:
            p = np.vstack([2 * run[0] - run[1], run, 2 * run[-1] - run[-2]])
            seg = []
            t = np.linspace(0, 1, subdiv, endpoint=False)[:, None]
            for i in range(1, len(p) - 2):
                p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
                seg.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
            seg = np.vstack(seg + [run[-1][None]])
        if out and np.allclose(out[-1][-1], seg[0]):
            seg = seg[1:]
        if len(seg):
            out.append(seg)
    return np.vstack(out)


def _orient(bm, axis):
    axis = axis.upper()
    if axis not in _AXIS_ROT:
        raise ValueError("axis must be 'X', 'Y' or 'Z', got %r" % (axis,))
    if axis != "Z":
        bmesh.ops.transform(bm, matrix=_AXIS_ROT[axis], verts=bm.verts)


# ---------------------------------------------------------------- primitives
def box(size=(1, 1, 1), bevel=0.0, segs=3, **kw):
    """Box of size (x, y, z); bevel rounds every edge by that radius (a little bevel makes boxes look real)."""
    s = core.vec3(size)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=s, verts=bm.verts)
    bevel_bm(bm, min(bevel, 0.499 * min(s)), segs)
    return finish(bm, "Box", **kw)


def cyl(r=0.5, h=1.0, r2=None, segs=48, bevel=0.0, bevel_segs=3, axis="Z", cap=True, **kw):
    """Cylinder / frustum along `axis`. r = bottom radius, r2 = top radius (defaults to r); bevel rounds the rims."""
    bm = bmesh.new()
    top = r if r2 is None else r2
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=int(segs), radius1=r, radius2=top, depth=h)
    bevel_bm(bm, min(bevel, 0.49 * h, 0.9 * max(min(r, top), 1e-6) if min(r, top) > 0 else bevel), bevel_segs)
    _orient(bm, axis)
    return finish(bm, "Cylinder", **kw)


def cone(r=0.5, h=1.0, segs=48, axis="Z", **kw):
    """Cone with its tip toward +axis."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=int(segs), radius1=r, radius2=0.0, depth=h)
    _orient(bm, axis)
    return finish(bm, "Cone", **kw)


def prism(sides=6, r=0.5, h=1.0, bevel=0.0, axis="Z", **kw):
    """Regular n-sided prism (flat shaded)."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=int(sides), radius1=r, radius2=r, depth=h)
    bevel_bm(bm, bevel, 2)
    _orient(bm, axis)
    kw.setdefault("smooth", 20 if bevel else False)
    return finish(bm, "Prism", **kw)


def sphere(r=0.5, radii=None, segs=48, **kw):
    """Sphere, or an ellipsoid with radii=(rx, ry, rz)."""
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=int(segs), v_segments=max(4, int(segs) // 2), radius=1.0)
    rad = core.vec3(radii if radii is not None else r)
    bmesh.ops.scale(bm, vec=rad, verts=bm.verts)
    return finish(bm, "Sphere", **kw)


def capsule(r=0.25, h=1.0, segs=32, axis="Z", **kw):
    """Capsule of total length h along `axis`."""
    h = max(h, 2 * r)
    half = h / 2 - r
    n = max(4, segs // 4)
    prof = [(r * math.sin(a), -half - r * math.cos(a)) for a in np.linspace(0, math.pi / 2, n + 1)]
    prof += [(r * math.cos(a), half + r * math.sin(a)) for a in np.linspace(0, math.pi / 2, n + 1)]
    bm = _lathe_bm(np.array(prof), int(segs), cap=False)
    _orient(bm, axis)
    return finish(bm, "Capsule", **kw)


def torus(R=0.5, r=0.1, segs=48, tsegs=16, axis="Z", **kw):
    """Torus: R = ring radius, r = tube radius; the hole runs along `axis`."""
    u = np.linspace(0, 2 * math.pi, int(segs), endpoint=False)
    v = np.linspace(0, 2 * math.pi, int(tsegs), endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    x = (R + r * np.cos(vv)) * np.cos(uu)
    y = (R + r * np.cos(vv)) * np.sin(uu)
    z = r * np.sin(vv)
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    faces = []
    ns, nt = int(segs), int(tsegs)
    for i in range(ns):
        for j in range(nt):
            a, b = i * nt + j, ((i + 1) % ns) * nt + j
            c, d = ((i + 1) % ns) * nt + (j + 1) % nt, i * nt + (j + 1) % nt
            faces.append((a, b, c, d))
    bm = _bm_from(verts, faces)
    _orient(bm, axis)
    return finish(bm, "Torus", **kw)


def tube(r=0.5, h=1.0, wall=0.05, segs=48, axis="Z", bevel=0.0, **kw):
    """Hollow cylinder (pipe, ring, cup wall): outer radius r, wall thickness."""
    ri = max(r - wall, 1e-5)
    prof = np.array([(ri, -h / 2), (r, -h / 2), (r, h / 2), (ri, h / 2), (ri, -h / 2)])
    bm = _lathe_bm(prof, int(segs), cap=False)
    bevel_bm(bm, min(bevel, 0.45 * wall, 0.45 * h), 2)
    _orient(bm, axis)
    return finish(bm, "Tube", **kw)


def plane(size=(1, 1), **kw):
    """Flat rectangle in XY facing +Z."""
    sx, sy = (size, size) if isinstance(size, (int, float)) else size
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    bmesh.ops.scale(bm, vec=(sx, sy, 1), verts=bm.verts)
    return finish(bm, "Plane", **kw)


def disc(r=0.5, segs=48, **kw):
    """Flat disc in XY facing +Z."""
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=int(segs), radius=r)
    return finish(bm, "Disc", **kw)


# ---------------------------------------------------------------- profile-driven shapes
def _bm_from(verts, faces):
    bm = bmesh.new()
    bv = [bm.verts.new(tuple(float(c) for c in v)) for v in verts]
    for f in faces:
        try:
            bm.faces.new([bv[i] for i in f])
        except ValueError:
            pass
    bm.verts.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def _lathe_bm(prof, segs, cap=True, eps=1e-7):
    prof = np.asarray(prof, dtype=np.float64)
    closed = len(prof) > 2 and np.allclose(prof[0], prof[-1])
    if closed:
        prof = prof[:-1]
    ang = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    cs, sn = np.cos(ang), np.sin(ang)
    verts, rings = [], []
    for r, z in prof:
        if abs(r) <= eps:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            start = len(verts)
            verts.extend(zip(r * cs, r * sn, [z] * segs))
            rings.append(list(range(start, start + segs)))
    faces = []
    pairs = list(zip(rings[:-1], rings[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
    for a, b in pairs:
        if len(a) == 1 and len(b) == 1:
            continue
        for j in range(segs):
            k = (j + 1) % segs
            if len(a) == 1:
                faces.append((a[0], b[k], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[k], b[0]))
            else:
                faces.append((a[j], a[k], b[k], b[j]))
    if cap and not closed:
        if len(rings[0]) > 1:
            faces.append(tuple(reversed(rings[0])))
        if len(rings[-1]) > 1:
            faces.append(tuple(rings[-1]))
    return _bm_from(verts, faces)


def lathe(profile, segs=64, smooth_profile=0, cap=True, axis="Z", **kw):
    """Revolve a profile around the axis: bottles, vases, wheels, lamps, bowls, chess pieces, domes.

    profile: [(radius, height), ...] from bottom to top. radius 0 closes the shape at the axis.
             Repeat the first point at the end for a hollow ring cross-section (cup walls, tyres).
    smooth_profile: N > 0 runs a smooth curve through the points with N samples per span;
             repeat a point twice to keep a sharp corner there.
    cap: close open ends with flat discs.
    Tip: k.ref()['front'].mask().profile() measures (height, left, right) off a reference picture.
    """
    pts = catmull_rom(profile, int(smooth_profile))
    pts = np.column_stack([np.maximum(pts[:, 0], 0.0), pts[:, 1]])
    bm = _lathe_bm(pts, int(segs), cap=cap)
    _orient(bm, axis)
    kw.setdefault("recenter", True)
    return finish(bm, "Lathe", **kw)


def _signed_area(loop):
    a = np.asarray(loop, dtype=np.float64)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _clean_loop(loop, eps=1e-9):
    pts = [tuple(map(float, p[:2])) for p in loop]
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < eps and abs(pts[0][1] - pts[-1][1]) < eps:
        pts = pts[:-1]
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    return out


_PLANES = {
    # plane: (u axis, v axis, depth axis, depth sign so the shape's face looks toward the viewer of that plane)
    "XZ": (0, 2, 1),   # front-view outline, depth along Y
    "XY": (0, 1, 2),   # top-view outline, depth along Z
    "YZ": (1, 2, 0),   # side-view outline, depth along X
}


def extrude(outline, depth=0.1, plane="XZ", holes=(), bevel=0.0, bevel_segs=2, smooth_outline=0, **kw):
    """Extrude a 2D outline into a solid: logos, plates, blades, keys, gears, brackets, letters, flat limbs.

    outline: [(u, v), ...] polygon in the given plane ('XZ' = as seen from the front, 'XY' = from the top,
             'YZ' = from the side). holes: list of inner polygons.
    depth: thickness, centred on the plane.  bevel: rounds the rim (soft, moulded look).
    smooth_outline: N > 0 runs a smooth closed curve through the points.
    Tip: k.ref()['front'].mask(regions=[...]).outline() traces an outline off a reference picture.
    """
    if plane.upper() not in _PLANES:
        raise ValueError("plane must be 'XZ', 'XY' or 'YZ'")
    ua, va, da = _PLANES[plane.upper()]
    loops = []
    for loop in [outline] + list(holes):
        pts = _clean_loop(loop)
        if smooth_outline and len(pts) >= 3:
            ring = np.array(pts + pts[:3])
            sm = catmull_rom(ring, int(smooth_outline))
            n = int(smooth_outline)
            pts = [tuple(p) for p in sm[n:n + len(pts) * n]]
        if len(pts) < 3:
            raise ValueError("outline needs at least 3 distinct points")
        loops.append(pts)
    if _signed_area(loops[0]) < 0:
        loops[0] = loops[0][::-1]
    for i in range(1, len(loops)):
        if _signed_area(loops[i]) > 0:
            loops[i] = loops[i][::-1]
    flat = [p for loop in loops for p in loop]
    tris = tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in loop] for loop in loops])
    n = len(flat)

    def v3(p, d):
        c = [0.0, 0.0, 0.0]
        c[ua], c[va], c[da] = p[0], p[1], d
        return tuple(c)

    verts = [v3(p, -depth / 2) for p in flat] + [v3(p, depth / 2) for p in flat]
    faces = [tuple(t) for t in tris] + [tuple(i + n for i in reversed(t)) for t in tris]
    start = 0
    for loop in loops:
        m = len(loop)
        for i in range(m):
            a, b = start + i, start + (i + 1) % m
            faces.append((a, b, b + n, a + n))
        start += m
    bm = _bm_from(verts, faces)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(0.5), verts=bm.verts, edges=bm.edges)
    bevel_bm(bm, min(bevel, 0.49 * depth), bevel_segs)
    kw.setdefault("recenter", True)
    return finish(bm, "Extrude", **kw)


def _superellipse(n_pts, w, h, n):
    t = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    e = 2.0 / max(float(n), 0.2)
    return (w / 2) * np.sign(c) * np.abs(c) ** e, (h / 2) * np.sign(s) * np.abs(s) ** e


def loft(sections, axis="Z", segs=48, interp=0, cap=True, **kw):
    """Skin a surface through cross-sections along an axis: bodies, hulls, fuselages, torsos, bottles, shoes.

    sections: list of dicts (or tuples in the same order) ordered along the axis:
        t    position along the axis
        w, h size of the cross-section along the two other axes
             (axis 'Z': w = X, h = Y;  axis 'Y': w = X, h = Z;  axis 'X': w = Y, h = Z)
        n    roundness exponent: 2 = ellipse (default), 4-6 = rounded box, 1 = diamond
        off  (ou, ov) offset of the section centre along the same two axes
    A section with w = h = 0 closes the end in a point.
    interp: N > 0 adds N smoothly interpolated sections per span (a few control sections become a smooth body).
    """
    rows = []
    for s in sections:
        if isinstance(s, dict):
            off = s.get("off", (0.0, 0.0))
            rows.append((s["t"], s.get("w", 0.0), s.get("h", s.get("w", 0.0)), s.get("n", 2.0), off[0], off[1]))
        else:
            s = tuple(s)
            off = s[4] if len(s) > 4 else (0.0, 0.0)
            rows.append((s[0], s[1], s[2] if len(s) > 2 else s[1], s[3] if len(s) > 3 else 2.0, off[0], off[1]))
    arr = np.array(rows, dtype=np.float64)
    if len(arr) < 2:
        raise ValueError("loft needs at least 2 sections")
    if interp:
        arr = catmull_rom(arr, int(interp))
        arr[:, 1:3] = np.maximum(arr[:, 1:3], 0.0)
        arr[:, 3] = np.maximum(arr[:, 3], 0.3)
    axis = axis.upper()
    idx = {"Z": (0, 1, 2), "Y": (0, 2, 1), "X": (1, 2, 0)}[axis]
    segs = int(segs)
    verts, rings = [], []
    for t, w, h, n, ou, ov in arr:
        if w <= 1e-9 and h <= 1e-9:
            c = [0.0, 0.0, 0.0]
            c[idx[0]], c[idx[1]], c[idx[2]] = ou, ov, t
            rings.append([len(verts)])
            verts.append(tuple(c))
            continue
        u, v = _superellipse(segs, w, h, n)
        start = len(verts)
        for j in range(segs):
            c = [0.0, 0.0, 0.0]
            c[idx[0]], c[idx[1]], c[idx[2]] = ou + u[j], ov + v[j], t
            verts.append(tuple(c))
        rings.append(list(range(start, start + segs)))
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        for j in range(segs):
            k = (j + 1) % segs
            if len(a) == 1:
                faces.append((a[0], b[k], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[k], b[0]))
            else:
                faces.append((a[j], a[k], b[k], b[j]))
    if cap:
        if len(rings[0]) > 1:
            faces.append(tuple(reversed(rings[0])))
        if len(rings[-1]) > 1:
            faces.append(tuple(rings[-1]))
    bm = _bm_from(verts, faces)
    kw.setdefault("recenter", True)
    return finish(bm, "Loft", **kw)


def sweep(path, r=0.02, radii=None, segs=16, smooth_path=0, cap=True, closed=False, squash=1.0, **kw):
    """Tube along a 3D path: handles, cables, pipes, horns, tails, tentacles, branches, straps, rails.

    path: [(x, y, z), ...].  r: radius, or radii=[...] per path point for tapering (0 ends in a point).
    smooth_path: N > 0 rounds the path with N samples per span.  closed: join the end to the start (rings, loops).
    squash: flatten the cross-section (0.3 = strap / ribbon like).
    """
    ctrl = np.asarray(path, dtype=np.float64)
    if len(ctrl) < 2:
        raise ValueError("sweep needs at least 2 path points")
    rad = np.full(len(ctrl), float(r)) if radii is None else np.asarray(radii, dtype=np.float64)
    if len(rad) != len(ctrl):
        rad = np.interp(np.linspace(0, 1, len(ctrl)), np.linspace(0, 1, len(rad)), rad)
    data = np.column_stack([ctrl, rad])
    if closed:
        data = np.vstack([data, data[:1]])
    if smooth_path:
        if closed:
            n = int(smooth_path)
            ring = np.vstack([data[-3:-1], data, data[1:3]])
            data = catmull_rom(ring, n)[2 * n: 2 * n + (len(ctrl)) * n + 1]
        else:
            data = catmull_rom(data, int(smooth_path))
    pts, rad = data[:, :3], np.maximum(data[:, 3], 0.0)
    if closed:
        pts, rad = pts[:-1], rad[:-1]
    n = len(pts)
    if closed:
        tang = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    else:
        tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0]) if abs(tang[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    nrm = np.cross(tang[0], ref)
    nrm /= np.linalg.norm(nrm)
    normals = [nrm]
    for i in range(1, n):
        v = np.cross(tang[i - 1], tang[i])
        s = np.linalg.norm(v)
        nrm = normals[-1]
        if s > 1e-9:
            ang = math.atan2(s, float(np.dot(tang[i - 1], tang[i])))
            k = v / s
            nrm = nrm * math.cos(ang) + np.cross(k, nrm) * math.sin(ang) + k * np.dot(k, nrm) * (1 - math.cos(ang))
        nrm = nrm - tang[i] * np.dot(nrm, tang[i])
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        normals.append(nrm)
    segs = int(segs)
    ang = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    verts, rings = [], []
    for i in range(n):
        if rad[i] <= 1e-9:
            rings.append([len(verts)])
            verts.append(tuple(pts[i]))
            continue
        b = np.cross(tang[i], normals[i])
        start = len(verts)
        for a in ang:
            verts.append(tuple(pts[i] + rad[i] * (math.cos(a) * normals[i] + squash * math.sin(a) * b)))
        rings.append(list(range(start, start + segs)))
    faces = []
    pairs = list(zip(rings[:-1], rings[1:]))
    if closed:
        pairs.append((rings[-1], rings[0]))
    for a, b in pairs:
        if len(a) == 1 and len(b) == 1:
            continue
        for j in range(segs):
            k = (j + 1) % segs
            if len(a) == 1:
                faces.append((a[0], b[k], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[k], b[0]))
            else:
                faces.append((a[j], a[k], b[k], b[j]))
    if cap and not closed:
        if len(rings[0]) > 1:
            faces.append(tuple(reversed(rings[0])))
        if len(rings[-1]) > 1:
            faces.append(tuple(rings[-1]))
    bm = _bm_from(verts, faces)
    kw.setdefault("recenter", True)
    return finish(bm, "Sweep", **kw)


def skin(nodes, edges=None, subdiv=2, **kw):
    """Organic branching form from a skeleton: creatures, limbs, trees, coral, antlers, cartoon bodies.

    nodes: {name: (x, y, z, radius)} or {name: (x, y, z, rx, ry)} (or a list of such tuples).
    edges: [(a, b), ...] by name / index. Omit to chain the nodes in order.
    The first node is the root. Thick at the torso, thin at the tips reads best.
    """
    if isinstance(nodes, dict):
        names = list(nodes)
        vals = [nodes[k] for k in names]
    else:
        vals = list(nodes)
        names = list(range(len(vals)))
    lookup = {n: i for i, n in enumerate(names)}
    if edges is None:
        edges = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
    me = bpy.data.meshes.new("Skin")
    me.from_pydata([tuple(v[:3]) for v in vals], [(lookup[a], lookup[b]) for a, b in edges], [])
    me.update()
    obj = bpy.data.objects.new("Skin", me)
    core.link(obj)
    obj.modifiers.new("Skin", "SKIN")
    layer = me.skin_vertices[0].data
    for i, v in enumerate(vals):
        layer[i].radius = (v[3], v[4] if len(v) > 4 else v[3])
        layer[i].use_root = i == 0
    sub = obj.modifiers.new("Subdivision", "SUBSURF")
    sub.levels = sub.render_levels = int(subdiv)
    dg = core.depsgraph()
    new = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(me)
    kw.setdefault("recenter", True)
    return finish(new, "Skin", **kw)


def text(body, size=0.2, depth=0.02, bevel=0.0, font=None, facing="front", **kw):
    """3D text as a mesh. facing: 'front' (upright, readable from the front) or 'top' (lying flat).
    font: path to a .ttf/.otf for scripts the built-in font lacks (e.g. Chinese)."""
    cu = bpy.data.curves.new("Text", "FONT")
    cu.body = body
    cu.size = size
    cu.extrude = depth / 2
    cu.bevel_depth = bevel
    cu.bevel_resolution = 2
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    if font:
        cu.font = bpy.data.fonts.load(font)
    tmp = bpy.data.objects.new("Text", cu)
    core.link(tmp)
    dg = core.depsgraph()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp, do_unlink=True)
    bpy.data.curves.remove(cu)
    if facing == "front":
        me.transform(Matrix.Rotation(math.radians(90), 4, "X"))
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=max(1e-6, size * 1e-4))
    me = bm
    kw.setdefault("recenter", True)
    kw.setdefault("smooth", 30)
    return finish(me, "Text", **kw)
