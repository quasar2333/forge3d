"""Layers in build scripts: a subject taken apart with `f3d layers`, every layer generated on its own."""
import os

from . import core, refkit
from .adopt import adopt as _adopt
from .conform import conform as _conform
from .paint import paint_from_ref as _paint_from_ref


def layer_ref(name, ref=None):
    """The reference pack of one layer: `<pack>_<name>` beside the whole-picture pack."""
    base = refkit.as_ref(ref) if ref is not None else refkit.ref()
    d = base.dir.rstrip(os.sep) + "_" + name
    if not os.path.isfile(os.path.join(d, "ref.json")):
        raise FileNotFoundError("no layer pack %s (make it with: f3d layers %s %s=<picture of that layer alone>)" % (
            d, os.path.basename(base.dir), name))
    return refkit.ref(d)


def seat(outer, inner, gap=0.004, reach=0.12, verbose=True):
    """Make `outer` lie on top of `inner` instead of cutting through it: every vertex of `outer` that is inside
    `inner` (or closer to its surface than `gap`) is pushed out to just above that surface, and its surroundings
    follow smoothly. For layers generated separately: a wig on a head, a coat over a body, a hat on hair.

    gap: clearance kept above the inner surface (world units).
    reach: vertices deeper inside than this are left alone (they belong to something that really passes
           through, such as hair hanging behind a body).
    Returns the number of vertices moved."""
    import numpy as np
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from scipy.sparse import coo_matrix
    outers = [o for o in core.expand(outer) if o.type == "MESH"]
    inners = [o for o in core.expand(inner) if o.type == "MESH"]
    if not outers or not inners:
        raise ValueError("seat: needs a mesh object (or several) for both the outer and the inner layer")
    if len(outers) > 1 or len(inners) > 1:
        return sum(seat(o, i, gap, reach, verbose) for i in inners for o in outers)
    outer, inner = outers[0], inners[0]
    dg = core.depsgraph()
    V, T = core.world_tris(inner, dg)
    tree = BVHTree.FromPolygons([Vector(p) for p in V.tolist()], T.tolist())
    me = outer.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    M = np.array(outer.matrix_world, dtype=np.float64)
    P = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    A = coo_matrix((np.ones(len(e) * 2, np.float32), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)[:, None]

    def wanted(P, keep_order):
        need = np.zeros((n, 3))
        hit = np.zeros(n, bool)
        depth = np.zeros(n)
        foot = np.zeros((n, 3))
        normal = np.zeros((n, 3))
        for i in range(n):
            loc, nor, _idx, dist = tree.find_nearest(Vector(P[i]), reach)
            if loc is None:
                continue
            d = (P[i] - np.array(loc)) @ np.array(nor)
            if d < gap:
                hit[i], depth[i], foot[i], normal[i] = True, d, loc, nor
        if hit.any():
            # a lock that lies inside the layer below has two sides; lifted to one and the same height they
            # would flicker through each other. What was deeper stays deeper: the penetration is squeezed
            # into a thin shell above the surface instead of flattened onto it
            lift = gap + (0.15 * (depth[hit] - depth[hit].min()) if keep_order else 0.0)
            need[hit] = foot[hit] + normal[hit] * np.reshape(lift, (-1, 1)) - P[hit]
        return need, hit

    need, hit = wanted(P, True)
    moved = int(hit.sum())
    if moved:
        d = need.copy()
        for _ in range(8):                                  # the surroundings follow, fading out over a few rings
            d = np.where(hit[:, None], need, (A @ d) / deg)
        P = P + d
        need2, hit2 = wanted(P, False)                      # smoothing never pulls a vertex back in; make sure
        P[hit2] += need2[hit2]
        inv = np.linalg.inv(M)
        me.vertices.foreach_set("co", (P @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel())
        me.update()
    if verbose:
        print("[f3d] seat(%s on %s): %d of %d vertices lifted out of the layer below" % (outer.name, inner.name, moved, n))
    return moved


def _from_outlines(L, name):
    """A layer's shape straight from its own pictures: a rounded solid with the layer's front and side outline,
    or the front outline puffed up when there is no side picture."""
    from . import sculpt
    front = next((L[n] for n in ("front", "back") if n in L), None)
    side = next((L[n] for n in ("right", "left") if n in L), None)
    if front is None:
        raise ValueError("layer %r has no front or back picture to take its shape from" % name)
    if side is None:
        return sculpt.inflate(front.mask(), name=name)
    span = max(front.height, side.height)
    return sculpt.volume(front.mask(), side.mask(), res=int(min(240, max(72, span / 0.003))), smooth=2, name=name)


def layer(name, mesh=None, ref=None, over=None, conform=False, paint=False, faces=150000, yaw=0, **paint_options):
    """The mesh of one layer of a subject that was taken apart with `f3d layers` (a wig of hair, a bare body, a
    hat): generated from the layer's own pictures by `f3d gen3d`, cleaned, and placed where those pictures have
    it. Layers share the world frame of the whole picture, so the mesh lands in its slot by itself. This is the
    way to an organic part that cannot be built from simple shapes (hair, fur, a creature's body).

    name: the layer's name as given to `f3d layers` (pack `<refs>_<name>`, mesh `gen/<name>.glb` beside it).
    mesh: another mesh file for this layer (any generator, or a .blend / .glb you made), or "outline": no
          generated mesh, a smooth solid with exactly the layer's front and side outlines instead. That is also
          what you get, with a note, when the mesh was never generated (quota used up).
    over: the object(s) this layer lies on: `k.layer("hair", over=S.below)` lifts the wig out of the head
          wherever the two cut through each other (see k.seat).
    conform: False (default) only stretches the mesh along X, Y and Z to the size the pictures show (see
          k.stretch_to_ref), which keeps it clean. True also warps its outlines onto the picture outlines; that
          creases a generated mesh, so use it only on a smooth one that misses the outline by little.
    paint: False (default): colour it yourself, e.g. `k.paint_part(obj, ["front", "back"], base="gradient",
          ref=k.layer_ref(name))`. True projects all of the layer's pictures (k.paint_from_ref; paint_options
          are forwarded): fuller, but smeared where the pictures disagree.
    yaw: rotation of the mesh about Z in degrees. Meshes from `f3d gen3d` face the front already (0); pass None
         to let the fit try the four quarter turns (meshes from elsewhere).
    Returns the object.
    """
    L = layer_ref(name, ref)
    path = os.path.join(os.path.dirname(L.dir), "gen", name + ".glb") if mesh in (None, "outline") else mesh
    if mesh == "outline" or (mesh is None and not os.path.isfile(core.in_path(path))):
        if mesh is None:
            print("[f3d] layer(%s): no generated mesh at %s; shaped from the layer's own outlines instead (make one "
                  "with `f3d gen3d %s` for a fuller form, or pass mesh=\"outline\" to say this is intended)" % (
                      name, path, os.path.basename(L.dir)))
        obj = _from_outlines(L, name)
        obj["f3d_from_outlines"] = True
    elif not os.path.isfile(core.in_path(path)):
        raise FileNotFoundError("layer %r has no mesh at %s (make it with: f3d gen3d %s)" % (name, path, os.path.basename(L.dir)))
    else:
        obj = _adopt(path, ref=L, name=name, faces=faces, yaw=yaw)
    if conform:
        _conform(obj, ref=L)
    elif not obj.get("f3d_from_outlines"):
        from .adopt import stretch_to_ref as _stretch
        _stretch(obj, L)
    below = [] if over is None else list(over) if isinstance(over, (list, tuple)) else [over]
    for b in below:
        seat(obj, b)
    if paint:
        _paint_from_ref(obj, ref=L, occluders=[obj], name="f3d_layer_" + name, **paint_options)
        from .paint import undercoat as _undercoat
        for b in below:
            _undercoat(b, obj)
    obj["f3d_layer"] = name
    return obj


def drape(objs, onto, view="front", gap=0.003, soften=0.01, verbose=True):
    """Lay a flat-built part onto the surface behind it, as seen from one side: an apron onto a skirt, a fringe
    onto a forehead, an emblem onto a chest, a collar onto shoulders. Every vertex slides along the viewing
    direction until the part's back rests `gap` above `onto`; its outline as seen from that view and its own
    thickness stay exactly as they were, so a part cut out of the front picture (k.inflate, k.extrude) still
    matches that picture afterwards. Paint applied before (k.paint_part) stays in place.

    objs: the part (one object or several, moved as one piece).   onto: the object(s) it lies on.
    view: the side it is seen from: 'front', 'back', 'right', 'left' or 'top'.
    soften: the surface underneath is smoothed over this distance first, so the part does not copy every fold.
    Where the part overhangs the edge of what it lies on, it continues at the depth of the nearest surface.
    """
    import numpy as np
    from scipy import ndimage
    from . import raster
    from .sculpt import _VIEW_FRAMES
    if view not in _VIEW_FRAMES:
        raise ValueError("drape: view must be one of %s" % ", ".join(_VIEW_FRAMES))
    items = [o for o in core.expand(objs) if o.type == "MESH"]
    under = [o for o in core.expand(onto) if o.type == "MESH"]
    if not items or not under:
        raise ValueError("drape: needs mesh object(s) for both the part and what it lies on")
    ua, va, da = (np.array(a, np.float64) for a in _VIEW_FRAMES[view])
    dg = core.depsgraph()
    tris = np.concatenate([(lambda V, T: V[T])(*core.world_tris(o, dg)) for o in under])
    pts = tris.reshape(-1, 3)
    u, v = pts @ ua, pts @ va
    pad = 0.05
    frame = (float(u.min() - pad), float(v.min() - pad), float(u.max() + pad), float(v.max() + pad))
    ppu = min(1500.0, 700.0 / max(frame[2] - frame[0], frame[3] - frame[1]))
    D, _to_px = raster.depth_buffer(tris.astype(np.float32), da, ua, va, frame, ppu)
    hit = np.isfinite(D) & (D > -1e30)
    if not hit.any():
        raise ValueError("drape: nothing of `onto` is visible from the %s" % view)
    idx = ndimage.distance_transform_edt(~hit, return_distances=False, return_indices=True)
    D = D[idx[0], idx[1]]
    if soften > 0:
        D = ndimage.gaussian_filter(D, max(0.5, soften * ppu))
    moved = []
    for o in items:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.array(o.matrix_world, dtype=np.float64)
        moved.append((o, me, M, co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]))
    back = min(float((P @ da).min()) for _o, _m, _M, P in moved)       # the part's own back plane
    h, w = D.shape
    for o, me, M, P in moved:
        x = np.clip(((P @ ua) - frame[0]) * ppu, 0, w - 1.001)
        y = np.clip((frame[3] - (P @ va)) * ppu, 0, h - 1.001)
        surf = ndimage.map_coordinates(D, [y, x], order=1, mode="nearest")
        P = P + np.outer(surf + gap - back, da)
        inv = np.linalg.inv(M)
        me.vertices.foreach_set("co", (P @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel())
        me.update()
    if verbose:
        print("[f3d] drape(%s onto %s, seen from the %s)" % (", ".join(o.name for o in items[:3]),
                                                             ", ".join(o.name for o in under[:3]), view))
    return objs
