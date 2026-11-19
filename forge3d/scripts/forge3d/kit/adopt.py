"""Bring in a mesh made elsewhere (image-to-3D generators, scans, downloaded assets) and fit it to the reference."""
import hashlib
import json
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from . import core, raster, refkit


def _key(path, params):
    st = os.stat(path)
    h = hashlib.sha1(json.dumps([os.path.abspath(path), st.st_size, int(st.st_mtime), params], sort_keys=True,
                                default=str).encode()).hexdigest()[:16]
    return os.path.join(os.path.dirname(os.path.abspath(path)), ".f3d-cache", "%s-%s.blend" % (
        os.path.splitext(os.path.basename(path))[0], h))


def _mesh_arrays(me):
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    return co.reshape(-1, 3), t.reshape(-1, 3)


def drop_floaters(obj, keep=0.02):
    """Delete disconnected bits smaller than `keep` x the largest piece (generator debris). Returns pieces removed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    me = obj.data
    n = len(me.vertices)
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    cnt, lab = connected_components(g, directed=False)
    if cnt <= 1:
        return 0
    co, _ = _mesh_arrays(me)
    size = np.zeros(cnt)
    for c in range(cnt) if cnt < 2000 else []:
        pts = co[lab == c]
        size[c] = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 0.0
    if cnt >= 2000:
        size = np.bincount(lab, minlength=cnt).astype(float)
    bad = np.flatnonzero(size < keep * size.max())
    if len(bad) == 0:
        return 0
    kill = np.isin(lab, bad)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.flatnonzero(kill)], context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    return int(len(bad))


def fit_to_ref(obj, ref, yaw=None, views=None, verbose=True):
    """Rotate (about Z, in quarter turns), scale and move obj so its silhouettes match the reference views.

    yaw: fix the rotation in degrees instead of trying 0 / 90 / 180 / 270.
    Returns {'yaw', 'scale', 'iou': {view: value}}. The transform is baked into the mesh.
    """
    R = refkit.as_ref(ref)
    names = [n for n in R.views if views is None or n in views]
    if not names:
        raise ValueError("the reference pack has no usable views")
    obj = core.get(obj)
    bpy.context.view_layer.update()
    me = obj.data
    co, tri = _mesh_arrays(me)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    P0 = co.astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    if len(tri) > 120000:                                   # silhouettes do not need every triangle
        tri_s = tri[np.linspace(0, len(tri) - 1, 120000).astype(int)]
    else:
        tri_s = tri
    recs = {n: R[n].rec for n in names}
    masks = {}
    import cv2
    zoom = {n: min(1.0, 256.0 / max(recs[n]["size_px"])) for n in names}
    from scipy import ndimage
    for n in names:
        m = cv2.imread(R[n].path("mask"), 0)
        # outlines only: a gap the picture leaves open (the face under a wig, the body between two locks) is
        # no evidence about where the mesh belongs
        masks[n] = ndimage.binary_fill_holes(cv2.resize(m, (int(round(m.shape[1] * zoom[n])), int(round(m.shape[0] * zoom[n]))),
                                                        interpolation=cv2.INTER_AREA) > 127)
    upright = [n for n in names if n in ("front", "back", "left", "right")]
    # the subject need not stand on the ground or in the middle (a layer pack: a wig, a hat): start where the
    # pictures have it
    z_lo = min((R[n].bbox[1] for n in upright), default=0.0)
    target_h = max((R[n].bbox[3] for n in upright), default=R.height) - z_lo
    start = np.zeros(3)
    start[2] = z_lo
    for axis, vnames in ((0, ("front", "back")), (1, ("right", "left"))):
        mids = [(R[n].bbox[0] + R[n].bbox[2]) / 2.0 * R[n].rec["u_vec"][axis] for n in vnames if n in names]
        if mids:
            start[axis] = float(np.mean(mids))

    def score(P):
        vals = {}
        for n in names:
            sil = ndimage.binary_fill_holes(raster.silhouette(P[tri_s].astype(np.float32), recs[n], zoom[n]) > 0)
            vals[n] = raster.iou(masks[n], sil)
        return float(np.mean(list(vals.values()))), vals

    def place(yaw_deg, s, off):
        a = math.radians(yaw_deg)
        rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        P = P0 @ rot.T
        lo, hi = P.min(0), P.max(0)
        base = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        return (P - base) * s + off

    def settle(yaw_deg, s, off, rounds=2):
        """Slide the mesh onto the reference using each view's best 2D shift."""
        for _ in range(rounds):
            P = place(yaw_deg, s, off)
            d = np.zeros(3)
            w = np.zeros(3)
            for n in names:
                sil = raster.silhouette(P[tri_s].astype(np.float32), recs[n], zoom[n])
                dx, dy, _i = raster.best_shift(masks[n].astype(np.uint8), sil, 0.35)
                ppu = recs[n]["px_per_unit"] * zoom[n]
                mv = np.array(recs[n]["u_vec"]) * (dx / ppu) + np.array(recs[n]["v_vec"]) * (-dy / ppu)
                d += mv
                w += np.abs(np.array(recs[n]["u_vec"])) + np.abs(np.array(recs[n]["v_vec"]))
            off = off + d / np.maximum(w, 1.0)
        return off

    best = None
    for y in ([yaw] if yaw is not None else (0, 90, 180, 270)):
        P = place(y, 1.0, np.zeros(3))
        s0 = target_h / max(1e-9, P[:, 2].max() - P[:, 2].min())
        off = settle(y, s0, start.copy())
        sc, vals = score(place(y, s0, off))
        if y == 0:
            sc += 0.015                                     # generators put the front toward -Y: prefer it on a tie
        if best is None or sc > best[0]:
            best = (sc, y, s0, off, vals)
    _sc, y, s, off, vals = best
    for step in (0.04, 0.015, 0.006):                       # refine the scale, re-settling the offset each time
        improved = True
        while improved:
            improved = False
            for cand in (s * (1 + step), s * (1 - step)):
                o2 = settle(y, cand, off, 1)
                sc, v2 = score(place(y, cand, o2))
                if sc > np.mean(list(vals.values())) + 1e-4:
                    s, off, vals, improved = cand, o2, v2, True
                    break
    P = place(y, s, off)
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.update()
    obj.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    info = dict(yaw=y, scale=float(s), iou={k: round(v, 3) for k, v in vals.items()})
    if verbose:
        print("[f3d] fit_to_ref(%s): turned %d deg, scaled x%.3f, silhouette IoU %s" % (
            obj.name, y, s, ", ".join("%s %.2f" % kv for kv in info["iou"].items())))
        if min(vals.values()) < 0.7:
            print("[f3d]   low agreement: the mesh may not be the same subject / pose as the reference, or the view "
                  "names of the pack are wrong")
    return info


def stretch_to_ref(obj, ref, limit=0.3, verbose=True):
    """Make obj as wide, deep and tall as the reference pictures show it, by scaling it separately along X, Y and
    Z and moving it (never bending it, so a clean mesh stays clean). The gentle way to bring a generated or
    imported mesh into proportion; k.conform is the forceful one (it warps outlines and can crease a mesh).

    limit: the most any axis may be stretched or squeezed (0.3 = 30 %). Returns the three scale factors."""
    R = refkit.as_ref(ref)
    obj = core.get(obj)
    bpy.context.view_layer.update()
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    P = co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]
    lo, hi = P.min(0), P.max(0)
    want = [[], [], []]
    for v in R:
        b = v.bbox
        for (a0, a1), vec in (((b[0], b[2]), v.u_vec), ((b[1], b[3]), v.v_vec)):
            ax = int(np.argmax(np.abs(vec)))
            if abs(vec[ax]) > 0.99:
                want[ax].append(sorted((a0 * vec[ax], a1 * vec[ax])))
    scale = np.ones(3)
    for ax in range(3):
        if not want[ax]:
            continue
        t0, t1 = float(np.mean([w[0] for w in want[ax]])), float(np.mean([w[1] for w in want[ax]]))
        s = float(np.clip((t1 - t0) / max(hi[ax] - lo[ax], 1e-9), 1.0 - limit, 1.0 + limit))
        P[:, ax] = (P[:, ax] - (lo[ax] + hi[ax]) / 2.0) * s + (t0 + t1) / 2.0
        scale[ax] = s
    me.vertices.foreach_set("co", P.astype(np.float32).ravel())
    me.update()
    obj.matrix_world = Matrix.Identity(4)
    bpy.context.view_layer.update()
    if verbose:
        print("[f3d] stretch_to_ref(%s): x %.2f, y %.2f, z %.2f" % (obj.name, scale[0], scale[1], scale[2]))
    return tuple(float(v) for v in scale)


def adopt(path, ref=None, faces=150000, name=None, yaw=None, fit=True, smooth=True, keep_materials=True, cache=True):
    """Import a mesh made elsewhere and make it a clean, reference-aligned part of this model.

    Joins it into one object, removes floating debris, reduces it to about `faces` triangles and (when a reference
    pack is available) turns / scales / moves it so its silhouettes match the reference views.
    path: .glb .gltf .obj .fbx .stl .ply .usd*     ref: pack or folder (default: the active k.ref())
    yaw: force the rotation about Z (degrees) if the automatic quarter-turn choice is wrong.
    The processed mesh is cached next to the source, so re-running the build script is fast.
    """
    from . import io, modify, shapes
    path = core.in_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("adopt: no such file %s" % path)
    R = None
    if fit:
        try:
            R = refkit.as_ref(ref) if ref is not None else refkit.ref()
        except (ValueError, FileNotFoundError):
            R = None
    params = dict(faces=faces, yaw=yaw, ref=(R.dir, os.path.getmtime(os.path.join(R.dir, "ref.json"))) if R else None,
                  keep_materials=keep_materials, v=5)
    cfile = _key(path, params)
    name = name or os.path.splitext(os.path.basename(path))[0]
    if cache and os.path.isfile(cfile):
        with bpy.data.libraries.load(cfile) as (src, dst):
            dst.objects = list(src.objects)
        obj = next((o for o in dst.objects if o is not None), None)
        if obj is not None:
            core.link(obj)
            obj.name = name
            shapes.shade(obj, smooth)
            print("[f3d] adopt(%s): loaded from cache, %d triangles" % (os.path.basename(path), len(obj.data.polygons)))
            return obj
    top = io.import_mesh(path)
    parts = [o for o in core.expand(top) if o.type == "MESH"]
    if not parts:
        raise ValueError("adopt: %s contains no mesh" % path)
    obj = modify.join(parts, name=name)
    for o in list(top):
        try:
            if o.name in bpy.data.objects and o.type != "MESH":
                bpy.data.objects.remove(o, do_unlink=True)
        except ReferenceError:
            pass
    if not keep_materials:
        obj.data.materials.clear()
    before = len(obj.data.polygons)
    removed = drop_floaters(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if not obj.data.uv_layers:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    _co, tri = _mesh_arrays(obj.data)
    if faces and len(tri) > faces * 1.05:
        modify.decimate(obj, ratio=faces / float(len(tri)))
        modify.apply(obj)
    info = None
    if R is not None:
        info = fit_to_ref(obj, R, yaw=yaw)
    else:
        core.ground(obj)
    shapes.shade(obj, smooth)
    obj["f3d_adopted"] = True
    _co, tri = _mesh_arrays(obj.data)
    print("[f3d] adopt(%s): %d -> %d triangles%s" % (os.path.basename(path), before, len(tri),
                                                    ", removed %d floating bits" % removed if removed else ""))
    if cache:
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        bpy.data.libraries.write(cfile, {obj}, compress=True)
    return obj
