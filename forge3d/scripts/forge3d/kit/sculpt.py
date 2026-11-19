"""Shapes grown from pictures: inflate an outline into a soft solid, or build a rounded volume whose front and
side outlines are the reference's own."""
import math

import bpy
import numpy as np

from . import core, refkit
from .shapes import finish

_VIEW_FRAMES = {
    # view: (u axis, v axis, depth axis pointing toward that view's camera) as world unit vectors
    "front": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "right": ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    "left": ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "bottom": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
}


def _grid_mask(shape, view, res):
    """shape (RefMask or polygon) -> (bool grid (h, w) with a 2-cell empty border, u0, v0, cell, view name).
    Row 0 of the grid is the lowest v."""
    import cv2
    if isinstance(shape, refkit.RefMask):
        vname = shape.view.name
        if shape.empty:
            raise ValueError("the mask is empty (wrong region ids, or a box that misses the subject?)")
        u0, v0, u1, v1 = shape.bbox
    else:
        vname = view
        pts = np.asarray(shape, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("outline must be a RefMask or a list of at least 3 (u, v) points")
        u0, v0, u1, v1 = pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()
    if vname not in _VIEW_FRAMES:
        raise ValueError("view must be one of %s" % ", ".join(_VIEW_FRAMES))
    cell = max(u1 - u0, v1 - v0) / float(res)
    if isinstance(shape, refkit.RefMask):
        # a small part does not need a finer grid than a large one: below about 1/500 of the subject's height
        # the extra triangles show nothing and make the model heavy
        cell = max(cell, 0.002 * shape.view.pack.height)
    w, h = int(math.ceil((u1 - u0) / cell)) + 4, int(math.ceil((v1 - v0) / cell)) + 4
    ou, ov = u0 - 2 * cell, v0 - 2 * cell
    if isinstance(shape, refkit.RefMask):
        vw = shape.view
        us = ou + (np.arange(w) + 0.5) * cell
        vs = ov + (np.arange(h) + 0.5) * cell
        xs = np.clip(np.floor(vw.x_of(us)).astype(int), 0, shape.array.shape[1] - 1)
        ys = np.clip(np.floor(vw.y_of(vs)).astype(int), 0, shape.array.shape[0] - 1)
        inside_u = (vw.x_of(us) >= 0) & (vw.x_of(us) < shape.array.shape[1])
        inside_v = (vw.y_of(vs) >= 0) & (vw.y_of(vs) < shape.array.shape[0])
        g = shape.array[np.ix_(ys, xs)] & inside_v[:, None] & inside_u[None, :]
        # supersampled coverage would be nicer; a light blur + threshold removes the pixel stair-steps instead
        g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), max(0.6, 0.5 * vw.px_per_unit * cell * 0.5)) > 0.5 \
            if vw.px_per_unit * cell < 1.0 else g
    else:
        canvas = np.zeros((h, w), np.uint8)
        px = np.stack([(pts[:, 0] - ou) / cell, (pts[:, 1] - ov) / cell], -1)
        cv2.fillPoly(canvas, [np.round(px * 16).astype(np.int32)], 255, shift=4)
        g = canvas > 0
    return g, ou, ov, cell, vname


def _mesh_from_field(field, origin, cell, axes, name, smooth_iters, kw):
    """Marching cubes of a 3D field (positive inside), indexed [a, b, c] along three world axes."""
    from skimage import measure
    if field.max() <= 0:
        raise ValueError("%s: the shape is empty" % name)
    verts, faces, _n, _v = measure.marching_cubes(field, level=0.0)
    A = np.array(axes, float)                                  # rows: world direction of grid axes a, b, c
    world = np.asarray(origin, float) + (verts * cell) @ A
    if np.linalg.det(A) < 0:
        faces = faces[:, ::-1]
    me = bpy.data.meshes.new(name)
    me.from_pydata(world.tolist(), [], faces.tolist())
    me.validate()
    me.update()
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=cell * 1e-3)
    for _ in range(int(smooth_iters)):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    kw.setdefault("recenter", True)
    return finish(bm, name, **kw)


def inflate(shape, view="front", thickness=None, at=0.0, front=1.0, back=1.0, res=96, smooth=3, name="Inflate", **kw):
    """Blow a flat outline up into a soft, rounded solid, like an inflated sticker: fins, ears, tails, bows,
    leaves, hair locks, wings, hands, flat-ish bodies. Thin parts of the outline come out thin, wide parts thick.

    shape: a RefMask (e.g. k.ref()['front'].mask(regions=[7, 11])) or a polygon [(u, v), ...] in the view plane
           (front view: u = x, v = z; right view: u = y, v = z; top view: u = x, v = y), world units.
    view: which way the outline is seen (taken from the mask when shape is a RefMask).
    thickness: greatest thickness along the view direction (default: 0.6 x the outline's narrower side).
    at: where the middle of the solid sits along the view axis: front / back views: its y; right / left: its x;
        top: its z.
    front, back: 0..1 how much of the thickness goes toward the viewer and away (back=0 gives a flat back, e.g. a
        relief or badge lying on a surface).
    res: grid cells along the outline's longer side.  smooth: smoothing passes.
    """
    from scipy import ndimage
    g, ou, ov, cell, vname = _grid_mask(shape, view, int(res))
    d = ndimage.distance_transform_edt(g) * cell                    # world distance to the outline
    if thickness is None:
        ys, xs = np.flatnonzero(g.any(1)), np.flatnonzero(g.any(0))
        thickness = 0.6 * min(len(ys), len(xs)) * cell
    r = max(thickness / 2.0, cell)
    dd = np.minimum(d, r)
    hgt = np.sqrt(np.maximum(2 * r * dd - dd * dd, 0.0))             # quarter-circle rim, flat plateau of height r
    nz = int(math.ceil(r / cell)) + 2
    z = (np.arange(-nz, nz + 1)) * cell
    side = np.where(z >= 0, max(front, 1e-3), max(back, 1e-3))
    field = hgt[:, :, None] * side[None, None, :] - np.abs(z)[None, None, :]
    field[~g] = np.minimum(field[~g], -cell * 0.5)
    field = ndimage.gaussian_filter(field, 0.6)
    ua, va, da = (np.array(a, float) for a in _VIEW_FRAMES[vname])
    origin = ua * (ou + 0.5 * cell) + va * (ov + 0.5 * cell) + da * z[0]
    # depth axis: da points toward the camera; `at` is a world coordinate along the axis the view looks down
    axis = int(np.argmax(np.abs(da)))
    origin[axis] += at
    return _mesh_from_field(field, origin, cell, (va, ua, da), name, smooth, kw)


def volume(front, side, res=112, roundness=2.3, smooth=3, name="Volume", **kw):
    """A rounded solid whose front silhouette is `front` and whose side silhouette is `side`: at every height its
    cross-section is a rounded shape as wide as the front outline and as deep as the side outline there.
    The quickest way to a body, head, hair mass, dress or whole creature that already fits the reference from
    two directions. Separate lobes in the front outline (legs, arms beside the body) stay separate.

    front: RefMask from the front (or back) view.   side: RefMask from the right (or left) view.
    roundness: 2 = elliptical sections, 3-4 = boxier, 1.5 = diamond-ish.
    res: grid cells along the height.
    """
    from scipy import ndimage
    if not isinstance(front, refkit.RefMask) or not isinstance(side, refkit.RefMask):
        raise ValueError("volume needs two RefMask arguments, e.g. volume(R['front'].mask(), R['right'].mask())")
    fv, sv = front.view, side.view
    if fv.name not in ("front", "back") or sv.name not in ("right", "left"):
        raise ValueError("volume: `front` must come from the front / back view and `side` from the right / left view")
    fb, sb = front.bbox, side.bbox
    v0, v1 = max(fb[1], sb[1]), min(fb[3], sb[3])
    if v1 <= v0:
        raise ValueError("volume: the two masks do not overlap in height (front %.3f..%.3f, side %.3f..%.3f)" % (
            fb[1], fb[3], sb[1], sb[3]))
    v0, v1 = min(fb[1], sb[1]), max(fb[3], sb[3])
    cell = max((v1 - v0) / float(res), 0.0028 * fv.pack.height)
    fx = 1.0 if fv.name == "front" else -1.0                      # world x per unit of the front view's u
    sy = 1.0 if sv.name == "right" else -1.0                      # world y per unit of the side view's u
    xa, xb = sorted((fb[0] * fx, fb[2] * fx))
    ya, yb = sorted((sb[0] * sy, sb[2] * sy))
    nx, ny, nz = (int(math.ceil((b - a) / cell)) + 4 for a, b in ((xa, xb), (ya, yb), (v0, v1)))
    X = xa - 2 * cell + (np.arange(nx) + 0.5) * cell
    Y = ya - 2 * cell + (np.arange(ny) + 0.5) * cell
    Z = v0 - 2 * cell + (np.arange(nz) + 0.5) * cell
    field = np.full((nx, ny, nz), -1.0, np.float32)
    n = float(roundness)
    for k, z in enumerate(Z):
        fr = [tuple(sorted((a * fx, b * fx))) for a, b in front.spans(z)]
        sr = [tuple(sorted((a * sy, b * sy))) for a, b in side.spans(z)]
        if not fr or not sr:
            continue
        sl = np.full((nx, ny), -1.0, np.float32)
        for a, b in fr:
            if b - a < cell * 0.5:
                continue
            u = (X - (a + b) / 2) / ((b - a) / 2)
            for c, d in sr:
                if d - c < cell * 0.5:
                    continue
                w = (Y - (c + d) / 2) / ((d - c) / 2)
                rr = (np.abs(u)[:, None] ** n + np.abs(w)[None, :] ** n) ** (1.0 / n)
                # signed, roughly in world units, so the surface position does not depend on the lobe size
                sl = np.maximum(sl, (1.0 - rr) * min(b - a, d - c) / 2 / cell)
        field[:, :, k] = sl
    field = ndimage.gaussian_filter(np.clip(field, -2.0, 2.0), (0.5, 0.5, 0.9))
    origin = (X[0], Y[0], Z[0])
    return _mesh_from_field(field, origin, cell, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), name, smooth, kw)
