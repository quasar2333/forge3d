"""Ironing: make the surface under a drawn feature smooth, so the drawing lies on it like a print."""
import numpy as np

from . import core, refkit, register

_ZONES = []        # (RefMask, view name): areas ironed in this build; paint_from_ref gives them to that view


def zones():
    return list(_ZONES)


def clear():
    del _ZONES[:]


def iron(objs=None, region=None, smooth=0.04, strength=1.0, own=True, verbose=True):
    """Iron out the relief under part of a reference picture, so that what is drawn there (a face with its eyes,
    a print, an emblem) lies on a smooth surface instead of breaking over bumps that are in the wrong place.

    Why: an image-to-3D mesh has its own idea of where a lock of hair hangs or a fold runs. Where the picture
    draws an eye across such a lock, projection prints half the eye on the lock and half on the face; from the
    picture's own camera that looks perfect, from every other angle it is torn. Under a drawn feature the
    picture should carry the detail and the geometry only the overall form.

    region: a RefMask from the view that draws the feature, e.g. the face read off refs/front_grid.png:
            k.iron(base, k.ref()["front"].mask(box=(-0.2, 0.45, 0.2, 0.70)))
    smooth: how much form survives, as a fraction of the subject height: relief narrower than about twice this
            is flattened, broader form (the roundness of a head) stays. 0.02 = light, 0.04 = default, 0.08 = strong.
    strength: 0..1 blend between the original surface and the ironed one.
    own: paint_from_ref then lets only this view paint the ironed area (no second pair of eyes from the profile).
    Everything in front of the ironed sheet (the lock) is pressed into it; nothing behind the first surface
    moves. Call it after k.conform and before k.paint_from_ref. Returns the objects.
    """
    import cv2
    from scipy import ndimage
    if not isinstance(region, refkit.RefMask):
        raise ValueError("iron: region must be a RefMask, e.g. k.ref()['front'].mask(box=(u0, v0, u1, v1))")
    if region.empty:
        raise ValueError("iron: the region is empty (box outside the silhouette?)")
    v = region.view
    R = v.pack
    from . import modify
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    if not items:
        raise ValueError("iron: nothing to iron")
    modify.apply(items)
    dg = core.depsgraph()
    tris = []
    for o in [o for o in core.scene_objects() if o.type == "MESH"]:
        V, T = core.world_tris(o, dg)
        if len(T):
            tris.append(V[T])
    sc = register.view_scale(v)
    D, _to_px, ppu = register.view_buffer(np.concatenate(tris), v, sc)
    h, w = D.shape
    inside = D > register.EMPTY
    M = cv2.resize(region.array.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    M &= inside
    if not M.any():
        raise ValueError("iron: the model is not under that part of the picture")
    sigma = max(1.5, smooth * R.height * ppu)
    wgt = M.astype(np.float32)
    Zs = cv2.GaussianBlur(np.where(M, D, 0.0).astype(np.float32), (0, 0), sigma) / np.maximum(cv2.GaussianBlur(wgt, (0, 0), sigma), 1e-6)
    # fade in from the border of the region, so the ironed sheet joins the surface around it without a step
    t = np.clip(cv2.distanceTransform(M.astype(np.uint8), cv2.DIST_L2, 5) / sigma, 0.0, 1.0)
    t = t * t * (3 - 2 * t) * float(strength)
    Zt = np.where(M, np.where(M, D, 0.0) + t * (Zs - np.where(M, D, 0.0)), 0.0).astype(np.float32)
    f = v.frame
    moved = 0
    tau = 0.01 * R.height
    for o in items:
        me = o.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(o.matrix_world, dtype=np.float64)
        P = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        x, y = (P @ v.u_vec - f[0]) * ppu, (f[3] - P @ v.v_vec) * ppu
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ixc, iyc = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
        sel = ok & M[iyc, ixc]
        if not sel.any():
            continue
        z = P @ v.toward_camera
        first = D[iyc, ixc].astype(np.float64)
        target = ndimage.map_coordinates(Zt, np.stack([y - 0.5, x - 0.5]), order=1, mode="nearest")
        target = np.where(M[iyc, ixc], target, z)
        new = z.copy()
        # in front of the sheet: pressed onto it. Layers keep their order (the first surface lands on the sheet,
        # what lay behind it stays a hair behind), otherwise they would flicker through each other
        low = sel & (target < first)
        new[low] = np.minimum(z[low], target[low] - 0.03 * (first[low] - z[low]))
        up = sel & (target > first)                        # the sheet lies in front: the visible skin comes up to it
        new[up] = z[up] + (target[up] - first[up]) * np.exp(-np.maximum(first[up] - z[up], 0.0) / tau)
        d = new - z
        P += np.outer(d, v.toward_camera)
        moved += int((np.abs(d) > 1e-7).sum())
        inv = np.linalg.inv(m)
        me.vertices.foreach_set("co", (P @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32).ravel())
        me.update()
    import bpy
    bpy.context.view_layer.update()
    if own:
        _ZONES.append((region, v.name))
    if verbose:
        b = region.bbox
        print("[f3d] iron: smoothed the surface under the %s picture's area (%.2f, %.2f)-(%.2f, %.2f), %d vertices "
              "moved%s" % (v.name, b[0], b[1], b[2], b[3], moved,
                           "; that view alone will paint it" if own else ""))
    return objs
