"""Silhouette conforming: gently warp the finished model so its outlines land on the reference outlines."""
import numpy as np

from . import core, raster, refkit


def _flow(model, target, max_step, sigma):
    """Displacement field (dx, dy in pixels) that moves the model silhouette toward the target silhouette.

    Each outline pixel of the model wants to travel along the outline normal until it meets the target outline;
    those wishes are then spread smoothly over the whole picture, so the inside follows the outline it belongs to.
    """
    import cv2
    m, t = model > 0, target > 0
    if not m.any() or not t.any():
        return None
    dist = lambda a: cv2.distanceTransform(a.astype(np.uint8), cv2.DIST_L2, 5)
    phi_m = dist(~m) - dist(m)                      # signed distance, positive outside
    phi_t = dist(~t) - dist(t)
    edge = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    sm = cv2.GaussianBlur(phi_m, (0, 0), 1.5)
    gx, gy = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    n = np.maximum(np.sqrt(gx * gx + gy * gy), 1e-6)
    step = np.clip(-phi_t, -max_step, max_step)     # inside the target -> move outward
    wx, wy = np.where(edge, step * gx / n, 0.0), np.where(edge, step * gy / n, 0.0)
    w = edge.astype(np.float32)
    den = np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-6)
    # far from any outline the weights vanish: fade the motion out instead of amplifying noise
    conf = np.clip(cv2.GaussianBlur(w, (0, 0), sigma * 3) * (sigma * 12), 0.0, 1.0)
    fx = cv2.GaussianBlur(wx.astype(np.float32), (0, 0), sigma) / den * conf
    fy = cv2.GaussianBlur(wy.astype(np.float32), (0, 0), sigma) / den * conf
    return fx, fy


def conform(objs=None, ref=None, views=None, rounds=5, strength=1.0, limit=0.08, verbose=True):
    """Warp the model (all given objects together, as one piece of space) so that its silhouettes match the
    reference views. Use it as the last geometry step, before k.paint_from_ref: it turns "roughly the right
    proportions" into outlines that sit on the reference, and joints stay joined because every part is warped
    by the same field.

    rounds: refinement passes.  strength: 0..1 share of the correction applied per pass.
    limit: no vertex moves further than this fraction of the subject height in total, so a part that is missing
           altogether is not faked by stretching its neighbours (build the missing part instead).
    Returns {view: (IoU before, IoU after)}.
    """
    import cv2
    from scipy import ndimage
    R = refkit.as_ref(ref) if ref is not None else refkit.ref()
    names = [n for n in R.views if views is None or n in views]
    if not names:
        raise ValueError("conform: the reference pack has none of the requested views")
    from . import modify
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    if not items:
        raise ValueError("conform: nothing to conform")
    modify.apply(items)
    geo = []
    for o in items:
        me = o.data
        nv = len(me.vertices)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(o.matrix_world, dtype=np.float64)
        P = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        me.calc_loop_triangles()
        t = np.empty(len(me.loop_triangles) * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", t)
        geo.append(dict(obj=o, P=P, P0=P.copy(), T=t.reshape(-1, 3), M=m))
    recs = {n: R[n].rec for n in names}
    zoom = {n: min(1.5, 320.0 / max(recs[n]["size_px"])) for n in names}
    target = {}
    for n in names:
        m = cv2.imread(R[n].path("mask"), 0)
        target[n] = cv2.resize(m, (int(round(m.shape[1] * zoom[n])), int(round(m.shape[0] * zoom[n]))),
                               interpolation=cv2.INTER_AREA) > 127
    cap = limit * R.height
    # a gap the picture leaves open but the model fills (the face under a wig, a body hidden between two locks)
    # is not an outline to chase: only the outlines both have are matched
    filled = {n: ndimage.binary_fill_holes(target[n]) for n in names}
    for n in names:                                         # a layer may lie anywhere behind the rest of its subject
        rest = R[n].around()
        if rest is not None:
            filled[n] |= cv2.resize(rest.astype(np.uint8), (target[n].shape[1], target[n].shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0

    def silhouettes():
        tris = np.concatenate([g["P"][g["T"]] for g in geo]).astype(np.float32)
        return {n: raster.silhouette(tris, recs[n], zoom[n]) > 0 for n in names}

    sil = silhouettes()
    before = {n: raster.iou(target[n] | (filled[n] & sil[n]), sil[n]) for n in names}
    for _ in range(int(rounds)):
        fields = {}
        for n in names:
            ppu = recs[n]["px_per_unit"] * zoom[n]
            f = _flow(sil[n], target[n] | (filled[n] & sil[n]), max_step=0.02 * R.height * ppu,
                      sigma=max(2.0, 0.035 * R.height * ppu))
            if f is not None:
                fields[n] = (f, ppu)
        if not fields:
            break
        for g in geo:
            P = g["P"]
            move = np.zeros_like(P)
            wsum = np.zeros(3)
            for n, ((fx, fy), ppu) in fields.items():
                rec = recs[n]
                u_vec, v_vec = np.array(rec["u_vec"], float), np.array(rec["v_vec"], float)
                x = (P @ u_vec - rec["frame"][0]) * ppu
                y = (rec["frame"][3] - P @ v_vec) * ppu
                coords = np.stack([y - 0.5, x - 0.5])
                dx = ndimage.map_coordinates(fx, coords, order=1, mode="nearest")
                dy = ndimage.map_coordinates(fy, coords, order=1, mode="nearest")
                move += np.outer(dx / ppu, u_vec) - np.outer(dy / ppu, v_vec)
                wsum += np.abs(u_vec) + np.abs(v_vec)
            move /= np.maximum(wsum, 1.0)
            P = P + move * strength
            off = P - g["P0"]
            d = np.linalg.norm(off, axis=1)
            over = d > cap
            if over.any():
                P[over] = g["P0"][over] + off[over] * (cap / d[over])[:, None]
            g["P"] = P
        sil = silhouettes()
    after = {n: raster.iou(target[n] | (filled[n] & sil[n]), sil[n]) for n in names}
    for g in geo:
        inv = np.linalg.inv(g["M"])
        local = g["P"] @ inv[:3, :3].T + inv[:3, 3]
        g["obj"].data.vertices.foreach_set("co", local.astype(np.float32).ravel())
        g["obj"].data.update()
    import bpy
    bpy.context.view_layer.update()
    if verbose:
        print("[f3d] conform: silhouette IoU " + ", ".join("%s %.2f -> %.2f" % (n, before[n], after[n]) for n in names))
    return {n: (before[n], after[n]) for n in names}
