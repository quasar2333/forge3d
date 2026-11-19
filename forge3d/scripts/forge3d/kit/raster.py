"""Orthographic silhouettes of the model in reference-view frames, and how well two silhouettes agree."""
import math

import numpy as np

from . import core


def _rec(view):
    return getattr(view, "rec", view)


def scene_tris(objs=None):
    """All model triangles in world space: (N, 3, 3) float32."""
    items = [o for o in (core.scene_objects(ink=True) if objs is None else core.expand(objs))
             if o.type in core.GEOM_TYPES]
    dg = core.depsgraph()
    chunks = []
    for o in items:
        v, t = core.world_tris(o, dg)
        if len(t):
            chunks.append(v[t])
    return np.concatenate(chunks) if chunks else np.zeros((0, 3, 3), np.float32)


def silhouette(tris, view, zoom=1.0):
    """Orthographic silhouette of triangles (N, 3, 3) in a reference view's frame. -> uint8 mask (h, w)."""
    import cv2
    view = _rec(view)
    w, h = int(round(view["size_px"][0] * zoom)), int(round(view["size_px"][1] * zoom))
    mask = np.zeros((h, w), np.uint8)
    if len(tris) == 0:
        return mask
    s = view["px_per_unit"] * zoom
    u = tris @ np.array(view["u_vec"], np.float32)
    v = tris @ np.array(view["v_vec"], np.float32)
    pts = np.stack([(u - view["frame"][0]) * s, (view["frame"][3] - v) * s], -1)            # (N, 3, 2) pixels
    lo, hi = pts.min(1), pts.max(1)
    keep = (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w) & (lo[:, 1] < h)
    pts, lo, hi = pts[keep], lo[keep], hi[keep]
    big = ((hi - lo) > 1.25).any(1)
    # triangles smaller than a pixel: mark the pixels under their corners and centre (vectorised)
    small = pts[~big]
    if len(small):
        sam = np.concatenate([small.reshape(-1, 2), small.mean(1)])
        ix, iy = np.floor(sam[:, 0]).astype(np.int64), np.floor(sam[:, 1]).astype(np.int64)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        mask[iy[ok], ix[ok]] = 255
    # larger ones are filled one by one (a single fillPoly call would XOR overlapping triangles)
    bigp = np.round(np.clip(pts[big], -8.0 * (w + h), 8.0 * (w + h)) * 16.0).astype(np.int32)
    for t in bigp:
        cv2.fillConvexPoly(mask, t, 255, lineType=cv2.LINE_8, shift=4)
    if len(small):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def iou(a, b):
    a, b = a > 0, b > 0
    union = (a | b).sum()
    return float((a & b).sum()) / float(union) if union else 1.0


def best_shift(ref, mod, max_frac=0.2):
    """Translation (dx, dy pixels) of the model silhouette that best matches the reference, and the IoU there."""
    a, b = (ref > 0).astype(np.float32), (mod > 0).astype(np.float32)
    h, w = a.shape
    fa, fb = np.fft.rfft2(a, (2 * h, 2 * w)), np.fft.rfft2(b, (2 * h, 2 * w))
    corr = np.fft.irfft2(fa * np.conj(fb), (2 * h, 2 * w))
    lim_y, lim_x = max(1, int(h * max_frac)), max(1, int(w * max_frac))
    ys = list(range(0, lim_y + 1)) + list(range(2 * h - lim_y, 2 * h))
    xs = list(range(0, lim_x + 1)) + list(range(2 * w - lim_x, 2 * w))
    sub = corr[np.ix_(ys, xs)]
    iy, ix = np.unravel_index(int(sub.argmax()), sub.shape)
    inter = float(sub[iy, ix])
    dy = ys[iy] if ys[iy] <= lim_y else ys[iy] - 2 * h
    dx = xs[ix] if xs[ix] <= lim_x else xs[ix] - 2 * w
    union = float(a.sum() + b.sum()) - inter
    return dx, dy, (inter / union if union > 0 else 1.0)


def _bary_samples(n):
    """Barycentric sample points of a triangle split into n x n sub-triangles (corners and interior lattice)."""
    pts = [(i / n, j / n, 1.0 - i / n - j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    if n > 1:
        pts += [((i + 1 / 3) / n, (j + 1 / 3) / n, 1.0 - (i + j + 2 / 3) / n) for i in range(n) for j in range(n - i)]
    else:
        pts.append((1 / 3, 1 / 3, 1 / 3))
    return np.array(pts, np.float32)


def depth_buffer(tris, toward, u_vec, v_vec, frame, ppu):
    """Point-sampled z-buffer of triangles (N, 3, 3) seen along -toward.

    -> (D, to_px): D is (h, w) float32 holding, per pixel, the largest coordinate along `toward` (the surface nearest
    the camera), -inf where nothing is; to_px(points (M, 3)) -> (ix, iy, depth, inside).
    """
    import cv2
    w = max(1, int(math.ceil((frame[2] - frame[0]) * ppu)))
    h = max(1, int(math.ceil((frame[3] - frame[1]) * ppu)))
    uv, vv, dv = (np.asarray(a, np.float32) for a in (u_vec, v_vec, toward))

    def to_px(p):
        p = np.asarray(p, np.float32)
        x = (p @ uv - frame[0]) * ppu
        y = (frame[3] - p @ vv) * ppu
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        return ix, iy, p @ dv, (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    D = np.full((h, w), -np.inf, np.float32)
    if len(tris) == 0:
        return D, to_px
    x = (tris @ uv - frame[0]) * ppu
    y = (frame[3] - tris @ vv) * ppu
    z = tris @ dv
    ext = np.maximum(x.max(1) - x.min(1), y.max(1) - y.min(1))
    level = np.ceil(np.log2(np.maximum(ext, 1.0) / 0.9)).astype(np.int32).clip(0, 9)
    for lv in np.unique(level):
        sel = level == lv
        b = _bary_samples(int(2 ** lv))                                    # (S, 3)
        sx = (x[sel] @ b.T).ravel()
        sy = (y[sel] @ b.T).ravel()
        sz = (z[sel] @ b.T).ravel()
        ix, iy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        np.maximum.at(D, (iy[ok], ix[ok]), sz[ok])
    filled = cv2.dilate(D, np.ones((3, 3), np.uint8))                      # close pinholes between samples
    D = np.where(np.isfinite(D), D, filled)
    return D, to_px
