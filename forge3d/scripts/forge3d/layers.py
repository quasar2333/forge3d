"""Layers: one drawing taken apart into complete parts (bare body, hair, headwear ...) that share one frame.

A turnaround shows every part half hidden behind the others; a mesh generated from it is one fused skin, and
whatever is projected on it bleeds from part to part. Taken apart first (an image model redraws each part alone,
complete, in the same place on the sheet), every part can be generated, fitted and painted on its own and then
simply stands where it belongs: a layer pack uses the world frame of the pack it was taken from.
"""
import json
import os

import cv2
import numpy as np

from . import refs as refmod


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def base_sheet(doc, base_dir):
    """The sheet a pack was cut from, rebuilt from the pack's own views (the pack may hold an enlarged version of
    a small picture, so the source file is not what the layers are measured against). -> (bgr, mask)."""
    if len(doc.get("sources") or []) != 1 or not doc.get("sheet_size"):
        raise SystemExit("layers need a reference pack made from ONE sheet image (this one has %d sources). Put the "
                         "views side by side in one picture first." % len(doc.get("sources") or []))
    W, H = doc["sheet_size"]
    bgr = np.full((H, W, 3), 255, np.uint8)
    mask = np.zeros((H, W), np.uint8)
    for n, rec in doc["views"].items():
        m = cv2.imread(os.path.join(base_dir, rec["mask"]), 0)
        tex = cv2.imread(os.path.join(base_dir, rec["texture"]), 1)
        x0, y0 = rec["sheet_box"][0], rec["sheet_box"][1]
        h, w = min(m.shape[0], H - y0), min(m.shape[1], W - x0)
        sel = m[:h, :w] > 0
        bgr[y0:y0 + h, x0:x0 + w][sel] = tex[:h, :w][sel]
        mask[y0:y0 + h, x0:x0 + w] = np.maximum(mask[y0:y0 + h, x0:x0 + w], m[:h, :w])
    return bgr, mask


class _Matcher:
    """Scores where (and how large) a layer picture sits on the picture it was taken from. Colours are reduced to
    a few families; a placement is good when, family by family, the layer's areas coincide with areas of the
    original inside the layer's box. Ink lines count double: they pin the position down."""

    def __init__(self, bgr, mask, work=256, k=8):
        self.f = work / float(max(bgr.shape[:2]))
        self.bgr = cv2.resize(bgr, None, fx=self.f, fy=self.f, interpolation=cv2.INTER_AREA)
        self.mask = cv2.resize(mask, (self.bgr.shape[1], self.bgr.shape[0]), interpolation=cv2.INTER_AREA) > 127
        lab = _lab(self.bgr)
        pts = lab[self.mask]
        if len(pts) > 20000:
            pts = pts[np.random.RandomState(1).choice(len(pts), 20000, replace=False)]
        k = int(min(k, max(2, len(pts) // 50)))
        _c, _l, centers = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                     cv2.KMEANS_PP_CENTERS)
        self.centers = centers
        self.weight = np.where(centers[:, 0] < 0.35 * 255, 2.0, 1.0)
        self.label = self._labels(lab, self.mask, None)

    def _labels(self, lab, mask, tol):
        d = np.linalg.norm(lab[:, :, None, :] - self.centers[None, None], axis=3)
        lab_i = d.argmin(2)
        if tol is not None:
            lab_i[d.min(2) > tol] = -1
        lab_i[~mask] = -1
        return lab_i

    def search(self, bgr, mask, scales, reach, prior=(1.0, 0.0, 0.0), window=None):
        """Find the layer on the original. bgr / mask: the layer in its own pixels.
        A placement (s, dx, dy) means: original pixel (x, y) shows what the layer shows at s * (x + dx, y + dy).
        scales: values of s to try.  prior: the placement to start from.  reach: how far (original pixels) the
        middle of the layer may move from where the prior puts it.  window: (x0, y0, x1, y1) in original pixels,
        only that part of the original takes part.  -> (score, s, dx, dy), s None when nothing could be scored."""
        f = self.f
        lab_o = self.label
        ox0 = oy0 = 0
        if window is not None:
            ox0, oy0 = int(max(0, np.floor(window[0] * f))), int(max(0, np.floor(window[1] * f)))
            ox1, oy1 = int(np.ceil(window[2] * f)) + 1, int(np.ceil(window[3] * f)) + 1
            lab_o = lab_o[oy0:oy1, ox0:ox1]
        P = int(np.ceil(reach * f)) + 1
        best = (-1.0, None, 0.0, 0.0)
        ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        if len(ys) == 0 or lab_o.size == 0:
            return best
        bx0, by0, bx1, by1 = int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1
        crop, cmask = bgr[by0:by1, bx0:bx1], mask[by0:by1, bx0:bx1]
        s0, dx0, dy0 = prior
        mid = ((bx0 + bx1) / 2.0 / s0 - dx0, (by0 + by1) / 2.0 / s0 - dy0)       # middle of the layer, original pixels
        H, W = lab_o.shape[0] + 2 * P, lab_o.shape[1] + 2 * P
        fams = []
        for k in range(len(self.centers)):
            Ok = np.zeros((H, W), np.float32)
            Ok[P:P + lab_o.shape[0], P:P + lab_o.shape[1]] = (lab_o == k)
            fams.append(cv2.GaussianBlur(Ok, (0, 0), 0.8))
        for s in scales:
            g = f / s
            w, h = max(2, int(round(crop.shape[1] * g))), max(2, int(round(crop.shape[0] * g)))
            if h > H or w > W:
                continue
            lb = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
            lm = cv2.resize(cmask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 127
            ll = self._labels(_lab(lb), lm, 22.0)
            total = float((ll >= 0).sum())
            if total < 20:
                continue
            cx, cy = mid[0] * f - w / 2.0 - ox0 + P, mid[1] * f - h / 2.0 - oy0 + P    # corner of the crop under the prior
            ones = np.ones((h, w), np.float32)
            acc = None
            for k in range(len(self.centers)):
                Lk = (ll == k).astype(np.float32)
                nk = float(Lk.sum())
                if nk < 4:
                    continue
                inter = cv2.matchTemplate(fams[k], cv2.GaussianBlur(Lk, (0, 0), 0.8), cv2.TM_CCORR)
                inbox = cv2.matchTemplate(fams[k], ones, cv2.TM_CCORR)
                part = inter / np.maximum(nk + inbox - inter, 1.0) * (self.weight[k] * nk / total)
                acc = part if acc is None else acc + part
            if acc is None:
                continue
            x_lo, y_lo = max(int(round(cx)) - P, 0), max(int(round(cy)) - P, 0)
            x_hi, y_hi = min(int(round(cx)) + P, acc.shape[1] - 1), min(int(round(cy)) + P, acc.shape[0] - 1)
            if x_hi < x_lo or y_hi < y_lo:
                continue
            sub = acc[y_lo:y_hi + 1, x_lo:x_hi + 1]
            iy, ix = np.unravel_index(int(sub.argmax()), sub.shape)
            sc = float(sub[iy, ix])
            if sc > best[0]:
                X, Y = (x_lo + ix - P + ox0) / f, (y_lo + iy - P + oy0) / f          # corner of the crop, original pixels
                best = (sc, float(s), bx0 / float(s) - X, by0 / float(s) - Y)
        return best


def _place(doc0, s, dx, dy, per_view=None):
    """Where each base view's origin and scale end up in the layer sheet."""
    out = {}
    for n, rec in doc0["views"].items():
        sv, dxv, dyv = per_view.get(n, (s, dx, dy)) if per_view else (s, dx, dy)
        ox, oy = rec["sheet_box"][0] + rec["origin_px"][0], rec["sheet_box"][1] + rec["origin_px"][1]
        b = rec["sheet_box"]
        out[n] = dict(origin=(sv * (ox + dxv), sv * (oy + dyv)), ppu=rec["px_per_unit"] * sv,
                      box=(sv * (b[0] + dxv), sv * (b[1] + dyv), sv * (b[2] + dxv), sv * (b[3] + dyv)), s=sv, dx=dxv, dy=dyv)
    return out


def place_layer(image, doc0, o_bgr, o_mask, bg="auto", tol=10.0, holes="auto", align=True):
    """Read one layer picture, bring it to the size of the sheet it was taken from and find where every view of
    it sits on that sheet. -> dict(im, mask, where, owner, names, align, notes)."""
    notes = []
    im = refmod.read_image(image)
    H0, W0 = o_bgr.shape[:2]
    f_need = ((W0 * H0) / float(im.shape[1] * im.shape[0])) ** 0.5
    im, mask, info, _f = refmod.sharp_matte(im, bg, tol, holes, factor=f_need if f_need > 1.3 else None,
                                            sharpen="auto" if f_need > 1.3 else "off")
    if not mask.any():
        raise SystemExit("no subject found in %s (the whole image looks like background)" % image)
    H1, W1 = im.shape[:2]
    s_nom = ((W1 * H1) / float(W0 * H0)) ** 0.5
    if abs((W1 / float(H1)) / (W0 / float(H0)) - 1.0) > 0.03:
        notes.append("the layer picture has another shape (%dx%d) than the sheet it was taken from (%dx%d): its "
                     "place was found by matching alone; check layers_check.png" % (W1, H1, W0, H0))
    s, dx, dy, score = s_nom, 0.0, 0.0, None
    per_view = {}
    if align:
        far = 0.3 * max(W0, H0)
        lay, lmask = im[:, :, :3], mask > 0
        r1 = _Matcher(o_bgr, o_mask, work=256).search(lay, lmask, s_nom * np.geomspace(0.8, 1.25, 19), far, (s_nom, 0.0, 0.0))
        if r1[1] is not None:
            s, dx, dy, score = r1[1], r1[2], r1[3], r1[0]
            fine = _Matcher(o_bgr, o_mask, work=min(800, max(W0, H0)))
            r2 = _Matcher(o_bgr, o_mask, work=448).search(lay, lmask, s * np.geomspace(0.965, 1.035, 9), 0.02 * max(W0, H0),
                                                          (s, dx, dy))
            if r2[1] is not None:
                s, dx, dy, score = r2[1], r2[2], r2[3], r2[0]
            # each view on its own: an image model moves and resizes the views of a sheet independently
            for n, rec in doc0["views"].items():
                b = rec["sheet_box"]
                vh = max(b[3] - b[1], b[2] - b[0])
                win = (b[0] - 0.1 * vh, b[1] - 0.1 * vh, b[2] + 0.1 * vh, b[3] + 0.1 * vh)
                x0, y0 = int(max(0, s * (win[0] + dx))), int(max(0, s * (win[1] + dy)))
                x1, y1 = int(min(W1, s * (win[2] + dx))), int(min(H1, s * (win[3] + dy)))
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = np.zeros_like(lmask)
                sub[y0:y1, x0:x1] = lmask[y0:y1, x0:x1]
                if sub.sum() < 30:
                    continue
                stay = fine.search(lay, sub, [s], 0.0, (s, dx, dy), win)
                cand = fine.search(lay, sub, s * np.geomspace(0.95, 1.05, 11), 0.04 * vh, (s, dx, dy), win)
                if cand[1] is not None and cand[0] > stay[0] + 0.01:
                    per_view[n] = (cand[1], cand[2], cand[3])
    where = _place(doc0, s, dx, dy, per_view)
    # every layer pixel belongs to the view whose (placed) box is nearest
    names = list(doc0["order"])
    owner = np.full(mask.shape, -1, np.int16)
    ys, xs = np.nonzero(mask)
    dist = np.stack([np.maximum(np.maximum(where[n]["box"][0] - xs, xs - where[n]["box"][2]), 0) +
                     np.maximum(np.maximum(where[n]["box"][1] - ys, ys - where[n]["box"][3]), 0) for n in names])
    owner[ys, xs] = dist.argmin(0)
    return dict(im=im, mask=mask, where=where, owner=owner, names=names, notes=notes, image=os.path.abspath(image),
                align=dict(scale=round(float(s), 5), dx=round(float(dx), 2), dy=round(float(dy), 2),
                           score=None if score is None else round(float(score), 3),
                           views={n: [round(float(v), 4) for v in pv] for n, pv in per_view.items()}))


def _on_base(placed, n, rec0):
    """One view of a placed layer on the pixel grid of the base view. -> (bgr with the colours continued past
    the edge, mask bool)."""
    i = placed["names"].index(n)
    m = ((placed["mask"] > 0) & (placed["owner"] == i)).astype(np.uint8) * 255
    w, h = rec0["size_px"]
    if not m.any():
        return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool)
    pl = placed["where"][n]
    b = rec0["sheet_box"]
    A = np.float32([[pl["s"], 0, pl["s"] * (b[0] + pl["dx"])], [0, pl["s"], pl["s"] * (b[1] + pl["dy"])]])   # base px -> layer px
    bgr = refmod.bleed(placed["im"][:, :, :3], m, 0)
    flag = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
    return (cv2.warpAffine(bgr, A, (w, h), flags=flag, borderMode=cv2.BORDER_REPLICATE),
            cv2.warpAffine(m, A, (w, h), flags=flag, borderValue=0) > 127)


def _reach(o_lab, l_lab, alpha, r):
    """How far the colour of every original pixel is from what the layer shows within r pixels of it (0 = the
    layer has exactly that colour there, give or take a small shift)."""
    best = np.full(o_lab.shape[:2], 1e4, np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r + 1:
                continue
            a = np.roll(alpha, (dy, dx), (0, 1))
            d = np.linalg.norm(o_lab - np.roll(l_lab, (dy, dx), (0, 1)), axis=2)
            d[~a] = 1e4
            np.minimum(best, d, out=best)
    return best


GAP = -2


def claim(o_bgr, om, layers, tol=16.0, outer=1.5, background=None):
    """Which layer does every pixel of the original belong to? `layers`: [(bgr, mask)] on the original's grid,
    inner first. -> (label int8: layer index, -1 outside the subject, GAP = background seen through the subject;
    adopted bool: what no layer picture showed).

    The original is cut into small pieces along its own lines and colour edges (a lock of hair, a feather, a
    stretch of sleeve). A piece goes, whole, to the layer whose picture shows most of it in that colour at that
    place (an outer layer wins a tie). An image model never copies a drawing exactly (curls come out shorter, a
    fin smaller), so some pieces are in no layer picture: each of those joins the neighbour it is least separated
    from (no line between them, the same colour), which keeps a part whole however much of it was redrawn.
    Background-coloured pieces that only lines and other colours surround are holes the cut-out missed (the
    white between two curls): they belong to nobody."""
    import networkx as nx
    from scipy import ndimage
    from skimage import graph
    from skimage.segmentation import slic
    h, w = om.shape
    size = max(h, w)
    n = len(layers)
    # everything is decided on a picture of ordinary size and laid back over the full one
    z = min(1.0, 800.0 / size)
    sw, sh = max(8, int(round(w * z))), max(8, int(round(h * z)))
    small = cv2.resize(o_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(om.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    o_lab = _lab(cv2.GaussianBlur(small, (0, 0), 0.8))
    r = max(2, int(round(0.007 * max(sw, sh))))
    lays = [(cv2.resize(lb, (sw, sh), interpolation=cv2.INTER_AREA),
             cv2.resize(lm.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0) for lb, lm in layers]
    rank = outer * np.arange(n, dtype=np.float32)[:, None, None]

    def scores(pics):
        sc = np.full((n, sh, sw), 1e4, np.float32)
        for i, (lb, lm) in enumerate(pics):
            if lm.any():
                sc[i] = _reach(o_lab, _lab(cv2.GaussianBlur(lb, (0, 0), 0.8)), lm, r)
        return sc

    score = scores(lays)
    vs = (score - rank).argmin(0)
    ss = (score.min(0) < tol) & sm
    # an image model shifts the colours of what it copies a little; undo that and look again
    lays = [(_match_colours(lb, small, ss & (vs == i) & lm), lm) for i, (lb, lm) in enumerate(lays)]
    score = scores(lays)
    vs = (score - rank).argmin(0)
    ss = (score.min(0) < tol) & sm
    # pieces over the whole rectangle, then cut to the subject: the masked variant of slic seeds itself with a
    # k-means over every pixel, which takes ten times as long for the same pieces
    seg = slic(small[:, :, ::-1], n_segments=int(np.clip(sw * sh / 90.0, 200, 12000)), compactness=8.0, sigma=0.0,
               start_label=1, convert2lab=True)
    seg[~sm] = 0
    S = int(seg.max()) + 1
    inside = seg > 0
    area = np.bincount(seg[inside], minlength=S).astype(np.float64)
    counts = np.stack([np.bincount(seg[inside & ss & (vs == i)], minlength=S) for i in range(n)]).astype(np.float64)
    owner = counts.argmax(0)
    known = (counts.max(0) >= 0.5 * np.maximum(area, 1.0)) & (area > 0)
    pure = np.zeros(S, bool)
    if background is not None:
        # a piece in the exact colour of the paper is only a part of the subject if a layer shows exactly that
        # there (the highlight in an eye); a tinted white frill next to it does not make it one
        bgl = np.abs(small.astype(np.int32) - np.array(background, np.int32)).max(2) <= 6
        pure = np.bincount(seg[inside], weights=bgl[inside].astype(np.float64), minlength=S) / np.maximum(area, 1.0) > 0.7
        exact = np.bincount(seg[inside & (score.min(0) < 6.0)], minlength=S) / np.maximum(area, 1.0)
        known &= ~pure | (exact >= 0.5)
    # pieces no layer shows in place, but only one layer has that colour anywhere near (the image model drew
    # the fin smaller: its far end is in no picture, yet nothing but the body has feathers of that colour)
    pts = o_lab[sm]
    pts = pts[np.random.RandomState(1).choice(len(pts), min(len(pts), 20000), replace=False)]
    K = int(min(20, max(2, len(pts) // 50)))
    _c, _l, centres = cv2.kmeans(pts, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                 cv2.KMEANS_PP_CENTERS)

    def family(lab_img):
        return np.linalg.norm(lab_img[:, :, None, :] - centres[None, None], axis=3).argmin(2)

    alike = np.linalg.norm(centres[:, None] - centres[None], axis=2) < tol
    fo = family(o_lab)
    R = max(4, int(round(0.04 * max(sw, sh))))
    box = np.ones((2 * R + 1, 2 * R + 1), np.uint8)
    share = np.zeros((n, S))
    for i, (lb, lm) in enumerate(lays):
        if not lm.any():
            continue
        fl = family(_lab(cv2.GaussianBlur(lb, (0, 0), 0.8)))
        has = [cv2.dilate((lm & (fl == c)).astype(np.uint8), box) > 0 for c in range(K)]
        near = np.zeros((sh, sw), bool)
        for c in range(K):
            sel = fo == c
            if sel.any():
                near[sel] = np.any([has[c2][sel] for c2 in np.flatnonzero(alike[c])], axis=0)
        share[i] = np.bincount(seg[inside], weights=near[inside].astype(np.float64), minlength=S) / np.maximum(area, 1.0)
    only = ((share >= 0.6).sum(0) == 1) & ~known & ~pure & (area > 0)
    owner = np.where(only, share.argmax(0), owner)
    shown = known.copy()                                   # what a layer picture really showed in place
    known |= only
    lab_s = _lab(cv2.GaussianBlur(small, (0, 0), 0.8))
    mean = np.stack([np.bincount(seg[inside], weights=lab_s[:, :, c][inside], minlength=S) for c in range(3)], 1) / \
        np.maximum(area, 1.0)[:, None]
    gx, gy = cv2.Sobel(lab_s, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lab_s, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt((gx ** 2 + gy ** 2).sum(2)) / 8.0        # colour change per pixel across the boundary
    G = graph.rag_boundary(seg, edge.astype(np.float64))
    if 0 in G:
        G.remove_node(0)
    for u, v, d in G.edges(data=True):
        d["weight"] = float(d["weight"]) + 0.25 * float(np.linalg.norm(mean[u] - mean[v]))
    seeds = [int(x) for x in np.flatnonzero(known) if x in G]
    cost = np.zeros(S)
    if seeds:
        dist, paths = nx.multi_source_dijkstra(G, seeds)
        for node, path in paths.items():
            if not known[node]:
                owner[node] = owner[path[0]]
                cost[node] = dist[node]
        lost = [x for x in range(1, S) if area[x] > 0 and not known[x] and x not in paths]
    else:
        lost = [x for x in range(1, S) if area[x] > 0]
    owner = owner.astype(np.int64)
    if background is not None:
        bgl = np.abs(small.astype(np.int32) - np.array(background, np.int32)).max(2) <= 14
        light = np.bincount(seg[inside], weights=bgl[inside].astype(np.float64), minlength=S) / np.maximum(area, 1.0)
        owner[(light > 0.7) & ~known & ((cost > 25.0) | np.isin(np.arange(S), lost))] = GAP
    lab_small = owner[seg].astype(np.int16)
    lab_small[~inside] = -1
    if lost:                                                # pieces cut off from everything: the nearest owner
        cut = np.isin(seg, lost) & (lab_small != GAP)
        have = inside & ~cut & (lab_small >= 0)
        if have.any() and cut.any():
            idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
            lab_small[cut] = lab_small[idx[0][cut], idx[1][cut]]
    # thin spikes of one layer reaching into another (a sliver of a lock along a sleeve) join what surrounds them
    t = max(1, int(round(0.004 * max(sw, sh))))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))
    cores = [cv2.morphologyEx((lab_small == i).astype(np.uint8), cv2.MORPH_OPEN, ker) for i in range(n)]
    thin = (lab_small >= 0) & ~np.any([c > 0 for c in cores], axis=0)
    if thin.any():
        around = np.stack([cv2.boxFilter(c.astype(np.float32), -1, (8 * t + 1, 8 * t + 1)) for c in cores])
        lab_small[thin & (around.max(0) > 0)] = around.argmax(0)[thin & (around.max(0) > 0)]
    label = cv2.resize(lab_small, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int8)
    adopted = cv2.resize((~shown[seg] & inside).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    # rounding at the rim, and crumbs: a few stray pixels of one layer inside another's area join their surroundings
    small_ = om & (label == -1)
    floor = max(12.0, 0.00025 * float(om.sum()))
    for i in range(n):
        cnt, lab, st, _c = cv2.connectedComponentsWithStats(((label == i) & om).astype(np.uint8), connectivity=8)
        tiny = np.flatnonzero(st[:, cv2.CC_STAT_AREA] < floor)
        tiny = tiny[tiny > 0]
        if len(tiny):
            small_ |= np.isin(lab, tiny)
    solid = om & ~small_ & (label >= 0)
    if small_.any() and solid.any():
        idx = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
        label[small_] = label[idx[0][small_], idx[1][small_]]
    label[~om] = -1
    return label, adopted & om


def _face_zones(bgr, mask):
    """Faces drawn in a layer view (a face box with an eye in it, or the eyes with room around them)
    -> soft mask 0..1, None when there are none."""
    from . import vision
    pic = bgr.copy()
    pic[~mask] = 255
    eyes = vision.detect(pic, "eye")
    if not eyes:
        return None
    boxes = [f for f in (vision.detect(pic, "face") or [])
             if any(f[0] < (e[0] + e[2]) / 2 < f[2] and f[1] < (e[1] + e[3]) / 2 < f[3] for e in eyes)]
    for x0, y0, x1, y1, _s in eyes:
        if not any(f[0] < (x0 + x1) / 2 < f[2] and f[1] < (y0 + y1) / 2 < f[3] for f in boxes):
            gw, gh = 0.6 * (x1 - x0), 0.8 * (y1 - y0)
            boxes.append((x0 - gw, y0 - gh, x1 + gw, y1 + gh, 1.0))
    z = np.zeros(mask.shape, np.float32)
    for x0, y0, x1, y1, _s in boxes:
        z[max(0, int(y0)):int(y1) + 1, max(0, int(x0)):int(x1) + 1] = 1.0
    return cv2.GaussianBlur(z, (0, 0), 0.006 * max(mask.shape))


def _match_colours(src, ref, sel):
    """Undo the small overall colour shift an image model gives to what it copies: move the colours of `src` by
    the mean difference to `ref` over the pixels `sel` (never by much: this is a correction, not a recolouring)."""
    if sel.sum() < 200:
        return src
    a = _lab(src)
    shift = np.clip(_lab(ref)[sel].mean(0) - a[sel].mean(0), -10.0, 10.0)
    return cv2.cvtColor(np.clip(a + shift, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _mend(comp, own, hidden, total):
    """Small hidden patches inside a layer's visible area (hair behind an ear, a frill behind a strand) are not
    taken from the layer picture, whose shading never quite matches: they continue the original around them.
    Large hidden areas (a face under a fringe, a back behind hair) are what the layer picture is for.
    -> (picture, the patches that were mended)."""
    cnt, lab, st, _c = cv2.connectedComponentsWithStats(hidden.astype(np.uint8), connectivity=8)
    if cnt <= 1:
        return comp, np.zeros_like(hidden)
    ring = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
    edge = (ring > 0) & ~hidden                             # pixels just outside each patch
    touch_own = np.bincount(ring[edge & own], minlength=cnt).astype(np.float64)
    touch_all = np.maximum(np.bincount(ring[edge], minlength=cnt), 1.0)
    small = (st[:, cv2.CC_STAT_AREA] < 0.012 * total) & (touch_own / touch_all > 0.6)
    small[0] = False
    patch = small[lab]
    if not patch.any():
        return comp, patch
    h, w = hidden.shape
    z = min(1.0, 700.0 / max(h, w))                         # inpainting is slow and its result is smooth anyway
    sw, sh = max(8, int(round(w * z))), max(8, int(round(h * z)))
    hole = cv2.resize(patch.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
    hole = cv2.dilate(hole, np.ones((3, 3), np.uint8))
    filled = cv2.inpaint(cv2.resize(comp, (sw, sh), interpolation=cv2.INTER_AREA), hole, 4, cv2.INPAINT_TELEA)
    filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
    soft = cv2.GaussianBlur(cv2.dilate(patch.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 2.0)[:, :, None]
    return (filled * soft + comp * (1.0 - soft)).astype(np.uint8), patch


LAYER_TINTS = [(90, 170, 250), (235, 150, 70), (120, 220, 120), (220, 110, 220), (90, 220, 230), (150, 150, 255)]


def split(base_dir, items, bg="auto", tol=10.0, holes="auto", align=True, edge=None, regions=36, log=print):
    """Take the picture of a pack apart. items: [(name, layer picture, pack folder to write)], inner layer first.

    Every pixel of the original is given to one layer (see `claim`); a layer is then the original's own pixels
    wherever the original shows that layer, and the layer picture only where the layer was hidden behind
    another one. So nothing the original shows is ever lost or redrawn, whatever the image model did to it.
    Writes one pack per layer on the original's grid and layers_check.png. -> report dict."""
    from scipy import ndimage
    base_dir = os.path.abspath(base_dir)
    with open(os.path.join(base_dir, "ref.json")) as f:
        doc0 = json.load(f)
    o_sheet, o_mask = base_sheet(doc0, base_dir)
    bgc0 = None
    if str(doc0.get("background") or "").startswith("#"):
        hx = doc0["background"][1:]
        bgc0 = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
    placed = [place_layer(pic, doc0, o_sheet, o_mask, bg, tol, holes, align) for _n, pic, _d in items]
    docs = [dict(version=1, height=float(doc0["height"]), views={}, order=[], notes=pl["notes"], sources=[pl["image"]],
                 like=os.path.relpath(base_dir, os.path.abspath(d)), layer=name, guessed=None,
                 sheet_size=doc0.get("sheet_size"), align=pl["align"]) for (name, _p, d), pl in zip(items, placed)]
    recs = [dict() for _ in items]
    shown = [dict() for _ in items]
    stats = [dict(own=0.0, hidden=0.0, adopted=0.0, agreed=0.0, out=0.0, drawn=0.0) for _ in items]
    report = dict(views={}, layers={})
    rows = []
    faces_whole = []
    for n in doc0["order"]:
        v0 = doc0["views"][n]
        o_bgr = cv2.imread(os.path.join(base_dir, v0["texture"]), 1)
        om = cv2.imread(os.path.join(base_dir, v0["mask"]), 0) > 0
        h, w = om.shape
        on = [_on_base(pl, n, v0) for pl in placed]
        label, adopted = claim(o_bgr, om, on, background=bgc0)
        fe = max(2.0, 0.004 * max(h, w))
        tiles = []
        for i, ((name, pic, d), (lb, lm)) in enumerate(zip(items, on)):
            own = label == i
            hidden = lm & om & ~own & (label != GAP)
            st = stats[i]
            st["own"] += float(own.sum())
            st["hidden"] += float(hidden.sum())
            st["adopted"] += float((own & adopted).sum())
            st["out"] += float((lm & ~om).sum())
            st["drawn"] += float(lm.sum())
            st["agreed"] += float((own & lm & ~adopted).sum())
            mask = own | hidden
            if mask.sum() < 30:
                tiles.append(None)
                continue
            comp = o_bgr.copy()
            if hidden.any():
                lb = _match_colours(lb, o_bgr, own & lm & ~adopted)
                # the layer picture where the layer was hidden, the original where it shows; they meet inside
                # the shown part, a few pixels in from where the cover ends
                dist = cv2.distanceTransform((~hidden).astype(np.uint8), cv2.DIST_L2, 3)
                wl = np.where(hidden, 1.0, np.where(own & lm, np.clip(1.0 - dist / fe, 0.0, 1.0), 0.0))
                # a face is one thing: half the original's eye and half a redrawn one never meet exactly. A face
                # that was partly hidden is taken whole from the layer picture, which drew it complete
                zone = _face_zones(lb, lm)
                if zone is not None and (hidden & (zone > 0.5)).any():
                    wl = np.maximum(wl, np.where(lm, zone, 0.0))
                    faces_whole.append("%s/%s" % (name, n))
                wl = wl[:, :, None]
                comp = (lb * wl + o_bgr * (1.0 - wl)).astype(np.uint8)
                comp, mended = _mend(comp, own, hidden, float(mask.sum()))
                hidden = hidden & ~mended
            ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            pad = max(6, int(round(0.012 * max(h, w))))
            x0, y0, x1, y1 = int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])
            x0p, y0p, x1p, y1p = max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)
            m = mask[y0p:y1p + 1, x0p:x1p + 1].astype(np.uint8) * 255
            bx, by = v0["sheet_box"][0], v0["sheet_box"][1]
            c = dict(bgr=np.ascontiguousarray(comp[y0p:y1p + 1, x0p:x1p + 1]), mask=m,
                     box=(bx + x0p, by + y0p, bx + x1p, by + y1p), source=placed[i]["image"], bg=None,
                     bb=(x0 - x0p, y0 - y0p, x1 - x0p, y1 - y0p))
            shown[i][n] = own[y0p:y1p + 1, x0p:x1p + 1].astype(np.uint8) * 255
            c["h_px"], c["w_px"] = y1 - y0 + 1, x1 - x0 + 1
            _sx, sym = refmod.symmetry_axis(m, c["bgr"])
            rec = refmod._view_record(n, None, m, c["box"], c["source"])
            rec.update(px_per_unit=float(v0["px_per_unit"]),
                       origin_px=[float(v0["origin_px"][0] - x0p), float(v0["origin_px"][1] - y0p)],
                       symmetry=round(float(sym), 3), detail=round(float(refmod.detail_score(c["bgr"], m)), 2),
                       size_px=[int(m.shape[1]), int(m.shape[0])], source_scale=1.0)
            recs[i][n] = (rec, c)
            bgc = np.array(refmod.GRID_BG, np.float32)
            t = np.where(mask[:, :, None], comp, bgc).astype(np.uint8)
            t[hidden & (cv2.erode(hidden.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)] = (60, 60, 240)
            tiles.append(t)
        # the picture: original | who owns what | every layer (red line = from here on newly drawn)
        bgc = np.array(refmod.GRID_BG, np.float32)
        t0 = np.where(om[:, :, None], o_bgr, bgc).astype(np.uint8)
        t1 = (0.45 * t0 + 0.55 * bgc).astype(np.uint8)
        for i in range(len(items)):
            sel = label == i
            t1[sel] = (0.45 * t0[sel] + 0.55 * np.array(LAYER_TINTS[i % len(LAYER_TINTS)], np.float32)).astype(np.uint8)
        t1[adopted & (cv2.erode(adopted.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)] = (255, 255, 255)
        row = refmod.hstack([t0, t1] + [t for t in tiles if t is not None])
        rows.append((n, row))
        report["views"][n] = dict(adopted=round(float((adopted & (label >= 0)).sum()) / max(1.0, float(om.sum())), 4),
                                  gaps=round(float((label == GAP).sum()) / max(1.0, float(om.sum())), 4))
    for i, (name, _pic, d) in enumerate(items):
        if not recs[i]:
            raise SystemExit("the %s layer picture shows nothing where the views of %s are" % (name, base_dir))
        os.makedirs(d, exist_ok=True)
        for n, sh in shown[i].items():                      # where the original itself shows this layer
            recs[i][n][0]["shown"] = n + "_shown.png"
            refmod.write_image(os.path.join(d, n + "_shown.png"), sh)
        refmod._write_pack(docs[i], recs[i], d, edge, regions, docs[i]["notes"])
        st = stats[i]
        total = max(1.0, st["own"] + st["hidden"])
        report["layers"][name] = dict(original=round(st["own"] / total, 4), drawn=round(st["hidden"] / total, 4),
                                      adopted=round(st["adopted"] / max(1.0, st["own"]), 4),
                                      agreed=round(st["agreed"] / max(1.0, st["own"]), 4),
                                      outside=round(st["out"] / max(1.0, st["drawn"]), 4), align=placed[i]["align"],
                                      views=list(docs[i]["order"]))
    wmax = max(r.shape[1] for _n, r in rows)
    out = []
    for n, r in rows:
        r = cv2.copyMakeBorder(r, 26, 8, 0, wmax - r.shape[1], cv2.BORDER_CONSTANT, value=refmod.GRID_BG)
        refmod._put(r, "%s: original | who owns what (%s; white outline = no layer drew it, given to its neighbour) | "
                    "each layer (inside the red line: newly drawn, was hidden)" % (
                        n.upper(), ", ".join(name for name, _p, _d in items)), (6, 17), 0.5)
        out.append(r)
    sheet = np.vstack(out)
    if sheet.shape[1] > 2600:
        fz = 2600.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, None, fx=fz, fy=fz, interpolation=cv2.INTER_AREA)
    report["faces_whole"] = faces_whole
    report["picture"] = os.path.join(base_dir, "layers_check.png")
    refmod.write_image(report["picture"], sheet)
    return report
