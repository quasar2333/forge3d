"""Reference analysis: turn concept art / photos / turnaround sheets into a measured reference pack.

A pack is a folder with ref.json plus, per view, a cutout, a mask, a texture source (colours bled past the
edge), a measuring grid and a numbered region map. Every view carries the affine mapping between its pixels and
world coordinates, so the same numbers drive modelling (kit.ref), projection texturing and `f3d compare`.

World frame: Z up, the subject stands on z = 0 and faces -Y, x = 0 is its left/right symmetry plane.
"""
import json
import math
import os

import cv2
import numpy as np

# view name -> (azimuth, elevation, world axis shown to the right in the picture, world axis shown upward)
VIEW_AXES = {
    "front":  (0, 0, (1, 0, 0), (0, 0, 1)),
    "back":   (180, 0, (-1, 0, 0), (0, 0, 1)),
    "right":  (90, 0, (0, 1, 0), (0, 0, 1)),     # camera on the subject's +X side: the subject faces LEFT in the picture
    "left":   (-90, 0, (0, -1, 0), (0, 0, 1)),   # camera on the -X side: the subject faces RIGHT in the picture
    "top":    (0, 90, (1, 0, 0), (0, 1, 0)),     # front of the subject at the bottom of the picture
    "bottom": (0, -90, (1, 0, 0), (0, -1, 0)),
}
ALIASES = {"side": "right", "profile": "right", "rear": "back", "behind": "back", "face": "front",
           "side_left": "left", "side_right": "right", "faces_left": "right", "faces_right": "left"}
SKIP = ("skip", "-", "none", "x", "34", "three_quarter", "persp", "detail")
GRID_BG = (150, 146, 142)


def canonical_view(name):
    n = name.strip().lower().replace("-", "_")
    n = ALIASES.get(n, n)
    if n in SKIP:
        return None
    if n not in VIEW_AXES:
        raise SystemExit("unknown view name %r (use front, back, right, left, top, bottom, or skip). "
                         "right = subject faces LEFT in the picture, left = subject faces RIGHT." % name)
    return n


# ---------------------------------------------------------------- matting
def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise SystemExit("cannot read image: %s" % path)
    if im.dtype != np.uint8:
        im = (im.astype(np.float32) / float(np.iinfo(im.dtype).max) * 255).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


def write_image(path, im):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", im)
    if not ok:
        raise RuntimeError("cannot encode %s" % path)
    buf.tofile(path)
    return path


def _border(im, width=3):
    return np.concatenate([im[:width].reshape(-1, im.shape[2]), im[-width:].reshape(-1, im.shape[2]),
                           im[:, :width].reshape(-1, im.shape[2]), im[:, -width:].reshape(-1, im.shape[2])])


def matte(im, bg="auto", tol=10.0, holes="auto"):
    """BGRA image -> (mask uint8 0/255, info). Background removal for flat / transparent backgrounds.

    bg: 'auto' (alpha if the image has it, else the border colour), 'alpha', '#rrggbb', or 'rembg' (neural matting
    for photos and busy backgrounds; needs `f3d setup --extra rembg`).
    holes: 'auto' cuts enclosed areas of pure background colour out of the subject, 'keep' leaves them solid, or a
    minimum area as a fraction of the subject.
    """
    h, w = im.shape[:2]
    alpha = im[:, :, 3]
    info = {}
    use_alpha = bg == "alpha" or (bg == "auto" and float((alpha < 128).mean()) > 0.02)
    if use_alpha:
        mask = (alpha >= 128).astype(np.uint8) * 255
        info["background"] = "alpha"
    elif bg == "rembg":
        try:
            from rembg import remove
        except ImportError:
            raise SystemExit("rembg is not installed: run `python3 f3d.py setup --extra rembg` first")
        out = remove(cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB))
        mask = (np.asarray(out)[:, :, 3] >= 128).astype(np.uint8) * 255
        info["background"] = "rembg"
    else:
        bgr = im[:, :, :3].astype(np.int32)
        if bg == "auto":
            b = _border(im[:, :, :3]).astype(np.int32)
            color = np.median(b, axis=0)
            spread = float(np.percentile(np.abs(b - color).max(1), 90))
            info["border_spread"] = round(spread, 1)
            if spread > 3 * tol:
                info["warning"] = ("the border is not a flat colour (spread %.0f): the cutout may be wrong. For photos "
                                   "or busy backgrounds use --bg rembg, or supply an image with transparency." % spread)
        else:
            s = bg.lstrip("#")
            color = np.array([int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)], dtype=np.float64)
        info["background"] = "#%02x%02x%02x" % (int(color[2]), int(color[1]), int(color[0]))
        dist = np.abs(bgr - color).max(2)
        bg_like = (dist <= tol).astype(np.uint8)
        # leak guard: the outside region may only enter through openings wider than the guard width, so thin breaks
        # in line art do not let it flood white clothes, eyes or paper-coloured props
        k = max(1, int(round(min(h, w) * 0.004)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        core = cv2.erode(cv2.copyMakeBorder(bg_like, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1), ker)
        n, lab = cv2.connectedComponents(core, connectivity=4)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        outside = np.isin(lab, edge[edge > 0]).astype(np.uint8)[1:-1, 1:-1]
        grown = cv2.dilate(outside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 3, 2 * k + 3)))
        mask = ((grown & bg_like) == 0).astype(np.uint8) * 255
        if holes != "keep":
            # enclosed areas of pure background colour are see-through gaps (inside a loop of hair, between an arm
            # and the body, a cup handle); shaded or off-white areas (clothes, eyes, paper props) stay solid
            frac = 0.0003 if holes == "auto" else float(holes)
            inside = (bg_like & (mask > 0)).astype(np.uint8)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=4)
            fg_area = float((mask > 0).sum())
            pure_tol = max(2.0, tol * 0.25)
            count = 0
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] < max(16.0, frac * fg_area):
                    continue
                comp = lab == i
                if float(np.median(dist[comp])) <= pure_tol:
                    mask[comp] = 0
                    count += 1
            if count:
                rim = cv2.dilate((mask == 0).astype(np.uint8), np.ones((3, 3), np.uint8)) & bg_like
                mask[rim > 0] = 0
                info["holes"] = count
    # drop specks (dust, jpeg noise, stray marks)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= max(12, 0.0004 * areas.max()))[0] + 1
        mask = np.isin(lab, keep).astype(np.uint8) * 255
    info["coverage"] = round(float((mask > 0).mean()), 4)
    return mask, info


# ---------------------------------------------------------------- view splitting
def _runs(occ, min_gap):
    """Indices where occ is true -> [(start, end)] merging gaps narrower than min_gap."""
    idx = np.flatnonzero(occ)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 >= min_gap:
            runs.append((int(start), int(prev)))
            start = i
        prev = i
    runs.append((int(start), int(prev)))
    return runs


def split_views(mask, min_gap=None):
    """Recursive XY cut of a sheet into views. Returns [(x0, y0, x1, y1)] in reading order."""
    h, w = mask.shape
    if min_gap is None:
        min_gap = max(4, int(round(0.008 * max(h, w))))
    boxes = []

    def cut(x0, y0, x1, y1, axis, tried):
        sub = mask[y0:y1 + 1, x0:x1 + 1] > 0
        if not sub.any():
            return
        if axis == 0:
            runs = _runs(sub.any(0), min_gap)
            parts = [(x0 + a, y0, x0 + b, y1) for a, b in runs]
        else:
            runs = _runs(sub.any(1), min_gap)
            parts = [(x0, y0 + a, x1, y0 + b) for a, b in runs]
        if len(parts) <= 1:
            if tried:
                s = mask[y0:y1 + 1, x0:x1 + 1] > 0
                ys, xs = np.flatnonzero(s.any(1)), np.flatnonzero(s.any(0))
                boxes.append((x0 + int(xs[0]), y0 + int(ys[0]), x0 + int(xs[-1]), y0 + int(ys[-1])))
            else:
                cut(x0, y0, x1, y1, 1 - axis, True)
            return
        for p in parts:
            cut(p[0], p[1], p[2], p[3], 1 - axis, False)

    cut(0, 0, w - 1, h - 1, 0, False)
    if not boxes:
        return []
    area = [int((mask[b[1]:b[3] + 1, b[0]:b[2] + 1] > 0).sum()) for b in boxes]
    big = [i for i, a in enumerate(area) if a >= 0.15 * max(area)]
    merged = {i: list(boxes[i]) for i in big}
    for i, b in enumerate(boxes):                     # stray bits (floating props, detached hair) join the nearest view
        if i in merged:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

        def gap(j):
            m = merged[j]
            return max(m[0] - cx, 0, cx - m[2]) + max(m[1] - cy, 0, cy - m[3])
        j = min(merged, key=gap)
        m = merged[j]
        merged[j] = [min(m[0], b[0]), min(m[1], b[1]), max(m[2], b[2]), max(m[3], b[3])]
    out = _split_stacked(mask, [tuple(merged[i]) for i in big])
    rows = []                                          # reading order: rows top to bottom, then left to right
    for b in sorted(out, key=lambda b: b[1]):
        for r in rows:
            if b[1] <= r["y1"] - 0.5 * (r["y1"] - r["y0"]):
                r["items"].append(b)
                r["y1"] = max(r["y1"], b[3])
                break
        else:
            rows.append({"y0": b[1], "y1": b[3], "items": [b]})
    return [b for r in rows for b in sorted(r["items"], key=lambda b: b[0])]


def _split_stacked(mask, boxes):
    """Views of a grid sheet that touch (the feet of one picture reach the wing tips of the one below) come out
    of the cut as one tall box. Where the neighbouring column shows a gap between two pictures, and this box is
    nearly empty at that height too, it is two pictures: cut it there. The same for rows."""
    boxes = list(boxes)
    for _ in range(8):
        hit = None
        for i, b in enumerate(boxes):
            for axis in (1, 0):                                    # 1: stacked vertically, 0: side by side
                a0, a1, c0, c1 = (b[1], b[3], b[0], b[2]) if axis == 1 else (b[0], b[2], b[1], b[3])
                span = a1 - a0 + 1
                sub = mask[b[1]:b[3] + 1, b[0]:b[2] + 1] > 0
                occ = sub.sum(1 if axis == 1 else 0).astype(np.float64)
                typical = float(np.median(occ[occ > 0])) if (occ > 0).any() else 0.0
                others = sorted([o for j, o in enumerate(boxes) if j != i], key=lambda o: o[1] if axis == 1 else o[0])
                for o1, o2 in zip(others[:-1], others[1:]):
                    e1, s2 = (o1[3], o2[1]) if axis == 1 else (o1[2], o2[0])
                    p1, q1, p2, q2 = (o1[0], o1[2], o2[0], o2[2]) if axis == 1 else (o1[1], o1[3], o2[1], o2[3])
                    if s2 <= e1 or min(q1, q2) < max(p1, p2):       # no gap, or not in one column / row
                        continue
                    mid = (e1 + s2) // 2
                    if not a0 + 0.25 * span < mid < a1 - 0.25 * span:
                        continue
                    reach = max(3, int(round(0.06 * span)))
                    lo, hi = max(1, mid - a0 - reach), min(span - 2, mid - a0 + reach)
                    k = lo + int(np.argmin(occ[lo:hi + 1]))
                    if typical > 0 and occ[k] <= 0.25 * typical:
                        hit = (i, axis, a0 + k)
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            break
        i, axis, cut = hit
        b = boxes.pop(i)
        halves = [(b[0], b[1], b[2], cut - 1), (b[0], cut, b[2], b[3])] if axis == 1 else \
                 [(b[0], b[1], cut - 1, b[3]), (cut, b[1], b[2], b[3])]
        for h in halves:
            sub = mask[h[1]:h[3] + 1, h[0]:h[2] + 1] > 0
            if sub.any():
                ys, xs = np.flatnonzero(sub.any(1)), np.flatnonzero(sub.any(0))
                boxes.append((h[0] + int(xs[0]), h[1] + int(ys[0]), h[0] + int(xs[-1]), h[1] + int(ys[-1])))
    return boxes


# ---------------------------------------------------------------- per-view measurements
def symmetry_axis(mask, bgr=None):
    """Column about which the subject mirrors best (silhouette and, if given, colours). -> (x, score 0..1)."""
    h, w = mask.shape
    f = 160.0 / max(h, w)
    size = (max(8, int(round(w * f))), max(8, int(round(h * f))))
    m = cv2.resize((mask > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    img = None
    if bgr is not None:
        img = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sw = size[0]
    best = (sw / 2.0, -1.0)
    for c2 in range(int(sw * 0.7), int(sw * 1.3) + 1):          # c2 = 2 * axis position
        idx = c2 - np.arange(sw)
        ok = (idx >= 0) & (idx < sw)
        fm = np.zeros_like(m)
        fm[:, ok] = m[:, idx[ok]]
        union = np.maximum(m, fm).sum()
        if union <= 0:
            continue
        both = np.minimum(m, fm)
        if img is not None:
            fi = np.zeros_like(img)
            fi[:, ok] = img[:, idx[ok]]
            both = both * (1.0 - np.minimum(1.0, np.abs(img - fi).mean(2) * 2.5))
        score = float(both.sum() / union)
        if score > best[1]:
            best = (c2 / 2.0, score)
    idx = int(round(best[0] * 2)) - np.arange(sw)                # report the plain silhouette score at that axis
    ok = (idx >= 0) & (idx < sw)
    fm = np.zeros_like(m)
    fm[:, ok] = m[:, idx[ok]]
    union = np.maximum(m, fm).sum()
    return best[0] * w / sw, float(np.minimum(m, fm).sum() / union) if union > 0 else 0.0


def footprint_center(mask, frac=0.04):
    ys = np.flatnonzero((mask > 0).any(1))
    y1 = ys[-1]
    y0 = max(ys[0], int(y1 - frac * (y1 - ys[0] + 1)))
    xs = np.flatnonzero((mask[y0:y1 + 1] > 0).any(0))
    return (xs[0] + xs[-1]) / 2.0


def detail_score(bgr, mask):
    """Mean edge strength inside the silhouette (faces, prints and panel lines score high; plain backs low)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    return float(e[inner].mean()) if inner.any() else 0.0


def _alike(a, b, size=128):
    """IoU of two silhouettes brought to one height and laid over each other at their symmetry axes (0 when a
    mask is missing). A second pose of the same side scores high even when a tail or an arm swings out."""
    ms = []
    for v in (a, b):
        m, bb = v.get("mask"), v.get("bb")
        if m is None or bb is None:
            return 0.0
        f = size / float(bb[3] - bb[1] + 1)
        crop = (m[bb[1]:bb[3] + 1] > 0).astype(np.uint8)
        small = cv2.resize(crop, (max(1, int(round(crop.shape[1] * f))), size), interpolation=cv2.INTER_NEAREST) > 0
        canvas = np.zeros((size, 6 * size), bool)
        x0 = 3 * size - int(round(v.get("sym_x", crop.shape[1] / 2.0) * f))
        x0 = min(max(x0, 0), canvas.shape[1] - small.shape[1])
        canvas[:, x0:x0 + small.shape[1]] = small[:, :canvas.shape[1] - x0]
        ms.append(canvas)
    return float((ms[0] & ms[1]).sum()) / max(1.0, float((ms[0] | ms[1]).sum()))


def guess_names(views):
    """Heuristic view labels for unlabelled sheets. views: list of dicts with sym, detail, aspect. -> (names, note)."""
    n = len(views)
    if n == 1:
        return ["front"], "single image: assumed to be the front view"
    sym = [v["sym"] for v in views]
    order = np.argsort(sym)[::-1]
    names = [None] * n
    symmetric = [i for i in order if sym[i] >= 0.86]
    if n == 2:
        if len(symmetric) == 2:
            a, b = sorted(symmetric, key=lambda i: -views[i]["detail"])
            names[a], names[b] = "front", "back"
        else:
            names[order[0]], names[order[1]] = "front", "right"
        return names, "guessed from silhouette symmetry"
    if len(symmetric) >= 2:
        a, b = sorted(symmetric[:2], key=lambda i: -views[i]["detail"])
        names[a], names[b] = "front", "back"
    else:
        names[order[0]] = "front"
    rest = [i for i in range(n) if names[i] is None]
    side_names = ["right", "left", "top", "bottom"]
    extra = 0
    head_on = [j for j in range(n) if names[j] in ("front", "back")]
    side_sym = min([views[i]["sym"] for i in rest], default=1.0)
    for i in rest:
        v = views[i]
        # a sheet with more pictures than views: a second pose of the back (tail swung aside), a colour variant.
        # Such a picture mirrors about as well as the front, clearly better than the real side view, and has
        # nearly the outline of the front or back picture
        if len(rest) >= 2 and len(head_on) == 2 and v["sym"] >= 0.72 and v["sym"] - side_sym >= 0.2 and \
                max(_alike(v, views[j]) for j in head_on) >= 0.6:
            extra += 1
            continue
        if v["aspect"] < 0.6 and "top" not in names and n >= 4:      # a squat outline on a turnaround is usually the top
            names[i] = "top"
            continue
        for s in side_names:
            if s not in names:
                names[i] = s if s in ("right", "left") else None
                break
        if names[i] in ("right", "left"):
            names[i] = "right" if v["facing"] <= 0 else "left"
            if names[i] in [x for j, x in enumerate(names) if j != i]:
                names[i] = "left" if names[i] == "right" else "right"
    note = "guessed from silhouette symmetry and detail"
    if extra:
        note += "; %d more picture(s) seen head-on (a second pose or variant of the front / back) were left out, " \
                "name them with --views if they are real views" % extra
    return names, note


def facing_guess(bgr, mask):
    """-1 if the detailed (face) half of the upper silhouette is on the picture's left, +1 if on the right."""
    h = mask.shape[0]
    top = slice(0, max(4, int(h * 0.5)))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    e = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    inner = (cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0)[top]
    e = e[top] * inner
    xs = np.flatnonzero(inner.any(0))
    if len(xs) < 4:
        return 0
    mid = (xs[0] + xs[-1]) // 2
    left, right = e[:, :mid].sum(), e[:, mid:].sum()
    return -1 if left > right * 1.1 else (1 if right > left * 1.1 else 0)


def bleed(bgr, mask, erode=1):
    """Fill everything outside the silhouette with the nearest inside colour (texture padding)."""
    from scipy import ndimage
    inner = cv2.erode(mask, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0 if erode else mask > 0
    if not inner.any():
        inner = mask > 0
    idx = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True)
    return bgr[idx[0], idx[1]]


def cel_flat(bgr, radius):
    """The picture as flat colour areas: thin dark lines (ink outlines, hatching, lashes) are closed up with the
    colours beside them, then every area is evened out while its edges stay crisp. This is what a surface shows
    where the full drawing cannot be trusted (seen at a slant, hidden from every view): calm, in the right
    colours, never a smear. Returns (flat BGR image, mask of the line pixels that were removed)."""
    from scipy import ndimage
    r = max(1, int(round(radius)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    line = (cv2.morphologyEx(lum, cv2.MORPH_CLOSE, ker).astype(np.int16) - lum) > 18
    if line.all():
        return bgr.copy(), line
    idx = ndimage.distance_transform_edt(line, return_distances=False, return_indices=True)
    closed = np.ascontiguousarray(bgr[idx[0], idx[1]])
    h, w = closed.shape[:2]
    if max(h, w) <= 1000:
        return cv2.pyrMeanShiftFiltering(closed, sp=2.5 * r, sr=18, maxLevel=1), line
    # evening out is by far the slowest step and its result is flat by nature: do it at half size
    small = cv2.resize(closed, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    small = cv2.pyrMeanShiftFiltering(small, sp=1.25 * r, sr=18, maxLevel=1)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), line


def ink_of(picture, mask, px_per_unit, height=1.0):
    """Does a cutout picture have ink outlines? -> dict(color='#rrggbb', width=<world units>) or None.

    picture: BGR uint8 cutout; mask: bool silhouette. Reads how light the picture is at each depth inside the
    silhouette edge: line art gives a dark band a few pixels wide right behind the edge (after the antialiased
    fringe), a photograph or render does not."""
    m = np.ascontiguousarray(mask.astype(np.uint8))
    if m.sum() < 400:
        return None
    lum = cv2.cvtColor(np.ascontiguousarray(picture[:, :, :3]), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    depth = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    kmax = max(6, int(round(0.02 * height * px_per_unit)))
    inner = lum[(depth > kmax) & (depth <= 2.5 * kmax)]
    if len(inner) < 100:
        return None
    l_in = float(np.median(inner))
    ring = np.clip(np.ceil(depth).astype(np.int32), 0, kmax + 1)
    count = np.bincount(ring.ravel(), minlength=kmax + 2)[1:kmax + 1]
    prof = np.bincount(ring.ravel(), weights=lum.ravel(), minlength=kmax + 2)[1:kmax + 1] / np.maximum(count, 1)
    prof[count == 0] = l_in
    half = max(3, kmax // 2)
    k0 = int(np.argmin(prof[:half]))
    l_edge = float(prof[k0])
    if l_in - l_edge < 28:                              # no dark band behind the edge
        return None
    mid = (l_in + l_edge) / 2.0
    after = np.flatnonzero(prof[k0:] > mid)
    if len(after) == 0:
        return None
    before = np.flatnonzero(prof[:k0] > mid)            # the light fringe where the edge blends into the paper
    start = int(before[-1]) + 1 if len(before) else 0
    end = k0 + int(after[0])
    width_px = float(end - start)
    band = (ring > start) & (ring <= end)
    dark = band & (lum < mid)
    if dark.sum() < 0.55 * band.sum() or width_px / px_per_unit > 0.014 * height:
        return None
    b, g, r = (int(round(float(np.median(picture[:, :, i][dark])))) for i in range(3))
    return dict(color="#%02x%02x%02x" % (r, g, b), width=width_px / px_per_unit)


def pack_ink(doc, out_dir):
    """The line art of a pack (the median over its views): dict(color, width) or None."""
    found = []
    for n in doc["order"]:
        v = doc["views"][n]
        pic = cv2.imread(os.path.join(out_dir, v["image"]), -1)
        mask = cv2.imread(os.path.join(out_dir, v["mask"]), 0)
        found.append(ink_of(pic, mask > 0, float(v["px_per_unit"]), float(doc["height"]))
                     if pic is not None and mask is not None else None)
    hits = sorted([f for f in found if f], key=lambda f: f["width"])
    if not hits or len(hits) * 2 < len(found):
        return None
    return hits[len(hits) // 2]


def soft_alpha(bgr, mask, bg_color):
    """Antialiased alpha: hard inside, falling off over the one-pixel rim according to how far the colour is from bg."""
    a = (mask > 0).astype(np.float32)
    if bg_color is None:
        return (cv2.GaussianBlur(a, (3, 3), 0) * 255).astype(np.uint8)
    rim = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0) & ~(cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0)
    d = np.abs(bgr.astype(np.float32) - np.array(bg_color, np.float32)).max(2) / 60.0
    a[rim] = np.clip(d[rim], 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def row_runs(mask, n=100):
    """Silhouette spans at n heights, bottom to top: [[z_frac, [[x0, x1], ...]], ...] in pixels of the crop."""
    h = mask.shape[0]
    out = []
    for i in range(n):
        zf = (i + 0.5) / n
        y = int(round((1 - zf) * (h - 1)))
        runs = _runs(mask[y] > 0, 2)
        out.append((zf, runs))
    return out


# ---------------------------------------------------------------- palette and regions
def palette_of(bgr, mask, k=10, seed=1):
    """Dominant colours inside the mask. -> [(hex, share)] sorted by share, and (labels image, centres BGR)."""
    pts = bgr[mask > 0].reshape(-1, 3)
    if len(pts) == 0:
        return [], None, None
    lab = cv2.cvtColor(pts.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(seed)
    sample = lab if len(lab) <= 60000 else lab[rng.choice(len(lab), 60000, replace=False)]
    k = int(min(k, max(1, len(np.unique(sample.astype(np.int16), axis=0)))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    cv2.setRNGSeed(seed)
    _c, _l, centres = cv2.kmeans(sample, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    d = ((lab[:, None, :] - centres[None, :, :]) ** 2).sum(2) if len(lab) * k < 4e7 else None
    if d is None:
        labels = np.empty(len(lab), np.int32)
        for s in range(0, len(lab), 200000):
            labels[s:s + 200000] = ((lab[s:s + 200000, None, :] - centres[None]) ** 2).sum(2).argmin(1)
    else:
        labels = d.argmin(1)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    order = np.argsort(-counts)
    cb = cv2.cvtColor(centres.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_LAB2BGR).reshape(-1, 3)
    pal = [("#%02x%02x%02x" % (int(cb[i][2]), int(cb[i][1]), int(cb[i][0])), float(counts[i] / counts.sum()))
           for i in order if counts[i] > 0]
    remap = np.empty(k, np.int32)
    remap[order] = np.arange(k)
    lab_img = np.full(mask.shape, -1, np.int32)
    lab_img[mask > 0] = remap[labels]
    return pal, lab_img, cb[order]


def regions_of(bgr, mask, max_regions=36, k=9):
    """Flat-colour regions for set-of-mark prompting. -> (region label image int32 (0 = none), [region dicts])."""
    h, w = mask.shape
    fg = mask > 0
    if not fg.any():
        return np.zeros((h, w), np.int32), []
    smooth = cv2.pyrMeanShiftFiltering(bgr, sp=max(3, int(round(max(h, w) * 0.008))), sr=18)
    _pal, lab_img, centres = palette_of(smooth, mask, k=k)
    area_fg = float(fg.sum())
    min_area = max(30.0, 0.0035 * area_fg)
    comps = []
    for c in range(len(centres)):
        m = (lab_img == c).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                comps.append((int(stats[i, cv2.CC_STAT_AREA]), c, lab == i))
    comps.sort(key=lambda t: -t[0])
    comps = comps[:max_regions]
    comps.sort(key=lambda t: (np.flatnonzero(t[2].any(1))[0], np.flatnonzero(t[2].any(0))[0]))   # top to bottom
    label = np.zeros((h, w), np.int32)
    out = []
    for rid, (area, c, m) in enumerate(comps, 1):
        label[m] = rid
        ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
        py, px = np.unravel_index(int(dist.argmax()), dist.shape)
        col = bgr[m].mean(0)
        out.append(dict(id=rid, box_px=[int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])], area=area,
                        share=round(area / area_fg, 4), pin_px=[int(px), int(py)],
                        color="#%02x%02x%02x" % (int(col[2]), int(col[1]), int(col[0]))))
    return label, out


# ---------------------------------------------------------------- drawing
def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1, plate=(35, 35, 35)):
    """Text on a solid plate (stays legible over any artwork and after downscaling)."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if plate is not None:
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), plate, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fmt(v):
    s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _nice_step(span, target=10):
    raw = span / target
    e = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= raw:
            return m * e
    return 10 * e


def over_bg(bgra, bg=GRID_BG):
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (bgra[:, :, :3].astype(np.float32) * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)


def canvas_for(view, cut, target=900, margin=46):
    """Upscaled cutout on a neutral canvas with room for axis labels. -> (canvas, to_px(u, v), zoom)."""
    h, w = cut.shape[:2]
    zoom = min(4.0, max(1.0, target / float(max(h, w))))
    big = cv2.resize(cut, (int(round(w * zoom)), int(round(h * zoom))), interpolation=cv2.INTER_LANCZOS4)
    H, W = big.shape[:2]
    canvas = np.full((H + 2 * margin, W + 2 * margin, 3), GRID_BG, np.uint8)
    canvas[margin:margin + H, margin:margin + W] = over_bg(big)
    s = view["px_per_unit"] * zoom
    ox, oy = view["origin_px"][0] * zoom + margin, view["origin_px"][1] * zoom + margin

    def to_px(u, v):
        return int(round(ox + u * s)), int(round(oy - v * s))
    return canvas, to_px, zoom, margin


def draw_grid(view, cut, title):
    canvas, to_px, zoom, margin = canvas_for(view, cut)
    H, W = canvas.shape[:2]
    u0, v0, u1, v1 = view["frame"]
    step = _nice_step(max(u1 - u0, v1 - v0), 10)
    ua, va = view["u_axis"], view["v_axis"]
    k = math.floor(u0 / step)
    while k * step <= u1 + 1e-9:
        u = k * step
        x, _ = to_px(u, 0)
        if margin <= x <= W - margin:
            zero = abs(u) < 1e-9
            cv2.line(canvas, (x, margin), (x, H - margin), (60, 60, 235) if zero else (236, 236, 236), 2 if zero else 1,
                     cv2.LINE_AA)
            _put(canvas, _fmt(u), (x - 12, H - margin + 18), 0.5, (255, 255, 255), 1)
            _put(canvas, _fmt(u), (x - 12, margin - 8), 0.5, (255, 255, 255), 1)
        k += 1
    k = math.floor(v0 / step)
    while k * step <= v1 + 1e-9:
        v = k * step
        _, y = to_px(0, v)
        if margin <= y <= H - margin:
            zero = abs(v) < 1e-9
            cv2.line(canvas, (margin, y), (W - margin, y), (60, 60, 235) if zero else (236, 236, 236), 2 if zero else 1,
                     cv2.LINE_AA)
            _put(canvas, _fmt(v), (4, y + 5), 0.5, (255, 255, 255), 1)
            _put(canvas, _fmt(v), (W - margin + 4, y + 5), 0.5, (255, 255, 255), 1)
        k += 1
    _put(canvas, "%s   right = %s   up = %s" % (title, ua, va), (margin, 18), 0.55, (120, 255, 255), 1)
    return canvas


def draw_marks(view, cut, label, regions, title):
    canvas, to_px, zoom, margin = canvas_for(view, cut)
    h, w = label.shape
    big = cv2.resize(label.astype(np.float32), (int(round(w * zoom)), int(round(h * zoom))),
                     interpolation=cv2.INTER_NEAREST).astype(np.int32)
    edge = np.zeros(big.shape, bool)
    edge[:-1] |= big[:-1] != big[1:]
    edge[:, :-1] |= big[:, :-1] != big[:, 1:]
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    roi = canvas[margin:margin + big.shape[0], margin:margin + big.shape[1]]
    roi[edge] = (40, 255, 255)
    for r in regions:
        x, y = int(r["pin_px"][0] * zoom) + margin, int(r["pin_px"][1] * zoom) + margin
        txt = str(r["id"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x - tw // 2 - 4, y - th // 2 - 5), (x + tw // 2 + 4, y + th // 2 + 5), (0, 0, 0), -1)
        cv2.putText(canvas, txt, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    _put(canvas, "%s regions (ids for kit ref masks)" % title, (margin, 18), 0.55, (120, 255, 255), 1)
    return canvas


def hstack(images, bg=GRID_BG, gap=8):
    h = max(i.shape[0] for i in images)
    w = sum(i.shape[1] for i in images) + gap * (len(images) - 1)
    out = np.full((h, w, 3), bg, np.uint8)
    x = 0
    for i in images:
        out[:i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


# ---------------------------------------------------------------- pack
def _view_record(name, crop_bgra, mask, box, source):
    az, el, ua, va = VIEW_AXES[name]
    return dict(name=name, az=az, el=el, u_vec=list(ua), v_vec=list(va),
                u_axis=_axis_label(ua), v_axis=_axis_label(va), sheet_box=[int(b) for b in box], source=source)


def _axis_label(vec):
    i = int(np.argmax(np.abs(vec)))
    return ("+" if vec[i] > 0 else "-") + "XYZ"[i]


def enlarge(im, mask, target=1600.0, log=None, factor=None):
    """Small pictures are redrawn larger by a super-resolution model before anything is measured (see
    forge3d.vision): the outlines get smoother, and the painted model shows crisp lines instead of a blur.
    target: wanted size of the tallest view in pixels; factor: enlarge by exactly this much instead.
    -> (image, factor); factor 1.0 = left as it was (large enough, or the model is not available)."""
    boxes = split_views(mask)
    if not boxes:
        return im, 1.0
    tallest = max(max(b[3] - b[1], b[2] - b[0]) + 1 for b in boxes)
    f = min(4.0, float(factor) if factor else target / float(tallest))
    if f < 1.3:
        return im, 1.0
    if f > 3.3 and not factor:
        f = 4.0
    from . import vision
    src = im
    if float((im[:, :, 3] < 128).mean()) > 0.02:             # transparent surroundings carry no usable colour
        src = np.dstack([bleed(im[:, :, :3], mask, 0), im[:, :, 3]])
    big = vision.upscale(src, f, keep=mask > 0)
    if big is None:
        if log:
            log("the picture is small (views about %d px) and the super-resolution model is not available (%s): "
                "textures will be soft" % (tallest, vision.why_not("upscale4x")))
        return im, 1.0
    return big, float(big.shape[0]) / float(im.shape[0])


def sharp_matte(im, bg="auto", tol=10.0, holes="auto", sharpen="auto", log=None, factor=None):
    """Cut the subject out and, when the picture is small, enlarge it first. -> (image, mask, info, factor)."""
    mask, info = matte(im, bg, tol, holes)
    if sharpen == "off" or not mask.any() or bg == "rembg":
        return im, mask, info, 1.0
    big, f = enlarge(im, mask, log=log, factor=factor)
    if f <= 1.0:
        return im, mask, info, 1.0
    # the outline follows the sharp picture; what counts as a see-through gap was decided on the original,
    # whose background is exactly one colour
    fine = matte(big, bg, tol, holes)[0]
    fine[cv2.resize(mask, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_CUBIC) < 64] = 0
    return big, fine, info, f


def build_pack(images, out_dir, views=None, height=1.0, bg="auto", tol=10.0, scale="auto", edge=None, pad=6,
               regions=36, holes="auto", sharpen="auto"):
    """Analyse reference image(s) and write a reference pack. Returns the ref.json dict."""
    os.makedirs(out_dir, exist_ok=True)
    cuts, notes = [], []
    factors, backgrounds = {}, []
    for path in images:
        im, mask, info, f = sharp_matte(read_image(path), bg, tol, holes, sharpen, notes.append)
        factors[os.path.abspath(path)] = f
        backgrounds.append(info["background"])
        if f > 1.0:
            pad = max(pad, int(round(6 * f)))
            notes.append("%s was small: redrawn %.3gx larger by a super-resolution model before measuring (same "
                         "picture, crisp lines); pixel positions you read off the original still work with "
                         "view.px()" % (os.path.basename(path), f))
        if info.get("holes"):
            notes.append("%s: %d enclosed background-coloured area(s) were cut out as see-through gaps; if one of them is "
                         "really a white part of the subject, re-run with --holes keep" % (os.path.basename(path), info["holes"]))
        if info.get("warning"):
            notes.append("%s: %s" % (os.path.basename(path), info["warning"]))
        if not mask.any():
            raise SystemExit("no subject found in %s (the whole image looks like background); try --bg '#rrggbb' or "
                             "--bg rembg" % path)
        bgc = None
        if info["background"].startswith("#"):
            s = info["background"][1:]
            bgc = (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))
        for box in split_views(mask):
            x0, y0, x1, y1 = box
            x0p, y0p = max(0, x0 - pad), max(0, y0 - pad)
            x1p, y1p = min(im.shape[1] - 1, x1 + pad), min(im.shape[0] - 1, y1 + pad)
            m = mask[y0p:y1p + 1, x0p:x1p + 1].copy()
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)       # neighbours' bits in the pad
            if n > 2:
                keep = [i for i in range(1, n) if not (stats[i, cv2.CC_STAT_AREA] < 0.02 * stats[1:, cv2.CC_STAT_AREA].max()
                        and (stats[i, cv2.CC_STAT_LEFT] == 0 or stats[i, cv2.CC_STAT_TOP] == 0
                             or stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] == m.shape[1]
                             or stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] == m.shape[0]))]
                m = np.isin(lab, keep).astype(np.uint8) * 255
            bgr = im[y0p:y1p + 1, x0p:x1p + 1, :3].copy()
            cuts.append(dict(bgr=bgr, mask=m, box=(x0p, y0p, x1p, y1p), source=os.path.abspath(path), bg=bgc,
                             single_sheet=len(images) == 1))
    if not cuts:
        raise SystemExit("no views found")
    for c in cuts:
        ys, xs = np.flatnonzero(c["mask"].any(1)), np.flatnonzero(c["mask"].any(0))
        c["bb"] = (int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1]))
        c["h_px"], c["w_px"] = c["bb"][3] - c["bb"][1] + 1, c["bb"][2] - c["bb"][0] + 1
        c["sym_x"], c["sym"] = symmetry_axis(c["mask"], c["bgr"])
        c["detail"] = detail_score(c["bgr"], c["mask"])
        c["facing"] = facing_guess(c["bgr"], c["mask"])
        c["aspect"] = c["h_px"] / float(c["w_px"])
    guessed = None
    if views:
        names = [canonical_view(v) for v in views]
        if len(names) != len(cuts):
            raise SystemExit("--views names %d view(s) but %d were found in the image(s) (in reading order). Name every "
                             "one; use 'skip' for views to ignore." % (len(names), len(cuts)))
    else:
        names, guessed = guess_names(cuts)
    seen = set()
    for i, n in enumerate(names):
        if n is not None and n in seen:
            raise SystemExit("view %r is named twice" % n)
        seen.add(n)
    used = [(n, c) for n, c in zip(names, cuts) if n is not None]
    upright = [(n, c) for n, c in used if n in ("front", "back", "left", "right")]
    same_scale = False
    if upright:
        hs = [c["h_px"] for _, c in upright]
        spread = (max(hs) - min(hs)) / float(max(hs))
        if scale == "same" or (scale == "auto" and all(c["single_sheet"] for _, c in upright) and spread <= 0.15):
            same_scale = True
        if scale == "auto" and not same_scale and len(upright) > 1 and all(c["single_sheet"] for _, c in upright):
            notes.append("views differ in height by %.0f %%: each was scaled to the same height separately "
                         "(pass --scale same if the sheet really is drawn at one scale)" % (spread * 100))
        ref_h = max(hs)
    doc = dict(version=1, height=float(height), views={}, order=[], notes=notes,
               sources=[os.path.abspath(p) for p in images], background=backgrounds[0] if backgrounds else None)
    recs = {}
    for n, c in used:
        rec = _view_record(n, None, c["mask"], c["box"], c["source"])
        x0, y0, x1, y1 = c["bb"]
        if n in ("front", "back", "left", "right"):
            ppu = (ref_h if same_scale else c["h_px"]) / float(height)
            oy = y1 + 1.0                                         # ground line: the lowest silhouette row
            if n in ("front", "back"):
                ox = c["sym_x"] if c["sym"] >= 0.8 else footprint_center(c["mask"])
            else:
                ox = footprint_center(c["mask"])
        else:
            ppu = None
            ox, oy = c["sym_x"] if c["sym"] >= 0.8 else (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
        rec.update(px_per_unit=ppu, origin_px=[float(ox), float(oy)], symmetry=round(c["sym"], 3),
                   detail=round(c["detail"], 2), size_px=[int(c["mask"].shape[1]), int(c["mask"].shape[0])],
                   source_scale=round(factors.get(c["source"], 1.0), 5))
        recs[n] = (rec, c)
    # plan views take their scale and depth origin from the upright views
    widths = [(c["w_px"] / r["px_per_unit"]) for n, (r, c) in recs.items() if n in ("front", "back")]
    for n in ("top", "bottom"):
        if n not in recs:
            continue
        rec, c = recs[n]
        if widths:
            rec["px_per_unit"] = c["w_px"] / (sum(widths) / len(widths))
        else:
            rec["px_per_unit"] = max(c["w_px"], c["h_px"]) / float(height)
            notes.append("%s view has no front/back view to take its scale from: its longest side was set to the "
                         "height" % n)
        side = next((recs[s] for s in ("right", "left") if s in recs), None)
        if side is not None:
            srec, sc = side
            sx0, _, sx1, _ = sc["bb"]
            yc = ((sx0 + sx1 + 1) / 2.0 - srec["origin_px"][0]) / srec["px_per_unit"] * srec["u_vec"][1]
            sign = rec["v_vec"][1]
            rec["origin_px"][1] = (c["bb"][1] + c["bb"][3] + 1) / 2.0 + sign * yc * rec["px_per_unit"]
    doc["guessed"] = guessed
    if len(images) == 1:
        im0 = read_image(images[0])
        f0 = factors.get(os.path.abspath(images[0]), 1.0)
        doc["sheet_size"] = [int(round(im0.shape[1] * f0)), int(round(im0.shape[0] * f0))]
    return _write_pack(doc, recs, out_dir, edge, regions, notes)


def find_features(bgr, mask, rec):
    """Drawn faces and eyes of one view, as boxes in world units [u0, v0, u1, v1]. The painter needs them (a
    face is painted by one picture only) and so does the check that counts eyes on the finished model.
    -> (faces, eyes), or (None, None) when the detectors are not available."""
    from . import vision
    pic = bgr.copy()
    pic[mask == 0] = 255
    ppu, (ox, oy) = rec["px_per_unit"], rec["origin_px"]
    out = []
    for kind in ("face", "eye"):
        found = vision.detect(pic, kind)
        if found is None:
            return None, None
        out.append([[round((x0 - ox) / ppu, 4), round((oy - y1) / ppu, 4), round((x1 - ox) / ppu, 4),
                     round((oy - y0) / ppu, 4)] for x0, y0, x1, y1, _s in found])
    return out[0], out[1]


def _write_pack(doc, recs, out_dir, edge, regions, notes):
    """Write every view's files (cutout, mask, texture, flat twin, grid, marks) and ref.json.
    recs: {view: (record with px_per_unit and origin_px, cut with bgr / mask / bb / bg / h_px / w_px)}."""
    pal_pts, pal_w = [], []
    for n, (rec, c) in recs.items():
        ppu, (ox, oy) = rec["px_per_unit"], rec["origin_px"]
        h, w = c["mask"].shape
        x0, y0, x1, y1 = c["bb"]
        rec["frame"] = [(0 - ox) / ppu, (oy - h) / ppu, (w - ox) / ppu, oy / ppu]
        rec["bbox"] = [(x0 - ox) / ppu, (oy - (y1 + 1)) / ppu, (x1 + 1 - ox) / ppu, (oy - y0) / ppu]
        e = edge if edge is not None else max(1, int(round(0.009 * max(c["h_px"], c["w_px"]))))
        alpha = soft_alpha(c["bgr"], c["mask"], c["bg"])
        cut = np.dstack([c["bgr"], alpha])
        cut[alpha == 0, :3] = 0
        tex = bleed(c["bgr"], c["mask"], e)
        label, regs = (None, [])
        if regions:
            # colour regions are found on a picture of ordinary size (they are areas, not outlines) and laid
            # back over the full one
            rz = min(1.0, 640.0 / max(h, w))
            if rz < 1.0:
                small = cv2.resize(c["bgr"], None, fx=rz, fy=rz, interpolation=cv2.INTER_AREA)
                smask = cv2.resize(c["mask"], (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                label, regs = regions_of(small, smask, max_regions=regions)
                label = cv2.resize(label.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                label[c["mask"] == 0] = 0
                for r in regs:
                    r["box_px"] = [int(r["box_px"][0] / rz), int(r["box_px"][1] / rz),
                                   min(w - 1, int((r["box_px"][2] + 1) / rz)), min(h - 1, int((r["box_px"][3] + 1) / rz))]
                    r["pin_px"] = [min(w - 1, int((r["pin_px"][0] + 0.5) / rz)), min(h - 1, int((r["pin_px"][1] + 0.5) / rz))]
            else:
                label, regs = regions_of(c["bgr"], c["mask"], max_regions=regions)
        if rec["name"] in ("front", "back", "left", "right"):
            faces, eyes = find_features(c["bgr"], c["mask"], rec)
            if faces is not None:
                rec["faces"], rec["eyes"] = faces, eyes
        for r in regs:
            bx0, by0, bx1, by1 = r.pop("box_px")
            px, py = r.pop("pin_px")
            r["box"] = [round((bx0 - ox) / ppu, 4), round((oy - (by1 + 1)) / ppu, 4), round((bx1 + 1 - ox) / ppu, 4),
                        round((oy - by0) / ppu, 4)]
            r["pin"] = [round((px + 0.5 - ox) / ppu, 4), round((oy - py - 0.5) / ppu, 4)]
            r["pin_px"] = [px, py]
            del r["area"]
        rec["image"], rec["mask"], rec["texture"] = n + ".png", n + "_mask.png", n + "_tex.png"
        rec["flat"] = n + "_flat.png"
        write_image(os.path.join(out_dir, rec["flat"]), cel_flat(tex, 0.0045 * max(c["h_px"], c["w_px"]))[0])
        rec["flat_kind"] = "cel"
        rec["grid"] = n + "_grid.png"
        write_image(os.path.join(out_dir, rec["image"]), cut)
        write_image(os.path.join(out_dir, rec["mask"]), c["mask"])
        write_image(os.path.join(out_dir, rec["texture"]), tex)
        write_image(os.path.join(out_dir, rec["grid"]), draw_grid(rec, cut, n.upper()))
        if regs:
            rec["marks"], rec["region_map"] = n + "_marks.png", n + "_regions.png"
            write_image(os.path.join(out_dir, rec["marks"]), draw_marks(rec, cut, label, regs, n.upper()))
            write_image(os.path.join(out_dir, rec["region_map"]), label.astype(np.uint8))
        rec["regions"] = [{k: v for k, v in r.items() if k != "pin_px"} for r in regs]
        rec["profile"] = [[round(rec["bbox"][1] + zf * (rec["bbox"][3] - rec["bbox"][1]), 4),
                           [[round((a - ox) / ppu, 4), round((b + 1 - ox) / ppu, 4)] for a, b in runs]]
                          for zf, runs in row_runs(c["mask"][y0:y1 + 1], 100)]
        fg = c["mask"] > 0
        pal_pts.append(c["bgr"][fg])
        doc["views"][n] = rec
        doc["order"].append(n)
    allpts = np.concatenate(pal_pts)
    pal, _l, _c = palette_of(allpts.reshape(-1, 1, 3), np.full((len(allpts), 1), 255, np.uint8), k=12)
    doc["palette"] = [[hx, round(share, 4)] for hx, share in pal]
    size = {}
    fb = [doc["views"][n] for n in ("front", "back") if n in doc["views"]]
    sd = [doc["views"][n] for n in ("right", "left") if n in doc["views"]]
    if fb:
        size["width"] = round(max(v["bbox"][2] - v["bbox"][0] for v in fb), 4)
    if sd:
        size["depth"] = round(max(v["bbox"][2] - v["bbox"][0] for v in sd), 4)
    up = fb + sd
    if up:
        size["height"] = round(max(v["bbox"][3] for v in up), 4)
    doc["size"] = size
    if len(fb) == 2:
        wa, wb = (v["bbox"][2] - v["bbox"][0] for v in fb)
        if abs(wa - wb) / max(wa, wb) > 0.08:
            notes.append("front and back widths differ by %.0f %% (%.3f vs %.3f): the drawings are not perfectly "
                         "consistent; trust the front view for width" % (abs(wa - wb) / max(wa, wb) * 100, wa, wb))
    tiles = []
    for n in doc["order"]:
        rec = doc["views"][n]
        g = cv2.imread(os.path.join(out_dir, rec["grid"]))
        tiles.append(g)
    th = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * th / t.shape[0]), th), interpolation=cv2.INTER_AREA) for t in tiles]
    sheet = hstack(tiles)
    if sheet.shape[1] > 2000:
        f = 2000.0 / sheet.shape[1]
        sheet = cv2.resize(sheet, (2000, int(sheet.shape[0] * f)), interpolation=cv2.INTER_AREA)
    doc["overview"] = "overview.png"
    write_image(os.path.join(out_dir, "overview.png"), sheet)
    sw = np.zeros((60, 60 * len(doc["palette"]), 3), np.uint8)
    for i, (hx, share) in enumerate(doc["palette"]):
        s = hx[1:]
        sw[:, i * 60:(i + 1) * 60] = (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))
    write_image(os.path.join(out_dir, "palette.png"), sw)
    doc["ink"] = pack_ink(doc, out_dir)
    with open(os.path.join(out_dir, "ref.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def describe(doc, out_dir):
    """Human / agent readable summary of a pack."""
    L = []
    s = doc.get("size", {})
    L.append("REFERENCE PACK  %s" % out_dir)
    L.append("  subject height = %g (world units, Z up, stands on z=0, faces -Y, x=0 is the symmetry plane)" % doc["height"])
    if s:
        L.append("  measured size: " + ", ".join("%s %s" % (k, _fmt(v)) for k, v in s.items()))
    if doc.get("guessed"):
        L.append("  VIEW NAMES WERE GUESSED (%s). Open overview.png and check them; if wrong re-run with "
                 "--views name,name,... in reading order." % doc["guessed"])
    for n in doc["order"]:
        v = doc["views"][n]
        b = v["bbox"]
        L.append("  [%s]  right=%s up=%s  silhouette %s..%s x %s..%s  (%s wide, %s tall)  symmetry %.2f" % (
            n, v["u_axis"], v["v_axis"], _fmt(b[0]), _fmt(b[2]), _fmt(b[1]), _fmt(b[3]), _fmt(b[2] - b[0]),
            _fmt(b[3] - b[1]), v["symmetry"]))
        L.append("      files: %s (cutout)  %s (measuring grid)  %s" % (
            v["image"], v["grid"], (v.get("marks", "") + " (numbered regions)") if v.get("marks") else ""))
    L.append("  palette: " + "  ".join("%s %.0f%%" % (hx, share * 100) for hx, share in doc["palette"][:12]))
    ink = doc.get("ink")
    if ink:
        L.append("  line art: the subject is drawn with ink outlines (%s, %.4f wide). Finish the model with k.outline() "
                 "so every separate piece gets that line." % (ink["color"], ink["width"]))
    else:
        L.append("  line art: none (no ink outlines in the pictures; k.outline() is not needed)")
    for n in doc.get("notes", []):
        L.append("  NOTE: " + n)
    return "\n".join(L)


def region_table(doc, view, limit=40):
    v = doc["views"][view]
    L = ["regions of [%s]  (id: colour, share of silhouette, box u0..u1 x v0..v1)" % view]
    for r in v.get("regions", [])[:limit]:
        b = r["box"]
        L.append("  %2d: %s %4.1f%%  %s..%s x %s..%s" % (r["id"], r["color"], r["share"] * 100, _fmt(b[0]), _fmt(b[2]),
                                                      _fmt(b[1]), _fmt(b[3])))
    return "\n".join(L)
