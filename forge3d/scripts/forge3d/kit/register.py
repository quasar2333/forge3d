"""Geometry <-> picture registration, one reference view at a time.

A model seen from a reference camera falls into surface patches (a fringe standing off a face, a sleeve in front
of a skirt); the picture draws the same parts, but hardly ever in exactly the same place, so the drawn fringe
lands partly on the face and the drawn face partly on the fringe. This module finds the patches and where the
picture draws each of them; `paint_from_ref` uses that to tidy thin strips of wrong colour along part edges and
to know each part's base colour (what hidden surfaces of that part are filled with).
"""
import numpy as np

from . import raster

EMPTY = -1e30


def view_scale(view, target=1600.0):
    """Working resolution relative to the picture: small pictures are enlarged, huge ones reduced."""
    return float(np.clip(target / max(view.size_px), 0.5, 3.0))


def view_buffer(tris, view, scale, flip=False):
    """Depth buffer of triangles (N, 3, 3) on the picture's own pixel grid enlarged by `scale`.
    -> (D, to_px, px_per_unit); D holds the coordinate toward the camera, EMPTY-ish where nothing is."""
    ppu = view.px_per_unit * scale
    geo = tris * np.array([-1.0, 1.0, 1.0], np.float32) if flip else tris
    D, to_px = raster.depth_buffer(geo, view.toward_camera, view.u_vec, view.v_vec, view.frame, ppu)
    w, h = int(round(view.size_px[0] * scale)), int(round(view.size_px[1] * scale))
    if D.shape != (h, w):                                   # rounding of the frame: keep the picture's grid
        out = np.full((h, w), -np.inf, np.float32)
        hh, ww = min(h, D.shape[0]), min(w, D.shape[1])
        out[:hh, :ww] = D[:hh, :ww]
        D = out
    return D, to_px, ppu


def geo_patches(D, ppu, height=1.0, calm=1.2, core=0.006):
    """Surface patches of a depth buffer: areas that face the camera, separated where the surface steps or turns
    steeply away (occlusion edges, the walls of raised parts).
    -> (labels int32, 0 = outside; count; steepness = tan of the angle away from head-on)."""
    import cv2
    from scipy import ndimage
    from skimage.segmentation import watershed
    inside = D > EMPTY
    if not inside.any():
        return np.zeros(D.shape, np.int32), 0, np.zeros(D.shape, np.float32)
    Z = cv2.GaussianBlur(np.where(inside, D, D[inside].min()), (0, 0), 1.0)
    gx = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * ppu
    gy = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * ppu
    steep = np.minimum(np.sqrt(gx * gx + gy * gy), 12.0)    # tan of the angle away from head-on
    steep[~inside] = 12.0
    r = max(1, int(round(core * height * ppu)))
    seeds = cv2.erode(((steep < calm) & inside).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    mk, n = ndimage.label(seeds)
    if n == 0:
        return inside.astype(np.int32), 1, steep
    area = np.bincount(mk.ravel(), minlength=n + 1)
    small = np.flatnonzero(area < (2.0 * r + 1) ** 2)
    if len(small) and len(small) < n:                       # specks do not start a patch of their own
        mk[np.isin(mk, small)] = 0
        mk, n = ndimage.label(mk > 0)
    lab = watershed(steep, mk, mask=inside).astype(np.int32)
    return lab, n, steep


def to_lab(bgr):
    import cv2
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100.0 / 255.0, 1.0, 1.0], np.float32)


def _modes(pix, k=6, min_share=0.04):
    """Dominant colours of a set of Lab pixels (a patch's own colours)."""
    import cv2
    if len(pix) > 4000:
        pix = pix[np.random.default_rng(0).choice(len(pix), 4000, replace=False)]
    k = min(k, max(1, len(pix) // 20))
    if k <= 1:
        return np.median(pix, 0)[None]
    cv2.setRNGSeed(1)
    _c, lb, cen = cv2.kmeans(np.ascontiguousarray(pix, np.float32), k, None,
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lb.ravel(), minlength=k) / float(len(pix))
    keep = share >= min_share
    return cen[keep] if keep.any() else cen[[int(share.argmax())]]


def drawn_layout(lab, flat_lab, ppu, height=1.0, delta=0.02, reach=0.04, stay=6.0, match=12.0, alien=26.0):
    """Where the picture draws each geometric patch. Every patch learns its own colours from its interior; a pixel
    within `reach` of a patch border then goes to the neighbouring patch when it clearly is one of that patch's
    colours (within `match` dE) and a worse fit for its own patch (by more than `stay`). Colours that belong to
    neither stay put: an eye under a fringe is not hair just because it is blue.
    -> (label image like `lab`, same ids, differing from it exactly where picture and geometry disagree;
        mask of border-zone pixels whose colour is foreign to the patch they lie on, further than `alien` dE)."""
    import cv2
    from scipy import ndimage
    n = int(lab.max())
    if n < 2:
        return lab.copy(), np.zeros(lab.shape, bool)
    h, w = lab.shape
    d_px = max(2, int(round(delta * height * ppu)))
    r_px = max(3, int(round(reach * height * ppu)))
    objs = ndimage.find_objects(lab)
    dist = np.zeros((h, w), np.float32)
    models = {}
    for g in range(1, n + 1):
        sl = objs[g - 1]
        if sl is None:
            continue
        m = lab[sl] == g
        dt = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
        dist[sl][m] = dt[m]
        inner = m & (dt >= min(d_px, 0.5 * float(dt.max())))
        if inner.sum() >= 12:
            models[g] = _modes(flat_lab[sl][inner])
    band = (dist < r_px) & (lab > 0)
    best = np.full((h, w), 1e9, np.float32)
    own = np.zeros((h, w), np.float32)
    out = lab.copy()
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
    soft = max(1.0, 0.004 * height * ppu)
    for g, modes in models.items():
        sl = objs[g - 1]
        y0, y1 = max(0, sl[0].start - r_px), min(h, sl[0].stop + r_px)
        x0, x1 = max(0, sl[1].start - r_px), min(w, sl[1].stop + r_px)
        m = (lab[y0:y1, x0:x1] == g).astype(np.uint8)
        near = (cv2.dilate(m, ker) > 0) & band[y0:y1, x0:x1]
        if not near.any():
            continue
        px = flat_lab[y0:y1, x0:x1]
        c = np.full(m.shape, 1e9, np.float32)
        for cen in modes:
            c = np.minimum(c, np.sqrt(((px - cen) ** 2).sum(2)))
        c = cv2.GaussianBlur(c, (0, 0), soft)
        own[y0:y1, x0:x1][m > 0] = c[m > 0]
        c = np.where(m > 0, c - stay, np.where(c <= match, c, 1e9))
        c = np.where(near, c, 1e9)
        b, o = best[y0:y1, x0:x1], out[y0:y1, x0:x1]
        better = c < b
        o[better] = g
        b[better] = c[better]
    keep = ~band | (best > 1e8)
    out[keep] = lab[keep]
    # a part drawn out of place is still one piece with the part itself: pixels claimed for a patch must hang
    # together with that patch. A blue print in the middle of an apron is not the neighbouring blue hair.
    for g in np.unique(out[out != lab]):
        sl = objs[g - 1]
        y0, y1 = max(0, sl[0].start - r_px), min(h, sl[0].stop + r_px)
        x0, x1 = max(0, sl[1].start - r_px), min(w, sl[1].stop + r_px)
        o, l = out[y0:y1, x0:x1], lab[y0:y1, x0:x1]
        cl, cnt = ndimage.label(o == g)
        home = np.zeros(cnt + 1, bool)
        home[np.unique(cl[l == g])] = True
        stray = (cl > 0) & ~home[cl]
        o[stray] = l[stray]
    # colours foreign to their patch count only where they reach the patch's edge (something from next door)
    foreign = band & (own > alien)
    if foreign.any():
        edge = ndimage.maximum_filter(lab, 5) != ndimage.minimum_filter(lab, 5)
        cl, cnt = ndimage.label(foreign)
        touch = np.zeros(cnt + 1, bool)
        touch[np.unique(cl[edge & foreign])] = True
        foreign &= touch[cl]
    return out, foreign


def clean_picture(lab, pic, tex, flat, line=None, ppu=1.0, height=1.0, thick=0.007):
    """The picture with the edges of its parts tidied onto the edges of the geometry.

    Where the picture draws a different part than the geometry has (`pic` != `lab`) in a thin strip along a patch
    border, the two are merely out of register: the strip takes the flat colour of the nearest pixel that does
    belong to the patch. Wider disagreements (more than about 2 x `thick` x height across) are a difference in
    design, not in register; there the picture is left as drawn. -> (tex, flat, share of pixels replaced)."""
    import cv2
    from scipy import ndimage
    bad = (pic != lab) & (lab > 0)
    if bad.any():
        r = max(1, int(round(thick * height * ppu)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        core = cv2.erode(bad.astype(np.uint8), ker)
        if core.any():                                      # whole connected areas that have a wide part stay
            cl, cnt = ndimage.label(bad)
            wide = np.zeros(cnt + 1, bool)
            wide[np.unique(cl[core > 0])] = True
            bad &= ~wide[cl]
    if not bad.any():
        return tex, flat, 0.0
    if line is not None:                                    # the ink outline of a displaced part goes with it
        bad |= line & (cv2.dilate(bad.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & (lab > 0)
    tex, flat = tex.copy(), flat.copy()
    ids = np.unique(lab[bad])
    objs = ndimage.find_objects(lab)
    for g in ids:
        sl = objs[g - 1]
        m = lab[sl] == g
        b = bad[sl] & m
        good = m & ~b
        if not good.any():
            continue                                        # nothing of its own to continue from: left as drawn
        idx = ndimage.distance_transform_edt(~good, return_distances=False, return_indices=True)
        src = flat[sl][idx[0][b], idx[1][b]]
        tex[sl][b] = src
        flat[sl][b] = src
    return tex, flat, float(bad.sum()) / float(max(1, (lab > 0).sum()))


def base_picture(subject, flat, ppu, height=1.0, k=10, thin=0.012):
    """The flat picture without its small details: colour areas narrower than about 2 x `thin` x height (lashes,
    highlights, trim, small prints, the rims of areas) take the colour of the nearest broad area; broad areas
    keep their own colour. This is what a part looks like where no picture shows it directly: hidden surfaces
    are filled from it, so skin under a fringe continues as skin and not as a smear of lashes.
    subject: bool mask of the pixels that matter."""
    import cv2
    from scipy import ndimage
    sub = np.asarray(subject, bool)
    if not sub.any():
        return flat.copy()
    pix = to_lab(flat)[sub]
    sample = pix if len(pix) <= 30000 else pix[np.random.default_rng(0).choice(len(pix), 30000, replace=False)]
    k = int(min(k, max(1, len(sample) // 50)))
    cv2.setRNGSeed(1)
    _c, _l, cen = cv2.kmeans(np.ascontiguousarray(sample, np.float32), k, None,
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    idx = np.full(sub.shape, -1, np.int32)
    idx[sub] = ((pix[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
    r = max(1, int(round(thin * height * ppu)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    broad = np.zeros(sub.shape, bool)
    for c in range(k):
        broad |= cv2.morphologyEx((idx == c).astype(np.uint8), cv2.MORPH_OPEN, ker) > 0
    if not broad.any():
        return flat.copy()
    near = ndimage.distance_transform_edt(~broad, return_distances=False, return_indices=True)
    out = flat.copy()
    todo = sub & ~broad
    out[todo] = flat[near[0][todo], near[1][todo]]
    return out
