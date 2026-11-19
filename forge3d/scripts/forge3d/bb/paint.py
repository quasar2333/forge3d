"""Pixel textures for Blockbench models: straight from the reference pictures, or procedural polish."""
import numpy as np

from ..kit import raster, refkit
from .model import FACE_AXES, FACES

FACE_NORMALS = {f: np.array(FACE_AXES[f][0], float) for f in FACES}


def _sources(R, views, mirror):
    names = [n for n in R.views if views is None or n in views]
    out = [dict(key=n, view=R[n], flip=False, d=R[n].toward_camera) for n in names]
    if mirror:
        for have, want in (("right", "left"), ("left", "right")):
            if have in names and want not in R.views:
                out.append(dict(key=want + "_mirror", view=R[have], flip=True,
                                d=R[have].toward_camera * np.array([-1.0, 1.0, 1.0])))
    return out


def quantize(colors, n, seed=1):
    """Reduce (N, 3) uint8 colours to n representative ones (k-means in Lab). -> remapped (N, 3) uint8."""
    import cv2
    if n is None or n <= 0 or len(colors) == 0:
        return colors
    uniq = np.unique(colors, axis=0)
    if len(uniq) <= n:
        return colors
    lab = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.3)
    _c, labels, centres = cv2.kmeans(lab, int(n), None, crit, 4, cv2.KMEANS_PP_CENTERS)
    out = np.empty_like(colors)
    labels = labels.ravel()
    for i in range(int(n)):
        sel = labels == i
        if sel.any():
            # the member closest to the centre keeps colours that really occur in the artwork (no muddy averages)
            d = ((lab[sel] - centres[i]) ** 2).sum(1)
            out[sel] = colors[sel][int(d.argmin())]
    return out


def paint_from_ref(model, ref, views=None, mirror=True, colors=48, supersample=3, bias=None, keep=False,
                   cutout="thin", verbose=True):
    """Paint the faces of every cube from the reference pictures.

    Each texel takes the colour the best-placed reference view shows at that spot (a view must face the texel and
    see it, i.e. no other cube stands in between). Texels no view can see copy their nearest painted neighbour on
    the same cube, so tops and undersides continue the sides.
    colors: reduce the whole texture to this many colours for a pixel-art look (None = keep every shade).
    supersample: reference samples per texel side (the texel shows their average).
    bias: {view: priority}, default front 1.5.   mirror: use the one side view for both sides.
    keep: leave faces you already painted by hand untouched.
    cutout: texels that fall outside the reference silhouette become transparent, so a flat plate takes the exact
            outline of the artwork (hair curtains, capes, wings). 'thin' (default) does this only for plates up to
            1.5 px thick; True for every cube; False never. A cube created with cutout=<RefMask> is always cut to
            that mask instead, whatever stands in front of it (a fin, bow or tail drawn over other parts).
    """
    from scipy.spatial import cKDTree
    R = refkit.as_ref(ref)
    sources = _sources(R, views, mirror)
    if not sources:
        raise ValueError("paint_from_ref: the reference pack has none of the requested views")
    if not model.cubes:
        raise ValueError("paint_from_ref: the model has no cubes")
    s = R.height / model.height                               # world units per model pixel
    tris = model.triangles() * s
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    diag = float(np.linalg.norm(hi - lo))
    flipv = np.array([-1.0, 1.0, 1.0], np.float32)
    pri = {"front": 1.5, "left_mirror": 0.8, "right_mirror": 0.8}
    pri.update(bias or {})
    for src in sources:
        v = src["view"]
        geo = tris * flipv if src["flip"] else tris
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cu, cv_ = (corners * (flipv if src["flip"] else 1.0)) @ v.u_vec, corners @ v.v_vec
        fr = (float(cu.min()) - 0.02 * diag, float(cv_.min()) - 0.02 * diag, float(cu.max()) + 0.02 * diag,
              float(cv_.max()) + 0.02 * diag)
        ppu = max(256.0, min(1024.0, 4.0 * model.height * model.density)) / max(fr[2] - fr[0], fr[3] - fr[1])
        src["depth"], src["to_px"] = raster.depth_buffer(geo, v.toward_camera, v.u_vec, v.v_vec, fr, ppu)
        src["px"] = 1.0 / ppu
    K = max(1, int(supersample))
    sub = (np.arange(K) + 0.5) / K
    jobs, seen_pos, seen_col = [], [], []
    masks = {id(src): src["view"].mask().array for src in sources}
    cuts = {}                                   # cube id -> (source, mask array, explicit)
    for c in model.cubes:
        sz = c.size + 2 * c.inflate
        ax = int(np.argmin(sz))
        Rc, _t = c.matrix()
        if c.cut_mask is not None:
            vname = c.cut_mask.view.name
            cand = [x for x in sources if x["view"].name == vname and x["flip"] == bool(c.cut_flip)]
            if not cand and c.cut_flip:           # mirrored plate but no mirrored source: project through the mirror
                base = next((x for x in sources if x["view"].name == vname and not x["flip"]), None)
                if base is not None:
                    cand = [dict(base, flip=True, key=base["key"] + "_flip", d=base["d"] * np.array([-1.0, 1.0, 1.0]))]
                    cand[0]["depth"] = None
            if not cand:
                raise ValueError("cube %r: its cutout mask comes from the %r view, which is not among the views being "
                                 "painted" % (c.name, vname))
            cuts[id(c)] = (cand[0], c.cut_mask.array, True)
        elif cutout is True or (cutout == "thin" and sz[ax] <= 1.5):
            axis_w = Rc[:, ax]
            src = max(sources, key=lambda x: abs(float(axis_w @ x["d"])))
            if abs(float(axis_w @ src["d"])) >= 0.6:
                cuts[id(c)] = (src, masks[id(src)], False)
    for c in model.cubes:
        cube_seen = []
        for f in FACES:
            face = c.faces[f]
            if face.hidden or (keep and face.painted):
                continue
            o, du, dv, n = c.face_frame(f)
            w, h = face.size
            ii, jj = np.meshgrid(np.arange(w), np.arange(h))
            best = np.full((h, w), -1.0, np.float32)
            col = np.zeros((h, w, 3), np.float32)
            cut = cuts.get(id(c))
            centre = (o + np.outer(ii.ravel() + 0.5, du) + np.outer(jj.ravel() + 0.5, dv)) * s      # (h*w, 3)
            texel = float(np.linalg.norm(du) + np.linalg.norm(dv)) * s * 0.5
            for src in sources:
                v = src["view"]
                facing = float(n @ src["d"])
                if facing < 0.25:
                    continue
                own = cut is not None and cut[2] and cut[0]["view"] is v and cut[0]["flip"] == src["flip"]
                Pm = centre * flipv if src["flip"] else centre
                ix, iy, z, inb = src["to_px"](Pm)
                D = src["depth"]
                ixc, iyc = np.clip(ix, 0, D.shape[1] - 1), np.clip(iy, 0, D.shape[0] - 1)
                slope = np.sqrt(max(0.0, 1.0 - facing * facing)) / facing
                vis = inb & (z >= D[iyc, ixc] - (0.004 * diag + (1.5 * src["px"] + texel) * slope))
                if own and facing > 0.6:
                    vis = inb                     # a plate that carries its own piece of the picture shows all of it
                u, w_ = Pm @ v.u_vec, Pm @ v.v_vec
                fr = v.frame
                vis &= (u >= fr[0]) & (u <= fr[2]) & (w_ >= fr[1]) & (w_ <= fr[3])
                score = np.where(vis, facing * facing * pri.get(src["key"], 1.0), -1.0).reshape(h, w)
                take = score > best
                if not take.any():
                    continue
                if own and facing > 0.6:
                    score = np.where(vis, 10.0, -1.0).reshape(h, w)
                    take = score > best
                acc = np.zeros((h * w, 3), np.float32)
                tex = v.texture
                th, tw = tex.shape[:2]
                for a in sub:
                    for b in sub:
                        P = (o + np.outer(ii.ravel() + a, du) + np.outer(jj.ravel() + b, dv)) * s
                        if src["flip"]:
                            P = P * flipv
                        x = np.clip(np.floor(v.x_of(P @ v.u_vec)).astype(int), 0, tw - 1)
                        y = np.clip(np.floor(v.y_of(P @ v.v_vec)).astype(int), 0, th - 1)
                        acc += tex[y, x][:, ::-1].astype(np.float32) ** 2.2           # average in ~linear light
                acc = (acc / (K * K)) ** (1 / 2.2)
                col[take] = acc.reshape(h, w, 3)[take]
                best[take] = score[take]
            seen = best > 0
            face._alpha = None
            if cut is not None:
                # every texel of a cut cube, on any of its faces, is kept or dropped by where it lands in the
                # cutting picture, so both sides and the rim of a plate agree
                csrc, cmask, _explicit = cut
                cv_ = csrc["view"]
                mh, mw = cmask.shape
                cov = np.zeros(h * w, np.float32)
                for a in sub:
                    for b in sub:
                        P = (o + np.outer(ii.ravel() + a, du) + np.outer(jj.ravel() + b, dv)) * s
                        if csrc["flip"]:
                            P = P * flipv
                        xf = np.floor(cv_.x_of(P @ cv_.u_vec)).astype(int)
                        yf = np.floor(cv_.y_of(P @ cv_.v_vec)).astype(int)
                        okp = (xf >= 0) & (xf < mw) & (yf >= 0) & (yf < mh)
                        cov += okp & cmask[np.clip(yf, 0, mh - 1), np.clip(xf, 0, mw - 1)]
                face._alpha = np.where((cov / (K * K)).reshape(h, w) < 0.5, 0, 255).astype(np.uint8)
            jobs.append((c, face, centre.reshape(h, w, 3), seen, col))
            if seen.any():
                cube_seen.append((centre.reshape(h, w, 3)[seen], col[seen]))
        if cube_seen:
            c._seen = (np.concatenate([a for a, _ in cube_seen]), np.concatenate([b for _, b in cube_seen]))
            seen_pos.append(c._seen[0])
            seen_col.append(c._seen[1])
        else:
            c._seen = None
    if not seen_pos:
        raise ValueError("paint_from_ref: no face of the model is visible in any reference view. Is the model at the "
                         "right scale? It must stand on z=0 around x=y=0 and be about %g px tall." % model.height)
    all_tree = cKDTree(np.concatenate(seen_pos))
    all_col = np.concatenate(seen_col)
    trees = {}
    total = unseen = 0
    for c, face, centre, seen, col in jobs:
        miss = ~seen
        total += seen.size
        unseen += int(miss.sum())
        if miss.any():
            if c.color is not None and c._seen is None:
                col[miss] = np.array(_rgb(c.color), np.float32)
            else:
                if c._seen is not None:
                    if id(c) not in trees:
                        trees[id(c)] = cKDTree(c._seen[0])
                    _d, idx = trees[id(c)].query(centre[miss])
                    col[miss] = c._seen[1][idx]
                else:
                    _d, idx = all_tree.query(centre[miss])
                    col[miss] = all_col[idx]
        face.px[:, :, :3] = np.clip(np.round(col), 0, 255).astype(np.uint8)
        face.px[:, :, 3] = 255 if face._alpha is None else face._alpha
        face.painted = True
    if colors:
        faces = [f for _c, f, *_ in jobs]
        flat = np.concatenate([f.px[:, :, :3].reshape(-1, 3) for f in faces])
        q = quantize(flat, colors)
        k = 0
        for f in faces:
            n = f.px.shape[0] * f.px.shape[1]
            f.px[:, :, :3] = q[k:k + n].reshape(f.px.shape[0], f.px.shape[1], 3)
            k += n
    if verbose:
        print("[f3d] bb paint_from_ref: %d cubes painted from %s; %.0f%% of texels seen by a reference view" % (
            len(model.cubes), "/".join(src["key"] for src in sources), 100.0 * (total - unseen) / max(1, total)))
    return model


def _rgb(c):
    from .model import parse_color
    return parse_color(c)[:3]


def shade(model, strength=0.10, noise=0.03, seed=3):
    """Pixel-art polish for flat-coloured faces: a soft top-to-bottom falloff, darker undersides, and a little
    per-texel variation so large faces do not look like plastic. Faces painted from a reference are left alone."""
    rng = np.random.default_rng(seed)
    for c in model.cubes:
        for name, f in c.faces.items():
            if getattr(f, "from_ref", False):
                continue
            h, w = f.px.shape[:2]
            rgb = f.px[:, :, :3].astype(np.float32)
            k = np.ones((h, w), np.float32)
            if name in ("front", "back", "left", "right") and h > 1:
                k *= (1 + strength * (0.5 - np.linspace(0, 1, h)))[:, None]
            if name == "bottom":
                k *= 1 - 2 * strength
            if name == "top":
                k *= 1 + strength
            k *= 1 + noise * rng.standard_normal((h, w)).astype(np.float32)
            f.px[:, :, :3] = np.clip(np.round(rgb * k[:, :, None]), 0, 255).astype(np.uint8)
    return model
