"""Model vs reference pack: same camera, same frame, numbers plus a picture of where they differ."""
import json
import math
import os
import tempfile

import bpy
import cv2
import numpy as np
from mathutils import Vector

from . import refs as refmod
from .kit import core, staging
from .kit.raster import best_shift, scene_tris, silhouette
from .kit.raster import iou as _iou

PANEL_BG = (150, 146, 142)


def load_pack(ref_dir):
    path = os.path.join(ref_dir, "ref.json")
    if not os.path.isfile(path):
        raise SystemExit("no reference pack at %s (make one with: f3d ref <image> -o %s)" % (ref_dir, ref_dir))
    with open(path) as f:
        doc = json.load(f)
    doc["dir"] = os.path.abspath(ref_dir)
    return doc


def view_camera(view, radius):
    """Orthographic camera that sees exactly the reference view's frame."""
    u0, v0, u1, v1 = view["frame"]
    uc, vc = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    centre = Vector(view["u_vec"]) * uc + Vector(view["v_vec"]) * vc
    d = staging.direction(view["az"], view["el"])
    data = bpy.data.cameras.new("f3d_ref_cam")
    data.type = "ORTHO"
    data.sensor_fit = "HORIZONTAL"
    data.ortho_scale = u1 - u0
    data.clip_start = 0.001
    data.clip_end = radius * 20 + 100
    cam = bpy.data.objects.new("f3d_ref_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    cam["f3d_helper"] = True
    m = staging.look_at_matrix(centre + d * (radius * 8 + 10), centre, 0.0, view["az"])
    cam.matrix_world = m
    return cam


def render_view(view, zoom, tmpdir, radius):
    w, h = int(round(view["size_px"][0] * zoom)), int(round(view["size_px"][1] * zoom))
    cam = view_camera(view, radius)
    path = os.path.join(tmpdir, "cmp_%s.png" % view["name"])
    staging.render_tile(cam, path, (w, h))
    staging.remove_camera(cam)
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError("render failed for view %s" % view["name"])
    if im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    return im


def _over(bgra, bg=PANEL_BG):
    a = bgra[:, :, 3:4].astype(np.float32) / 255.0
    return (bgra[:, :, :3].astype(np.float32) * a + np.array(bg, np.float32) * (1 - a)).astype(np.uint8)


def _runs_width(row):
    return float((row > 0).sum())


def _bbox(mask):
    ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if len(ys) == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) * np.array([100.0 / 255.0, 1.0, 1.0], np.float32)


def _hex(bgr):
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def analyse(view, ref_mask, ref_bgr, mod_mask, mod_bgr, regions, region_map):
    """Numbers for one view. Masks / images are the same size (reference crop scaled by zoom)."""
    h, w = ref_mask.shape
    s = view["px_per_unit"] * (w / float(view["size_px"][0]))
    f = view["frame"]
    r, m = ref_mask > 0, mod_mask > 0
    ra = float(r.sum())
    out = dict(view=view["name"], iou=_iou(r, m), missing=float((r & ~m).sum()) / max(ra, 1.0),
               extra=float((m & ~r).sum()) / max(ra, 1.0), empty=not m.any())
    rb, mb = _bbox(r), _bbox(m)
    edges = []
    if rb and mb:
        names = ("left", "top", "right", "bottom")
        for i, nm in enumerate(names):
            d = (mb[i] - rb[i]) / s
            if nm in ("top", "bottom"):
                d = -d                                      # image y is down, world v is up
            edges.append((nm, d))
    out["edges"] = edges
    bands = []
    if rb:
        y0, y1 = rb[1], rb[3]
        if mb:
            y0, y1 = min(y0, mb[1]), max(y1, mb[3])
        n = 12
        for i in range(n):
            a, b = y0 + (y1 - y0) * i // n, y0 + (y1 - y0) * (i + 1) // n
            if b <= a:
                continue
            wr = float(r[a:b].sum()) / (b - a) / s
            wm = float(m[a:b].sum()) / (b - a) / s
            v_hi, v_lo = f[3] - a / s, f[3] - b / s
            # centre of mass per band tells left/right drift apart from width
            cr = (np.nonzero(r[a:b])[1].mean() / s + f[0]) if r[a:b].any() else None
            cm = (np.nonzero(m[a:b])[1].mean() / s + f[0]) if m[a:b].any() else None
            bands.append(dict(v0=v_lo, v1=v_hi, ref=wr, model=wm, ref_c=cr, model_c=cm))
    out["bands"] = bands
    dx, dy, siou = best_shift(ref_mask, mod_mask)
    out["shift"] = (dx / s, -dy / s, siou)
    both = r & m
    if both.sum() > 50:
        k = max(3, int(round(0.012 * max(h, w))) | 1)
        la, lb = _lab(cv2.GaussianBlur(ref_bgr, (k, k), 0)), _lab(cv2.GaussianBlur(mod_bgr, (k, k), 0))
        de = np.sqrt(((la - lb) ** 2).sum(2))
        out["delta_e"] = float(de[both].mean())
        reg = []
        if region_map is not None:
            rm = cv2.resize(region_map, (w, h), interpolation=cv2.INTER_NEAREST)
            for rg in regions:
                sel = (rm == rg["id"]) & both
                if sel.sum() < 30:
                    cover = float(((rm == rg["id"]) & m).sum()) / max(1.0, float((rm == rg["id"]).sum()))
                    reg.append(dict(id=rg["id"], share=rg["share"], ref=rg["color"], model=None, de=None, cover=cover))
                    continue
                mc = mod_bgr[sel].mean(0)
                rc = ref_bgr[sel].mean(0)
                d = float(np.sqrt(((_lab(rc.reshape(1, 1, 3).astype(np.uint8)) - _lab(mc.reshape(1, 1, 3).astype(np.uint8))) ** 2).sum()))
                reg.append(dict(id=rg["id"], share=rg["share"], ref=_hex(rc), model=_hex(mc), de=d, cover=1.0))
        out["regions"] = reg
    else:
        out["delta_e"] = None
        out["regions"] = []
    return out


def _label(img, text, scale=0.55):
    refmod._put(img, text, (8, 20), scale, (120, 255, 255), 1)


def panels(view, ref_cut, mod_bgra, ref_mask, mod_mask, target=460):
    h, w = ref_mask.shape
    refp, modp = _over(ref_cut), _over(mod_bgra)
    blend = cv2.addWeighted(refp, 0.5, modp, 0.5, 0)
    for mask, col in ((ref_mask, (60, 220, 60)), (mod_mask, (230, 60, 230))):
        cs, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(blend, cs, -1, col, 1, cv2.LINE_AA)
    r, m = ref_mask > 0, mod_mask > 0
    diff = np.full((h, w, 3), PANEL_BG, np.uint8)
    g = cv2.cvtColor(refp, cv2.COLOR_BGR2GRAY)
    diff[r & m] = np.repeat((g[r & m].astype(np.float32) * 0.35 + 150)[:, None], 3, 1).astype(np.uint8)
    diff[r & ~m] = (50, 50, 235)             # reference only: the model is missing this
    diff[m & ~r] = (235, 120, 40)            # model only: the model has too much here
    out = []
    for img, text in ((refp, "%s  reference" % view["name"].upper()), (modp, "model"),
                      (blend, "overlay: green = ref outline, pink = model"), (diff, "red = MISSING, blue = EXTRA")):
        f = target / float(h)
        im = cv2.resize(img, (max(1, int(round(w * f))), target), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        _label(im, text, 0.5)
        out.append(im)
    return refmod.hstack(out, PANEL_BG, 6)


def run(ref_dir, out_png, views=None, objs=None, render=True):
    """Compare the current scene with a reference pack. Returns a report dict (and writes the picture)."""
    doc = load_pack(ref_dir)
    names = [n for n in doc["order"] if views is None or n in views]
    if not names:
        raise SystemExit("the reference pack has none of the requested views (it has: %s)" % ", ".join(doc["order"]))
    tris = scene_tris(objs)
    if len(tris) == 0:
        raise SystemExit("nothing to compare: the scene has no model geometry")
    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    radius = float(np.linalg.norm(hi - lo)) / 2 + 1e-6
    tmpdir = tempfile.mkdtemp(prefix="f3d_cmp_")
    cutouts = any(o.get("f3d_cutout") for o in core.scene_objects())
    rows, results = [], []
    renders = {}
    if render:
        with staging.evaluation_setup("albedo"):
            for n in names:
                v = doc["views"][n]
                zoom = float(np.clip(640.0 / max(v["size_px"]), 0.3, 3.0))
                renders[n] = render_view(v, zoom, tmpdir, radius)
    for n in names:
        v = doc["views"][n]
        zoom = float(np.clip(640.0 / max(v["size_px"]), 0.3, 3.0))
        mod_mask = silhouette(tris, v, zoom)
        h, w = mod_mask.shape
        cut = cv2.resize(refmod.read_image(os.path.join(doc["dir"], v["image"])), (w, h), interpolation=cv2.INTER_LANCZOS4)
        ref_mask = cv2.resize(cv2.imread(os.path.join(doc["dir"], v["mask"]), cv2.IMREAD_GRAYSCALE), (w, h),
                              interpolation=cv2.INTER_NEAREST)
        if render:
            mod = renders[n]
            if mod.shape[:2] != (h, w):
                mod = cv2.resize(mod, (w, h), interpolation=cv2.INTER_AREA)
            if cutouts:                       # see-through texels: the outline is what the render shows
                mod_mask = ((mod[:, :, 3] > 100) & (mod_mask > 0)).astype(np.uint8) * 255
        else:
            mod = np.dstack([np.full((h, w, 3), 200, np.uint8), mod_mask])
        rmap = None
        if v.get("region_map"):
            rmap = cv2.imread(os.path.join(doc["dir"], v["region_map"]), cv2.IMREAD_GRAYSCALE)
        res = analyse(v, ref_mask, cut[:, :, :3], mod_mask, _over(mod, (128, 128, 128)), v.get("regions", []), rmap)
        results.append(res)
        rows.append(panels(v, cut, mod, ref_mask, mod_mask))
    wmax = max(r.shape[1] for r in rows)
    sheet = np.full((sum(r.shape[0] for r in rows) + 6 * (len(rows) - 1), wmax, 3), PANEL_BG, np.uint8)
    y = 0
    for r in rows:
        sheet[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 6
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    refmod.write_image(out_png, sheet)
    shape = float(np.mean([r["iou"] for r in results]))
    des = [r["delta_e"] for r in results if r["delta_e"] is not None]
    colour = float(np.mean([max(0.0, 1.0 - d / 40.0) for d in des])) if des else 0.0
    rep = dict(image=os.path.abspath(out_png), views=results, shape=shape, colour=colour,
               score=100.0 * (0.6 * shape + 0.4 * colour), ref=doc["dir"])
    with open(os.path.splitext(out_png)[0] + ".json", "w") as f:
        json.dump(rep, f, indent=1, default=float)
    return rep


def _f(v):
    return refmod._fmt(v)


def format_report(rep, max_lines=6):
    # no single score: one number would rank a smeared projection of the pictures above a clean model
    L = ["COMPARE with %s:  outlines agree %.0f%%, colours agree %.0f%%  (per view below)" % (
        os.path.relpath(rep["ref"]) if not os.path.relpath(rep["ref"]).startswith("..") else rep["ref"],
        rep["shape"] * 100, rep["colour"] * 100)]
    for r in rep["views"]:
        ax = "x" if r["view"] in ("front", "back", "top", "bottom") else "y"
        head = "  [%s] IoU %.2f  missing %.0f%%  extra %.0f%%" % (r["view"], r["iou"], r["missing"] * 100, r["extra"] * 100)
        if r["delta_e"] is not None:
            head += "  colour dE %.0f" % r["delta_e"]
        L.append(head)
        if r["empty"]:
            L.append("      the model is not visible in this view's frame at all (wrong scale or position? the subject "
                     "must stand on z=0 at the origin and be %s tall)" % "the reference's height")
            continue
        tips = []
        dx, dy, siou = r["shift"]
        tol = 0.012
        if siou - r["iou"] > 0.04 and (abs(dx) > tol or abs(dy) > tol):
            tips.append("whole model looks offset: moving it %s%s in the picture (right/up) would raise IoU to %.2f" % (
                ("%+.3f right " % dx) if abs(dx) > tol else "", ("%+.3f up" % dy) if abs(dy) > tol else "", siou))
        for nm, d in r["edges"]:
            if abs(d) > 0.02:
                if nm in ("left", "right"):
                    side = "further right" if d > 0 else "further left"
                else:
                    side = "higher" if d > 0 else "lower"
                tips.append("%s edge of the model is %s %s than the reference" % (nm, _f(abs(d)), side))
        bad = sorted(r["bands"], key=lambda b: -abs(b["model"] - b["ref"]))
        for b in bad[:3]:
            d = b["model"] - b["ref"]
            if abs(d) < 0.03:
                continue
            extra = ""
            if b["ref_c"] is not None and b["model_c"] is not None and abs(b["model_c"] - b["ref_c"]) > 0.03:
                extra = ", centred %s too far %s" % (_f(abs(b["model_c"] - b["ref_c"])), "right" if b["model_c"] > b["ref_c"] else "left")
            tips.append("height %s-%s: model is %s %s than the reference (%s vs %s)%s" % (
                _f(b["v0"]), _f(b["v1"]), _f(abs(d)), "wider" if d > 0 else "narrower", _f(b["model"]), _f(b["ref"]), extra))
        regs = [g for g in r.get("regions", []) if g["de"] is not None and g["de"] > 14 and g["share"] >= 0.008]
        regs.sort(key=lambda g: -g["de"] * g["share"])
        for g in regs[:3]:
            tips.append("region %d should be %s but the model shows %s there (dE %.0f)" % (g["id"], g["ref"], g["model"], g["de"]))
        for t in tips[:max_lines]:
            L.append("      " + t)
    L.append("  reading these: outline IoU of 0.85 or more per view is a close match; fix the biggest outline error "
             "first. The pictures are drawings, not photographs of one object: where a view draws a part differently "
             "from the others (an eye in profile, a bow facing the viewer, a tail on the same paper side front and "
             "back), one clean 3D shape cannot match them all and should not try. Colour dE only says whether the "
             "right colours are in the right areas")
    L.append("  the numbers see the model from the reference cameras only. Torn or doubled features, smears and "
             "stray colours show in the three-quarter tiles of the sheet: judge those by eye before calling the "
             "model done")
    return "\n".join(L)
