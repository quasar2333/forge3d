"""Multi-view contact sheets: the model seen the way an art director orbits it."""
import math
import os
import tempfile

import bpy
import cv2
import numpy as np

from .kit import core, staging

DEFAULT_VIEWS = "front,az40el10,right,az140el10,back,az-40el10,top,clay:az40el10"
MODES = ("color", "clay", "silhouette", "parts")
BG = (225, 220, 217)       # BGR
AXIS_HINT = {"front": "X right, Z up", "back": "X left, Z up", "right": "Y right, Z up (model faces left)",
             "left": "Y left, Z up (model faces right)", "top": "X right, Y up (front at bottom)",
             "bottom": "X right"}


def parse_views(spec):
    """'front,clay:iso,az30el20' -> [(mode, name, az, el)]"""
    out = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        mode, name = ("color", tok)
        if ":" in tok:
            mode, name = tok.split(":", 1)
        if mode not in MODES:
            raise SystemExit("unknown render mode %r (use %s)" % (mode, ", ".join(MODES)))
        if name in staging.VIEWS:
            az, el = staging.VIEWS[name]
        elif name.startswith("az"):
            try:
                a, e = name[2:].split("el")
                az, el = float(a), float(e)
            except ValueError:
                raise SystemExit("bad view %r: custom angles look like az30el20" % name)
        else:
            raise SystemExit("unknown view %r (have: %s, or azNNelNN)" % (name, ", ".join(staging.VIEWS)))
        out.append((mode, name, az, el))
    return out


def is_ortho_view(az, el):
    return abs(el) < 1e-6 and abs(az % 90) < 1e-6 or abs(abs(el) - 90) < 1e-6


def composite(path, bg=BG):
    """RGBA render -> (BGR image over bg, alpha mask 0..1)."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError("render missing: %s" % path)
    if im.shape[2] == 3:
        return im, np.ones(im.shape[:2], np.float32)
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    out = im[:, :, :3].astype(np.float32) * a + np.array(bg, np.float32) * (1 - a)
    return out.astype(np.uint8), a[:, :, 0]


def nice_length(target):
    if target <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(target))
    for m in (5, 2, 1):
        if m * e <= target:
            return m * e
    return e


def fmt_len(v):
    if v >= 1:
        return "%g m" % v
    if v >= 0.01:
        return "%g cm" % (v * 100)
    return "%g mm" % (v * 1000)


def put(img, text, org, scale=0.5, color=(40, 40, 40), thick=1, plate=(236, 233, 230)):
    """Text on a solid plate so it stays legible over the render and after downscaling."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if plate is not None:
        cv2.rectangle(img, (x - 2, y - th - 3), (x + tw + 2, y + base), plate, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_views(views, tile=448, objs=None, tmpdir=None):
    """Render each requested view. Returns list of dicts: mode, name, az, el, image (BGR), alpha, ortho, scale, cam_z."""
    items = core.scene_objects() if objs is None else core.expand(objs)
    if not items:
        raise RuntimeError("nothing to render: the scene has no model geometry")
    pts = staging._sample_points(items, 20000)
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="f3d_")
    # one shared scale for every straight-on view so proportions can be compared between tiles
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    common = float(max(ext)) * 1.12
    results = [None] * len(views)
    for mode in MODES:
        idx = [i for i, v in enumerate(views) if v[0] == mode]
        if not idx:
            continue
        with staging.evaluation_setup(mode):
            for i in idx:
                _m, name, az, el = views[i]
                ortho = is_ortho_view(az, el)
                cam = staging.temp_camera(az, el, pts, ortho=ortho, lens=85.0, margin=1.12,
                                          ortho_scale=common if ortho else None)
                path = os.path.join(tmpdir, "tile_%02d_%s_%s.png" % (i, mode, name))
                staging.render_tile(cam, path, (tile, tile))
                info = dict(mode=mode, name=name, az=az, el=el, path=path, ortho=ortho,
                            scale=cam.data.ortho_scale if ortho else None,
                            cam_z=cam.matrix_world.translation.z)
                staging.remove_camera(cam)
                img, alpha = composite(path)
                info["image"], info["alpha"] = img, alpha
                results[i] = info
    return results


def decorate(info, tile):
    img = info["image"].copy()
    label = info["name"].upper() + ("" if info["mode"] == "color" else " [%s]" % info["mode"])
    put(img, label, (8, 20), 0.55, (30, 30, 30), 1)
    hint = AXIS_HINT.get(info["name"])
    if hint:
        put(img, hint, (8, 38), 0.38, (90, 90, 90), 1)
    if info["ortho"] and info["scale"]:
        px_per_m = tile / info["scale"]
        if abs(info["el"]) < 1e-6:
            row = int(round(tile / 2 + info["cam_z"] * px_per_m))
            if 0 <= row < tile:
                cv2.line(img, (0, row), (tile, row), (150, 150, 235), 1, cv2.LINE_AA)
                put(img, "z=0", (tile - 40, row - 4), 0.36, (90, 90, 200), 1)
        length = nice_length(info["scale"] * 0.25)
        w = int(round(length * px_per_m))
        y = tile - 12
        cv2.line(img, (10, y), (10 + w, y), (40, 40, 40), 2)
        cv2.line(img, (10, y - 4), (10, y + 4), (40, 40, 40), 2)
        cv2.line(img, (10 + w, y - 4), (10 + w, y + 4), (40, 40, 40), 2)
        put(img, fmt_len(length), (16 + w, y + 4), 0.4, (40, 40, 40), 1)
    cv2.rectangle(img, (0, 0), (tile - 1, tile - 1), (170, 170, 170), 1)
    return img


def make_sheet(out_path, views=DEFAULT_VIEWS, tile=448, cols=4, objs=None, title=None):
    """Render a labelled contact sheet. Returns (path, per-view infos)."""
    specs = parse_views(views) if isinstance(views, str) else views
    infos = render_views(specs, tile, objs)
    tiles = [decorate(i, tile) for i in infos]
    cols = min(cols, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    header = 26
    sheet = np.full((rows * tile + header, cols * tile, 3), BG, np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[header + r * tile: header + (r + 1) * tile, c * tile:(c + 1) * tile] = t
    bb = core.bbox(objs)
    s = bb.size
    text = "%ssize W(X) %.3f x D(Y) %.3f x H(Z) %.3f m   |   straight-on tiles share one scale   |   model faces -Y" % (
        (title + "   |   ") if title else "", s.x, s.y, s.z)
    put(sheet, text, (8, 18), 0.48, (30, 30, 30), 1)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return out_path, infos
