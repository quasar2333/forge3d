"""Image-to-3D generation through hosted models: a strong base mesh for organic subjects in under a minute.

Backends are public Hugging Face Spaces called with gradio_client (no account needed; set HF_TOKEN for a larger
GPU quota). The result is only a base: `k.adopt()` cleans and aligns it, `k.paint_from_ref()` puts the real
reference colours on it.
"""
import hashlib
import json
import os
import shutil
import time

import cv2
import numpy as np

from . import refs as refmod

# our view name -> the generator's slot. 'right' (subject faces LEFT in the picture) shows the subject's own left side.
HUNYUAN_SLOTS = {"front": "mv_image_front", "back": "mv_image_back", "right": "mv_image_left", "left": "mv_image_right"}

BACKENDS = {
    "hunyuan-mv": dict(space="tencent/Hunyuan3D-2mv", multiview=True, textured=False,
                       note="Hunyuan3D-2mv: up to four views (front/back/left/right) -> untextured mesh. Best when "
                            "the reference is a turnaround sheet."),
    "hunyuan": dict(space="tencent/Hunyuan3D-2", multiview=False, textured=False,
                    note="Hunyuan3D-2: single image (the front view) -> untextured mesh."),
}
QUALITY = {"draft": dict(steps=10, octree=196), "standard": dict(steps=30, octree=256), "high": dict(steps=40, octree=384)}


class GenError(Exception):
    pass


def _client(space):
    try:
        from gradio_client import Client
    except ImportError:
        raise GenError("the gen3d component is not installed. Run:  python3 <skill>/scripts/f3d.py setup --extra gen3d")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    kw = dict(verbose=False)
    if token:
        kw["token"] = token
    try:
        return Client(space, **kw)
    except TypeError:
        kw.pop("token", None)
        if token:
            kw["hf_token"] = token
        return Client(space, **kw)
    except Exception as e:  # noqa: BLE001
        raise GenError("cannot reach the %s Space (%s: %s)" % (space, type(e).__name__, str(e)[:200]))


def _file(res):
    if isinstance(res, dict):
        res = res.get("value") or res.get("path") or res.get("name")
    return res if isinstance(res, str) and os.path.isfile(res) else None


def prepare_inputs(source, workdir, size=1024):
    """Reference pack folder or single image -> {view: square RGBA png}, all views at one scale."""
    os.makedirs(workdir, exist_ok=True)
    out = {}
    if os.path.isdir(source):
        with open(os.path.join(source, "ref.json")) as f:
            doc = json.load(f)
        views = {n: v for n, v in doc["views"].items() if n in HUNYUAN_SLOTS}
        if not views:
            raise GenError("the reference pack has no front / back / left / right view to generate from")
        if doc.get("like"):
            return _layer_inputs(source, views, workdir, size)
        ppu = {n: v["px_per_unit"] for n, v in views.items()}
        ext = max(max(v["size_px"][0] / ppu[n], v["size_px"][1] / ppu[n]) for n, v in views.items())
        for n, v in views.items():
            im = refmod.read_image(os.path.join(source, v["image"]))
            s = (size * 0.9 / ext) / ppu[n]
            im = cv2.resize(im, (max(1, int(round(im.shape[1] * s))), max(1, int(round(im.shape[0] * s)))),
                            interpolation=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA)
            canvas = np.zeros((size, size, 4), np.uint8)
            h, w = im.shape[:2]
            # keep the subject's own axis in the middle of the picture so the views agree about where it stands
            cx = int(round(size / 2 - v["origin_px"][0] * s)) if n in ("front", "back") else (size - w) // 2
            cx = min(max(cx, 0), size - w)
            y0 = int(size * 0.95) - h
            canvas[max(y0, 0):max(y0, 0) + h, cx:cx + w] = im[:size - max(y0, 0)]
            p = os.path.join(workdir, "in_%s.png" % n)
            refmod.write_image(p, canvas)
            out[n] = p
        return out
    im = refmod.read_image(source)
    mask, _info = refmod.matte(im)
    if mask.any():
        ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        im = im[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1].copy()
        im[:, :, 3] = mask[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    s = size * 0.9 / max(im.shape[:2])
    im = cv2.resize(im, (int(round(im.shape[1] * s)), int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LANCZOS4)
    canvas = np.zeros((size, size, 4), np.uint8)
    h, w = im.shape[:2]
    canvas[(size - h) // 2:(size - h) // 2 + h, (size - w) // 2:(size - w) // 2 + w] = im
    p = os.path.join(workdir, "in_front.png")
    refmod.write_image(p, canvas)
    return {"front": p}


def _layer_inputs(source, views, workdir, size):
    """Views of a layer pack on square canvases. A layer need not stand on the ground or around the axis (a wig,
    a hat, a sword in one hand), so the views are placed by their world coordinates: every height lands on the
    same row in every view, and front / back (right / left) agree about the middle."""
    sign = {"front": 1.0, "right": 1.0, "back": -1.0, "left": -1.0}
    vmin = min(v["bbox"][1] for v in views.values())
    vmax = max(v["bbox"][3] for v in views.values())
    mid, width = {}, [vmax - vmin]
    for group in (("front", "back"), ("right", "left")):
        spans = [sorted((sign[n] * views[n]["bbox"][0], sign[n] * views[n]["bbox"][2])) for n in group if n in views]
        if spans:
            lo, hi = min(a for a, _b in spans), max(b for _a, b in spans)
            width.append(hi - lo)
            for n in group:
                mid[n] = sign[n] * (lo + hi) / 2.0
    S = size * 0.9 / max(width)
    out = {}
    for n, v in views.items():
        im = refmod.read_image(os.path.join(source, v["image"]))
        f = S / v["px_per_unit"]
        im = cv2.resize(im, (max(1, int(round(im.shape[1] * f))), max(1, int(round(im.shape[0] * f)))),
                        interpolation=cv2.INTER_LANCZOS4 if f > 1 else cv2.INTER_AREA)
        x0 = int(round(size / 2.0 + (v["frame"][0] - mid[n]) * S))
        y0 = int(round(size / 2.0 + ((vmax + vmin) / 2.0 - v["frame"][3]) * S))
        canvas = np.zeros((size, size, 4), np.uint8)
        h, w = im.shape[:2]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        ww, hh = min(w - sx0, size - dx0), min(h - sy0, size - dy0)
        if ww > 0 and hh > 0:
            canvas[dy0:dy0 + hh, dx0:dx0 + ww] = im[sy0:sy0 + hh, sx0:sx0 + ww]
        p = os.path.join(workdir, "in_%s.png" % n)
        refmod.write_image(p, canvas)
        out[n] = p
    return out


def _hunyuan(space, inputs, quality, seed, multiview):
    from gradio_client import handle_file
    q = QUALITY[quality]
    c = _client(space)
    kw = dict(caption=None, image=None, mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
              steps=q["steps"], guidance_scale=5.0, seed=seed, octree_resolution=q["octree"], check_box_rembg=False,
              num_chunks=8000, randomize_seed=False)
    if multiview:
        for n, p in inputs.items():
            kw[HUNYUAN_SLOTS[n]] = handle_file(p)
    else:
        kw["image"] = handle_file(inputs.get("front") or next(iter(inputs.values())))
    res = c.predict(api_name="/shape_generation", **kw)
    f = _file(res[0])
    if not f:
        raise GenError("the Space returned no mesh file")
    return f


def _remember(out, cached, meta):
    """Keep a machine-wide copy: free GPU quota is scarce, identical requests should never be paid for twice."""
    try:
        if not os.path.isfile(cached):
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            shutil.copy(out, cached)
            with open(cached + ".json", "w") as fh:
                json.dump(meta, fh)
    except OSError:
        pass


RUNNERS = {"hunyuan-mv": _hunyuan, "hunyuan": _hunyuan}


def generate(source, out, backend="auto", quality="standard", seed=1234, force=False, log=print):
    """Generate a mesh from a reference pack folder (or a single image). Returns the output path."""
    out = os.path.abspath(out)
    workdir = os.path.join(os.path.dirname(out), ".f3d-gen")
    inputs = prepare_inputs(source, workdir)
    if backend == "auto":
        order = ["hunyuan-mv", "hunyuan"] if len(inputs) > 1 else ["hunyuan", "hunyuan-mv"]
    else:
        if backend not in BACKENDS:
            raise GenError("unknown backend %r (have: %s)" % (backend, ", ".join(BACKENDS)))
        order = [backend]
    h = hashlib.sha1()
    for n in sorted(inputs):
        with open(inputs[n], "rb") as f:
            h.update(n.encode() + f.read())
    h.update(json.dumps([order, quality, seed]).encode())
    key = h.hexdigest()[:16]
    stamp = os.path.join(workdir, "done-%s.json" % key)
    store = os.path.join(os.environ.get("F3D_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "forge3d"),
                         "gen3d-cache")
    cached = os.path.join(store, key + ".glb")
    if os.path.isfile(out) and os.path.isfile(stamp) and not force:
        log("gen3d: %s is up to date (same pictures and settings); pass --force to generate again" % out)
        meta = json.load(open(stamp))
        _remember(out, cached, meta)
        return out, meta
    if os.path.isfile(cached) and os.path.isfile(cached + ".json") and not force:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy(cached, out)
        meta = json.load(open(cached + ".json"))
        with open(stamp, "w") as fh:
            json.dump(meta, fh)
        log("gen3d: reused an earlier generation for the same pictures and settings -> %s (--force generates anew)" % out)
        return out, meta
    errors = []
    for b in order:
        cfg = BACKENDS[b]
        used = inputs if cfg["multiview"] else {"front": inputs.get("front") or next(iter(inputs.values()))}
        log("gen3d: asking %s (%s) with view(s): %s ..." % (b, cfg["space"], ", ".join(used)))
        t = time.time()
        try:
            f = RUNNERS[b](cfg["space"], used, quality, seed, cfg["multiview"])
        except GenError as e:
            errors.append("%s: %s" % (b, e))
            log("gen3d:   %s did not deliver (%s)" % (b, str(e)[:160]))
            continue
        except Exception as e:  # noqa: BLE001 - network / quota / remote failures all end up here
            msg = str(e)
            hint = ""
            low = msg.lower()
            if "quota" in low or ("gpu" in low and "exceed" in low):
                hint = (" -> the free GPU quota of this network address is used up for now (anonymous use allows roughly "
                        "eight generations per day; the message above says when it refills). A free Hugging Face token "
                        "raises it: export HF_TOKEN=hf_...")
            errors.append("%s: %s: %s%s" % (b, type(e).__name__, msg[:300], hint))
            log("gen3d:   %s did not deliver (%s: %s)" % (b, type(e).__name__, msg[:160]))
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copy(f, out)
        meta = dict(backend=b, space=cfg["space"], textured=cfg["textured"], views=sorted(used), quality=quality,
                    seed=seed, seconds=round(time.time() - t, 1))
        with open(stamp, "w") as fh:
            json.dump(meta, fh)
        _remember(out, cached, meta)
        log("gen3d: saved %s (%.1f MB, %s, %.0fs)" % (out, os.path.getsize(out) / 1e6, b, time.time() - t))
        return out, meta
    raise GenError("no image-to-3D backend produced a mesh:\n  " + "\n  ".join(errors) +
                   "\nGo on without it: build the shape from the pictures' own outlines (k.volume(front_mask, "
                   "side_mask), k.inflate(mask) + k.drape; for hair see \"Only when the quota is used up\" in "
                   "references/parts.md), or put a mesh from any other generator at %s." % out)
