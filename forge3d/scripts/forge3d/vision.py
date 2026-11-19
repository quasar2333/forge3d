"""Small vision models that run on this machine (onnxruntime, CPU): no account, no quota, the same answer every
time.

- `upscale`: waifu2x (swin_unet, art) redraws a drawing four times larger with crisp lines. A turnaround sheet
  gives each view a few hundred pixels; stretched over a model that is rendered at 1500 that is mush. Nothing
  is invented: it is the reference itself, only sharp.
- `detect`: finds drawn faces and eyes. The one thing a painted character must never get wrong is that a face
  has two eyes: the painter uses the boxes to give the face to the front picture alone, and the build counts
  the eyes it can see on the finished model from every side.

Models are fetched once into $F3D_HOME/models and checked against a pinned SHA-256. Without onnxruntime or
without the files (offline), everything here reports itself unavailable and the callers carry on without it.
"""
import hashlib
import os
import sys

import numpy as np

HF = "https://huggingface.co/%s/resolve/main/%s"
MODELS = {
    # name: (repository, file, sha256, MB)
    "upscale4x": ("deepghs/waifu2x_onnx", "20230131/onnx_models/swin_unet/art/noise1_scale4x.onnx",
                  "33b418a36e0617e3f7a06c1e4dbb1fc1eb7cdf6a1cd65b2707409d41a311bb76", 19),
    "eye": ("deepghs/anime_eye_detection", "eye_detect_v1.0_s/model.onnx",
            "7c5f0259103cc407e2a0f0b047a51271246b9525d27920315295a54367c5c583", 45),
    "face": ("deepghs/anime_face_detection", "face_detect_v1.4_s/model.onnx",
             "403b5bc93b6ff789b7d183418df4a1364049bac00c24acd927604a7ff6891483", 45),
}
_sessions = {}
_failed = {}


def _log(msg):
    print("[f3d] " + msg, file=sys.stderr, flush=True)


def models_dir():
    home = os.environ.get("F3D_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "forge3d")
    return os.path.join(home, "models")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model_path(name, fetch=True):
    """Local path of a model file, downloading it on first use. None when it cannot be had."""
    repo, fname, sha, mb = MODELS[name]
    path = os.path.join(models_dir(), name + ".onnx")
    if os.path.isfile(path):
        return path
    if not fetch or os.environ.get("F3D_OFFLINE"):
        return None
    import urllib.request
    os.makedirs(models_dir(), exist_ok=True)
    tmp = path + ".part"
    _log("fetching the %s model (%d MB, once) ..." % (name, mb))
    try:
        req = urllib.request.Request(HF % (repo, fname), headers={"User-Agent": "forge3d"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                f.write(block)
        if _sha(tmp) != sha:
            raise IOError("checksum mismatch")
        os.replace(tmp, path)
        return path
    except Exception as e:                                  # offline, proxy, moved file: carry on without
        if os.path.exists(tmp):
            os.remove(tmp)
        _failed[name] = "%s: %s" % (type(e).__name__, e)
        return None


def session(name):
    """An onnxruntime session for a model, or None (the reason is in `why_not(name)`)."""
    if name in _sessions:
        return _sessions[name]
    if name in _failed:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        _failed[name] = "onnxruntime is not installed"
        return None
    path = model_path(name)
    if path is None:
        _failed.setdefault(name, "model file not available")
        return None
    try:
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        # a few threads that sleep when idle: the default (every core, busy-waiting) is several times slower as
        # soon as anything else runs on the machine, and part agents always run beside each other
        opt.intra_op_num_threads = max(2, min(6, (os.cpu_count() or 4) // 2))
        opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
        _sessions[name] = ort.InferenceSession(path, opt, providers=["CPUExecutionProvider"])
    except Exception as e:
        _failed[name] = "%s: %s" % (type(e).__name__, e)
        return None
    return _sessions[name]


def why_not(name):
    return _failed.get(name)


def available(name):
    return session(name) is not None


# ---------------------------------------------------------------- super-resolution
def upscale(bgr, factor=4.0, keep=None):
    """A BGR (or BGRA) uint8 picture `factor` times larger (1 < factor <= 4), drawn sharp by waifu2x; alpha is
    enlarged smoothly. keep: bool mask of the part that matters; tiles without any of it are only stretched
    (most of a sheet is empty paper). Returns None when the model is not available."""
    import cv2
    s = session("upscale4x")
    if s is None:
        return None
    alpha = bgr[:, :, 3] if bgr.shape[2] == 4 else None
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr[:, :, :3]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    off, core, step = 8, 240, 176                            # the net eats 8 px on every side; tiles overlap by 64
    ny, nx = max(1, -(-(h - core) // step) + 1), max(1, -(-(w - core) // step) + 1)
    H, W = (ny - 1) * step + core, (nx - 1) * step + core
    pad = np.pad(rgb, ((off, off + H - h), (off, off + W - w), (0, 0)), mode="edge")
    use = None if keep is None else np.pad(keep, ((off, off + H - h), (off, off + W - w)), mode="constant")
    out = np.zeros((H * 4, W * 4, 3), np.float32)
    wsum = np.zeros((H * 4, W * 4, 1), np.float32)
    ramp = np.minimum(np.arange(core * 4) + 1.0, np.arange(core * 4)[::-1] + 1.0)
    win = np.minimum(np.minimum.outer(ramp, ramp), 128.0)[:, :, None].astype(np.float32)
    name = s.get_inputs()[0].name
    for iy in range(ny):
        for ix in range(nx):
            y0, x0 = iy * step, ix * step
            tile = pad[y0:y0 + core + 2 * off, x0:x0 + core + 2 * off]
            if float(tile.max() - tile.min()) < 0.012:      # empty paper: nothing to redraw
                y = np.broadcast_to(tile.mean((0, 1)), (core * 4, core * 4, 3))
            elif use is not None and not use[y0:y0 + core + 2 * off, x0:x0 + core + 2 * off].any():
                y = cv2.resize(tile[off:off + core, off:off + core], (core * 4, core * 4), interpolation=cv2.INTER_LINEAR)
            else:
                y = s.run(None, {name: np.ascontiguousarray(tile.transpose(2, 0, 1)[None])})[0][0].transpose(1, 2, 0)
            out[y0 * 4:(y0 + core) * 4, x0 * 4:(x0 + core) * 4] += y * win
            wsum[y0 * 4:(y0 + core) * 4, x0 * 4:(x0 + core) * 4] += win
    big = np.clip(out[:h * 4, :w * 4] / wsum[:h * 4, :w * 4], 0.0, 1.0)
    big = cv2.cvtColor((big * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if alpha is not None:
        big = np.dstack([big, cv2.resize(alpha, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)])
    if factor < 3.999:
        big = cv2.resize(big, (int(round(w * factor)), int(round(h * factor))), interpolation=cv2.INTER_AREA)
    return big


# ---------------------------------------------------------------- detection
def detect(bgr, kind="eye", conf=0.35, iou=0.5, size=640):
    """Boxes of drawn `kind` ('eye' or 'face') in a BGR / BGRA picture (transparent parts count as white).
    -> [(x0, y0, x1, y1, score), ...] in pixels, best first; None when the model is not available."""
    import cv2
    s = session(kind)
    if s is None:
        return None
    if bgr.shape[2] == 4:
        a = bgr[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (bgr[:, :, :3] * a + 255.0 * (1.0 - a)).astype(np.uint8)
    h, w = bgr.shape[:2]
    r = min(size / float(h), size / float(w))
    nh, nw = max(1, int(round(h * r))), max(1, int(round(w * r)))
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    y = s.run(None, {s.get_inputs()[0].name: x})[0][0]
    if y.shape[0] < y.shape[1]:
        y = y.T                                              # -> (anchors, 4 + classes)
    score = y[:, 4:].max(1)
    y, score = y[score > conf], score[score > conf]
    if not len(y):
        return []
    boxes = np.stack([y[:, 0] - y[:, 2] / 2, y[:, 1] - y[:, 3] / 2, y[:, 0] + y[:, 2] / 2, y[:, 1] + y[:, 3] / 2], 1) / r
    keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                            [float(v) for v in score], conf, iou)
    keep = sorted(np.array(keep).reshape(-1).tolist(), key=lambda i: -score[i])
    return [(float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3]), float(score[i])) for i in keep]


def status():
    """{model: 'ready' | 'not fetched yet' | reason} for `f3d doctor`."""
    out = {}
    try:
        import onnxruntime                                   # noqa: F401
    except ImportError:
        return {n: "onnxruntime is not installed" for n in MODELS}
    for n in MODELS:
        out[n] = "ready" if model_path(n, fetch=False) else "not fetched yet (%d MB on first use)" % MODELS[n][3]
    return out
