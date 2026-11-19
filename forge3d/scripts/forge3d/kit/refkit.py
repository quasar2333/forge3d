"""Reference packs inside build scripts: measurements, masks and outlines in world units.

    R = k.ref()                      # the pack given to `f3d build --ref`, or k.ref("refs")
    F = R["front"]                   # a view: 'front', 'back', 'right', 'left', 'top', 'bottom'
    F.bbox                           # (u0, v0, u1, v1) of the silhouette; u = picture right, v = picture up
    F.span(0.5)                      # (u_left, u_right) of the silhouette at height 0.5
    m = F.mask(regions=[4, 8])       # part of the picture as a mask (numbered regions are in <view>_marks.png)
    m.outline()                      # [(u, v), ...] polygon for k.extrude / k.inflate
"""
import json
import math
import os

import numpy as np

_state = {"default": None, "cache": {}}


def set_default(path):
    _state["default"] = os.path.abspath(path) if path else None
    _state["cache"].clear()


def default_path():
    return _state["default"]


def _cv2():
    import cv2
    return cv2


def hex_to_bgr(h):
    s = h.lstrip("#")
    return (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))


class RefMask:
    """A boolean picture region tied to a reference view. Combine with | & - ; all sizes are world units."""

    def __init__(self, view, array):
        self.view = view
        self.array = np.asarray(array, dtype=bool)

    # -- set algebra
    def __or__(self, other):
        return RefMask(self.view, self.array | other.array)

    def __and__(self, other):
        return RefMask(self.view, self.array & other.array)

    def __sub__(self, other):
        return RefMask(self.view, self.array & ~other.array)

    def __invert__(self):
        return RefMask(self.view, ~self.array & self.view.mask().array)

    @property
    def empty(self):
        return not self.array.any()

    @property
    def area(self):
        """Area in world units squared."""
        return float(self.array.sum()) / self.view.px_per_unit ** 2

    @property
    def bbox(self):
        """(u0, v0, u1, v1) in world units."""
        ys, xs = np.flatnonzero(self.array.any(1)), np.flatnonzero(self.array.any(0))
        if len(ys) == 0:
            raise ValueError("the mask is empty")
        v = self.view
        return (v.u_of(xs[0]), v.v_of(ys[-1] + 1), v.u_of(xs[-1] + 1), v.v_of(ys[0]))

    @property
    def center(self):
        b = self.bbox
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def _px(self, d):
        return max(1, int(round(abs(d) * self.view.px_per_unit)))

    def grow(self, d):
        """Dilate by d world units."""
        cv2 = _cv2()
        k = 2 * self._px(d) + 1
        return RefMask(self.view, cv2.dilate(self.array.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0)

    def shrink(self, d):
        cv2 = _cv2()
        k = 2 * self._px(d) + 1
        return RefMask(self.view, cv2.erode(self.array.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0)

    def close(self, d):
        """Bridge gaps and notches narrower than about 2 d."""
        cv2 = _cv2()
        k = 2 * self._px(d) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        pad = k
        a = cv2.copyMakeBorder(self.array.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, ker)
        return RefMask(self.view, a[pad:-pad, pad:-pad] > 0)

    def open(self, d):
        """Remove spikes and specks thinner than about 2 d."""
        cv2 = _cv2()
        k = 2 * self._px(d) + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return RefMask(self.view, cv2.morphologyEx(self.array.astype(np.uint8), cv2.MORPH_OPEN, ker) > 0)

    def fill(self):
        """Fill enclosed holes."""
        from scipy import ndimage
        return RefMask(self.view, ndimage.binary_fill_holes(self.array))

    def hull(self):
        """The convex hull of the mask: the outline with every notch and bite bridged. For parts that are convex
        in reality but drawn with other things overlapping them (a head behind ears and fins, a torso behind
        arms): select what is visible of them by colour, then take the hull."""
        cv2 = _cv2()
        pts = cv2.findNonZero(self.array.astype(np.uint8))
        if pts is None:
            raise ValueError("the mask is empty")
        a = np.zeros(self.array.shape, np.uint8)
        cv2.fillConvexPoly(a, cv2.convexHull(pts), 1)
        return RefMask(self.view, a > 0)

    def largest(self, n=1):
        """Keep the n largest connected pieces."""
        cv2 = _cv2()
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(self.array.astype(np.uint8), connectivity=8)
        if cnt <= 2:
            return self
        order = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:n] + 1
        return RefMask(self.view, np.isin(lab, order))

    def clip(self, u0=None, v0=None, u1=None, v1=None):
        """Keep only the part inside a world-unit box (None = unbounded)."""
        v = self.view
        a = np.zeros_like(self.array)
        h, w = a.shape
        x0 = 0 if u0 is None else int(np.clip(round(v.x_of(u0)), 0, w))
        x1 = w if u1 is None else int(np.clip(round(v.x_of(u1)), 0, w))
        y0 = 0 if v1 is None else int(np.clip(round(v.y_of(v1)), 0, h))
        y1 = h if v0 is None else int(np.clip(round(v.y_of(v0)), 0, h))
        a[y0:y1, x0:x1] = self.array[y0:y1, x0:x1]
        return RefMask(v, a)

    def mirror(self):
        """Mirror about the view's u = 0 line (the symmetry plane in front / back views)."""
        v = self.view
        h, w = self.array.shape
        xs = np.arange(w)
        src = np.round(2 * v.origin_px[0] - 1 - xs).astype(int)
        ok = (src >= 0) & (src < w)
        a = np.zeros_like(self.array)
        a[:, ok] = self.array[:, src[ok]]
        return RefMask(v, a)

    def symmetric(self, mode="union"):
        """Make the mask mirror-symmetric about u = 0: 'union', 'intersect', 'left' or 'right' (copy that half)."""
        m = self.mirror()
        if mode == "union":
            return self | m
        if mode == "intersect":
            return self & m
        half = self.clip(u1=0) if mode == "left" else self.clip(u0=0)
        return half | half.mirror()

    def spans(self, v):
        """Runs [(u0, u1), ...] of the mask at height v."""
        vw = self.view
        y = int(round(vw.y_of(v) - 0.5))
        if y < 0 or y >= self.array.shape[0]:
            return []
        row = self.array[y]
        idx = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        return [(vw.u_of(a), vw.u_of(b)) for a, b in zip(idx[0::2], idx[1::2])]

    def span(self, v):
        """(u_min, u_max) of the mask at height v, or None where it is empty."""
        s = self.spans(v)
        return (s[0][0], s[-1][1]) if s else None

    def profile(self, n=24):
        """[(v, u_min, u_max), ...] bottom to top at n heights: feed it to k.loft / k.lathe style shapes."""
        b = self.bbox
        out = []
        for i in range(n):
            v = b[1] + (i + 0.5) / n * (b[3] - b[1])
            s = self.span(v)
            if s:
                out.append((v, s[0], s[1]))
        return out

    def outline(self, tol=0.004, holes=False):
        """Largest contour as a world-unit polygon [(u, v), ...]; tol = simplification tolerance.
        holes=True returns (outer, [hole polygons])."""
        cv2 = _cv2()
        a = self.array.astype(np.uint8)
        cs, hier = cv2.findContours(a, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if not cs:
            raise ValueError("the mask is empty")
        vw = self.view
        eps = max(0.3, tol * vw.px_per_unit)
        outer = [i for i in range(len(cs)) if hier[0][i][3] < 0]
        best = max(outer, key=lambda i: cv2.contourArea(cs[i]))

        def poly(c):
            c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
            return [(vw.u_of(x + 0.5), vw.v_of(y + 0.5)) for x, y in c]
        if not holes:
            return poly(cs[best])
        hs = [poly(cs[i]) for i in range(len(cs)) if hier[0][i][3] == best and cv2.contourArea(cs[i]) > 6]
        return poly(cs[best]), [h for h in hs if len(h) >= 3]

    def save(self, path):
        _cv2().imwrite(path, self.array.astype(np.uint8) * 255)
        return path


class RefView:
    """One picture of the subject with its mapping to world coordinates."""

    def __init__(self, pack, rec):
        self.pack = pack
        self.rec = rec
        self.name = rec["name"]
        self.az, self.el = rec["az"], rec["el"]
        self.u_vec, self.v_vec = np.array(rec["u_vec"], float), np.array(rec["v_vec"], float)
        self.px_per_unit = float(rec["px_per_unit"])
        self.origin_px = tuple(rec["origin_px"])
        self.frame = tuple(rec["frame"])
        self.bbox = tuple(rec["bbox"])
        self.size_px = tuple(rec["size_px"])
        self.regions = {r["id"]: r for r in rec.get("regions", [])}
        self._mask = self._img = self._tex = self._rmap = None

    def __repr__(self):
        return "<RefView %s bbox=(%.3f, %.3f, %.3f, %.3f)>" % ((self.name,) + self.bbox)

    def __getstate__(self):                         # saved with a cuboid part (cut-out masks): without the pictures
        st = dict(self.__dict__)
        for key in ("_mask", "_img", "_tex", "_rmap", "_flat"):
            if key in st:
                st[key] = None
        return st

    # -- pixel <-> world
    def u_of(self, x):
        return (x - self.origin_px[0]) / self.px_per_unit

    def v_of(self, y):
        return (self.origin_px[1] - y) / self.px_per_unit

    def x_of(self, u):
        return self.origin_px[0] + u * self.px_per_unit

    def y_of(self, v):
        return self.origin_px[1] - v * self.px_per_unit

    def px(self, points):
        """Pixel coordinates of the ORIGINAL reference image (x right, y down) -> [(u, v), ...] in world units.
        Handy when you read positions off the picture you were given rather than off the grid."""
        bx, by = self.rec["sheet_box"][0], self.rec["sheet_box"][1]
        f = float(self.rec.get("source_scale", 1.0))       # small pictures are enlarged before they are measured
        return [(self.u_of(x * f - bx), self.v_of(y * f - by)) for x, y in points]

    def around(self):
        """For a view of a layer pack: where the WHOLE subject is, on this view's pixel grid (bool array). A layer
        may lie anywhere behind the rest of the subject (hair behind a body) without that showing in its own
        outline. None for an ordinary pack."""
        like = self.pack.doc.get("like")
        if not like:
            return None
        base = os.path.normpath(os.path.join(self.pack.dir, like))
        try:
            with open(os.path.join(base, "ref.json")) as f:
                rec = json.load(f)["views"].get(self.name)
        except OSError:
            return None
        if rec is None:
            return None
        cv2 = _cv2()
        m = cv2.imread(os.path.join(base, rec["mask"]), 0)
        if m is None:
            return None
        k = rec["px_per_unit"] / self.px_per_unit           # layer pixel -> base pixel
        A = np.float32([[k, 0, rec["origin_px"][0] - k * self.origin_px[0]],
                        [0, k, rec["origin_px"][1] - k * self.origin_px[1]]])
        w, h = self.size_px
        return cv2.warpAffine(m, A, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0

    def shown(self):
        """For a view of a layer pack: where the original picture itself shows this layer (bool array); the rest
        of the layer's outline was hidden behind other layers and is newly drawn. None for an ordinary pack."""
        f = self.rec.get("shown")
        if not f or not os.path.isfile(os.path.join(self.pack.dir, f)):
            return None
        return _cv2().imread(os.path.join(self.pack.dir, f), 0) > 0

    @property
    def eyes(self):
        """Drawn eyes found in this view: [(u0, v0, u1, v1), ...] in world units (empty: none, or no detector)."""
        return [tuple(b) for b in self.rec.get("eyes") or []]

    @property
    def faces(self):
        """Drawn faces found in this view: [(u0, v0, u1, v1), ...] in world units."""
        return [tuple(b) for b in self.rec.get("faces") or []]

    @property
    def toward_camera(self):
        """Unit vector from the subject toward this view's camera."""
        a, e = math.radians(self.az), math.radians(self.el)
        return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])

    def world(self, u, v, depth=0.0):
        """World point for picture coordinates (u, v); depth moves it toward the camera."""
        p = self.u_vec * u + self.v_vec * v + self.toward_camera * depth
        return (float(p[0]), float(p[1]), float(p[2]))

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    def path(self, key):
        return os.path.join(self.pack.dir, self.rec[key])

    # -- pixels
    def _full_mask(self):
        if self._mask is None:
            m = _cv2().imread(self.path("mask"), 0)
            self._mask = m > 0
        return self._mask

    @property
    def image(self):
        """Cutout as a BGRA uint8 array."""
        if self._img is None:
            self._img = _cv2().imread(self.path("image"), -1)
        return self._img

    @property
    def texture(self):
        """Cutout colours bled past the silhouette edge, BGR uint8 (what projection painting samples)."""
        if self._tex is None:
            self._tex = _cv2().imread(self.path("texture"), 1)
        return self._tex

    @property
    def flat(self):
        """The texture as flat colour areas, without its ink lines (BGR uint8)."""
        if getattr(self, "_flat", None) is None:
            f = self.rec.get("flat")
            if f and self.rec.get("flat_kind") == "cel" and os.path.isfile(self.path("flat")):
                self._flat = _cv2().imread(self.path("flat"), 1)
            else:                                   # packs made by older versions: derive it from the texture
                from .. import refs
                self._flat = refs.cel_flat(self.texture, 0.0045 * max(self.texture.shape[:2]))[0]
        return self._flat

    def _region_map(self):
        if self._rmap is None:
            if not self.rec.get("region_map"):
                raise ValueError("this pack has no region maps (re-run f3d ref without --regions 0)")
            self._rmap = _cv2().imread(self.path("region_map"), 0)
        return self._rmap

    def mask(self, regions=None, box=None, polygon=None, color=None, tol=28.0):
        """Part of the silhouette as a RefMask. With no arguments: the whole silhouette. Arguments combine (AND).

        regions: ids from <view>_marks.png, e.g. [4, 8]     box: (u0, v0, u1, v1) world units
        polygon: [(u, v), ...] world units                  color: '#rrggbb' -> pixels within `tol` of that colour
        """
        cv2 = _cv2()
        a = self._full_mask().copy()
        if regions is not None:
            ids = [regions] if isinstance(regions, int) else list(regions)
            missing = [i for i in ids if i not in self.regions]
            if missing:
                raise ValueError("view %r has no region %s (ids are 1..%d, see %s)" % (
                    self.name, missing, len(self.regions), self.rec.get("marks")))
            a &= np.isin(self._region_map(), ids)
        if box is not None:
            a &= RefMask(self, np.ones_like(a)).clip(*box).array
        if polygon is not None:
            pts = np.array([[self.x_of(u), self.y_of(v)] for u, v in polygon], np.float32)
            canvas = np.zeros(a.shape, np.uint8)
            cv2.fillPoly(canvas, [np.round(pts * 16).astype(np.int32)], 255, shift=4)
            a &= canvas > 0
        if color is not None:
            lab = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2LAB).astype(np.float32)
            c = cv2.cvtColor(np.array([[hex_to_bgr(color)]], np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
            d = np.sqrt((((lab - c) * np.array([100.0 / 255.0, 1.0, 1.0], np.float32)) ** 2).sum(2))
            a &= d <= tol
        return RefMask(self, a)

    def region(self, rid):
        """Region record: {'id', 'color', 'share', 'box': (u0, v0, u1, v1), 'pin': (u, v)}."""
        if rid not in self.regions:
            raise ValueError("view %r has no region %r" % (self.name, rid))
        return self.regions[rid]

    def spans(self, v):
        return self.mask().spans(v)

    def span(self, v):
        """(u_min, u_max) of the silhouette at height v."""
        return self.mask().span(v)

    def color(self, u, v, radius=0.01):
        """Average colour of the picture around (u, v) as '#rrggbb' (None outside the subject)."""
        x, y, r = int(round(self.x_of(u))), int(round(self.y_of(v))), max(1, int(round(radius * self.px_per_unit)))
        h, w = self._full_mask().shape
        x0, x1, y0, y1 = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
        sel = self._full_mask()[y0:y1, x0:x1]
        if not sel.any():
            return None
        c = self.image[y0:y1, x0:x1, :3][sel].mean(0)
        return "#%02x%02x%02x" % (int(c[2]), int(c[1]), int(c[0]))


class Ref:
    """A reference pack (made by `f3d ref`)."""

    def __init__(self, path):
        self.dir = os.path.abspath(path)
        jf = os.path.join(self.dir, "ref.json")
        if not os.path.isfile(jf):
            raise FileNotFoundError("no reference pack at %s (make one with: f3d ref <image> -o %s)" % (path, path))
        with open(jf) as f:
            self.doc = json.load(f)
        self.height = float(self.doc["height"])
        self.size = dict(self.doc.get("size", {}))
        self.palette = [c for c, _ in self.doc.get("palette", [])]
        self.views = {n: RefView(self, self.doc["views"][n]) for n in self.doc["order"]}

    def __getitem__(self, name):
        if name not in self.views:
            raise KeyError("the reference pack has no %r view (it has: %s)" % (name, ", ".join(self.views)))
        return self.views[name]

    def __contains__(self, name):
        return name in self.views

    def __iter__(self):
        return iter(self.views.values())

    @property
    def side(self):
        """The side view, whichever of 'right' / 'left' the pack has (None if neither)."""
        return self.views.get("right") or self.views.get("left")

    @property
    def ink(self):
        """The line art of the pictures: dict(color='#rrggbb', width=<world units>), or None when the subject
        is not drawn with outlines (a photograph, a render). k.outline() uses it."""
        if not hasattr(self, "_ink"):
            if "ink" in self.doc:
                self._ink = self.doc["ink"]
            else:                                   # packs made by older versions: measure it now
                from .. import refs
                self._ink = refs.pack_ink(self.doc, self.dir)
        return self._ink

    def __repr__(self):
        return "<Ref %s views=%s size=%s>" % (self.dir, list(self.views), self.size)


def ref(path=None):
    """Open a reference pack. With no path: the one passed to `f3d build --ref` (or the first one opened by name)."""
    p = path or _state["default"]
    if p is None:
        raise ValueError("no reference pack: pass a folder to k.ref('refs') or run the script with f3d build --ref refs")
    from . import core
    p = core.in_path(p)
    key = os.path.abspath(p)
    if key not in _state["cache"]:
        _state["cache"][key] = Ref(key)
    if _state["default"] is None:            # a pack opened by name becomes the one later calls use by default
        _state["default"] = key
    return _state["cache"][key]


def as_ref(r):
    if isinstance(r, Ref):
        return r
    return ref(r)
