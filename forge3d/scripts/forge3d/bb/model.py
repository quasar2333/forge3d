"""Build Blockbench models in Python and write real .bbmodel files (Blockbench 4.x / 5.x project format).

    from forge3d import bb
    m = bb.Model("fox", height=24)                      # the subject is 24 px tall
    body = m.bone("body", pivot=(0, 0, 8))
    m.cube("torso", lo=(-4, -6, 6), hi=(4, 6, 14), bone=body, color="#d9772b")
    m.paint_from_ref("refs")                            # pixel-art textures taken from the reference pictures
    m.save("fox.bbmodel")

Axes are the same as everywhere else in forge3d, measured in Blockbench pixels (16 px = one Minecraft block):
X = the viewer's right in front view, Y = depth (the model faces -Y), Z = up, feet on z = 0.
Faces are named front, back, left (-X), right (+X), top, bottom.
"""
import base64
import json
import math
import os
import uuid as _uuid

import numpy as np

FORMATS = {"free": "Generic Model (any rotation, meshes allowed; exports glTF / OBJ / FBX)",
           "bedrock": "Bedrock Entity (bones + cubes, free rotation, one texture)",
           "bedrock_block": "Bedrock Block",
           "java_block": "Java Block/Item (rotation on one axis only, in steps of 22.5 degrees, size limits)",
           "modded_entity": "Modded Entity (Java, exports a Java class)",
           "optifine_entity": "OptiFine Entity"}
FACES = ("front", "back", "left", "right", "top", "bottom")
BB_FACE = {"front": "north", "back": "south", "left": "east", "right": "west", "top": "up", "bottom": "down"}
# kit world (x right, y back, z up) -> Blockbench (X east, Y up, Z south): a proper rotation
W2B = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# per face, in kit axes: outward normal, direction of the texture's +u (picture right) and +v (picture down)
FACE_AXES = {
    "front":  ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
    "back":   ((0, 1, 0), (-1, 0, 0), (0, 0, -1)),
    "left":   ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
    "right":  ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "top":    ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
    "bottom": ((0, 0, -1), (-1, 0, 0), (0, -1, 0)),
}


SAVED = []          # .bbmodel files written during this run (f3d build shows the last one)
PLAN = {}           # set by bb.plan() in a blockout script: the parts of this model are cuboid parts
PART = {"model": None}      # the Model of the part being built under `f3d part` (see bb.part)
PART_FILE = "part.bbpart"
_PANEL_SIDE = {"top": 0.21, "bottom": -0.27, "front": 0.04, "back": -0.10, "left": -0.11, "right": -0.09}


def _guid():
    return str(_uuid.uuid4())


def rot_matrix(rot):
    """XYZ degrees (applied X, then Y, then Z, about fixed axes) -> 3x3 matrix."""
    rx, ry, rz = (math.radians(a) for a in rot)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def euler_zyx(R):
    """3x3 matrix -> (x, y, z) degrees with R = Rz Ry Rx (three.js order 'ZYX', what Blockbench stores)."""
    sy = -R[2, 0]
    if abs(sy) < 0.999999:
        y = math.asin(max(-1.0, min(1.0, sy)))
        x = math.atan2(R[2, 1], R[2, 2])
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        y = math.copysign(math.pi / 2, sy)
        x = math.atan2(-R[1, 2], R[1, 1])
        z = 0.0
    return [round(math.degrees(a), 4) + 0.0 for a in (x, y, z)]


def to_bb_rot(rot):
    if rot is None or not any(rot):
        return [0, 0, 0]
    return euler_zyx(W2B @ rot_matrix(rot) @ W2B.T)


def from_bb_rot(rot):
    """Blockbench [x, y, z] degrees -> 3x3 matrix in kit axes."""
    return W2B.T @ rot_matrix(rot) @ W2B


def parse_color(c):
    """'#rrggbb' / '#rrggbbaa' / (r, g, b[, a]) 0-255 -> uint8 RGBA."""
    if isinstance(c, str):
        s = c.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        if len(s) not in (6, 8):
            raise ValueError("bad colour %r (use '#rrggbb')" % (c,))
        v = [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]
        return np.array(v + [255] * (4 - len(v)), np.uint8)
    v = list(c)
    return np.array([int(x) for x in v] + [255] * (4 - len(v)), np.uint8)


class Face:
    """One side of a cube: a small RGBA pixel grid `px` (rows top to bottom) that becomes part of the texture."""

    def __init__(self, cube, name, w, h):
        self.cube, self.name = cube, name
        self.px = np.zeros((h, w, 4), np.uint8)
        self.px[:, :, 3] = 255
        self.rect = None                                 # (x, y, w, h) in the atlas, set when packing
        self.painted = False
        self.hidden = False                              # fully covered by other cubes: gets no texture space

    @property
    def size(self):
        return self.px.shape[1], self.px.shape[0]

    def fill(self, color):
        self.px[:, :] = parse_color(color)
        self.painted = True
        return self

    def rect_fill(self, x0, y0, x1, y1, color):
        """Fill texels x0..x1-1, y0..y1-1 (origin top-left, y down)."""
        self.px[max(0, y0):y1, max(0, x0):x1] = parse_color(color)
        self.painted = True
        return self

    def pixel(self, x, y, color):
        if 0 <= y < self.px.shape[0] and 0 <= x < self.px.shape[1]:
            self.px[y, x] = parse_color(color)
        return self

    def rows(self, rows, palette):
        """Pixel art from text: rows of characters, palette = {char: colour}; '.' / ' ' leave the texel as is.
        The picture is centred on the face."""
        h, w = self.px.shape[:2]
        oy, ox = (h - len(rows)) // 2, (w - max(len(r) for r in rows)) // 2
        for j, row in enumerate(rows):
            for i, ch in enumerate(row):
                if ch in palette:
                    self.pixel(ox + i, oy + j, palette[ch])
        self.painted = True
        return self

    def gradient(self, top, bottom):
        a, b = parse_color(top).astype(np.float32), parse_color(bottom).astype(np.float32)
        h = self.px.shape[0]
        for j in range(h):
            t = j / max(1, h - 1)
            self.px[j, :] = np.round(a * (1 - t) + b * t).astype(np.uint8)
        self.painted = True
        return self

    def outline(self, color):
        c = parse_color(color)
        self.px[0, :] = self.px[-1, :] = c
        self.px[:, 0] = self.px[:, -1] = c
        return self


class Cube:
    def __init__(self, model, name, lo, hi, bone=None, rot=None, pivot=None, inflate=0.0):
        self.model, self.name = model, name
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        if np.any(self.hi < self.lo):
            self.lo, self.hi = np.minimum(self.lo, self.hi), np.maximum(self.lo, self.hi)
        self.bone = bone
        self.rot = tuple(rot) if rot is not None else (0.0, 0.0, 0.0)
        self.pivot = np.array(pivot, float) if pivot is not None else (self.lo + self.hi) / 2
        self.inflate = float(inflate)
        self.uuid = _guid()
        self.color = None
        self.cut_mask = None                      # RefMask: only this piece of the picture shows on the cube
        self.cut_flip = False
        d = model.density
        s = self.hi - self.lo
        dims = {"front": (s[0], s[2]), "back": (s[0], s[2]), "left": (s[1], s[2]), "right": (s[1], s[2]),
                "top": (s[0], s[1]), "bottom": (s[0], s[1])}
        self.faces = {f: Face(self, f, max(1, int(math.ceil(dims[f][0] * d - 1e-6))), max(1, int(math.ceil(dims[f][1] * d - 1e-6))))
                      for f in FACES}

    @property
    def size(self):
        return self.hi - self.lo

    @property
    def center(self):
        return (self.lo + self.hi) / 2

    def face(self, name):
        if name not in self.faces:
            raise ValueError("no face %r (use %s)" % (name, ", ".join(FACES)))
        return self.faces[name]

    def paint(self, color=None, **faces):
        """c.paint('#345') paints every face; c.paint(front='#fff', top='#ccc') paints single faces."""
        if color is not None:
            self.color = color
            for f in self.faces.values():
                f.fill(color)
        for k, v in faces.items():
            self.face(k).fill(v)
        return self

    def panel(self, color, style="panel", grain=2.0):
        """Paint the cube as a piece of hand-textured pixel art in one colour: every face shaded by the way it
        points (top light, underside dark), a lit edge up and left, a dark edge down and right, a little grain.
        This is what makes plain boxes read as armour plates, cloth or stone instead of flat colour.

        color: '#rrggbb' (read it off the picture: view.color(u, v)).
        style: 'panel' (a bevelled plate with an inner highlight; large faces get two rivets), 'ribbed'
               (horizontal ribs: vents, bellows, joints, tail segments), 'edge' (only the lit and dark edge:
               small parts), 'flat' (shading and grain only: skin, cloth, glow).
        Paint details on top afterwards with c.face(...).rect_fill / pixel / rows. Returns the cube."""
        import zlib
        if style not in ("panel", "ribbed", "edge", "flat"):
            raise ValueError("panel style must be 'panel', 'ribbed', 'edge' or 'flat'")
        base = parse_color(color)[:3].astype(float)
        lit, shadow, seam = np.minimum(base * 1.2 + 4, 255), base * 0.70, base * 0.49
        self.color = color
        stem = self.name.replace("left", "").replace("right", "")
        for name, f in self.faces.items():
            w, h = f.size
            y, x = np.mgrid[:h, :w]
            rng = np.random.default_rng(zlib.crc32((stem + name).encode()))
            noise = rng.integers(-1, 2, (h, w)) * grain
            gain = (0.35 - (y + 0.5) / h) * 0.25 + _PANEL_SIDE[name]
            col = base + (lit - shadow) * gain[..., None] + noise[..., None]
            edge = np.minimum.reduce([x, y, w - 1 - x, h - 1 - y])
            if style == "panel" and min(w, h) >= 5:
                col[(x == 0) | (y == 0)] = 0.32 * lit + 0.68 * base
                col[(y == h - 1) | (x == w - 1)] = 0.66 * shadow + 0.34 * seam
                if min(w, h) >= 9:
                    col[(y == 1) & (x >= 2) & (x < w - 2)] = 0.73 * lit + 0.27 * base
                    col[(x == 1) & (y >= 2) & (y < h - 2)] = 0.57 * lit + 0.43 * base
                    col[(y == h - 2) & (x >= 2) & (x < w - 2)] = 0.52 * shadow + 0.48 * base
                    if h >= 13 and w >= 10:
                        for px, py in ((2, 3), (w - 3, h - 4)):
                            col[py, px] = seam
                            col[py - 1, px] = 0.78 * lit + 0.22 * base
            elif style == "ribbed" and h >= 6:
                col[y % 4 == 0] = shadow
                col[y % 4 == 1] = 0.58 * lit + 0.42 * base
            elif style in ("panel", "edge") and min(w, h) >= 3:
                col[(x == 0) | (y == 0)] = 0.72 * lit + 0.28 * base
                col[(x == w - 1) | (y == h - 1)] = shadow
            f.px[:, :, :3] = np.round(np.clip(col, 0, 255)).astype(np.uint8)
            f.px[:, :, 3] = 255
            f.painted = True
        return self

    # -- geometry in kit world axes (pixels)
    def matrix(self):
        """(R, t): world = R @ local + t, including the cube's own rotation and every parent bone."""
        R = rot_matrix(self.rot)
        t = self.pivot - R @ self.pivot
        b = self.bone
        while b is not None:
            Rb = rot_matrix(b.rot)
            tb = b.pivot - Rb @ b.pivot
            R, t = Rb @ R, Rb @ t + tb
            b = b.parent
        return R, t

    def face_frame(self, name):
        """(origin, du, dv, normal) in world pixels: origin = top-left corner of the face's texture, du / dv =
        world step per texel to the right / down."""
        n, u, v = (np.array(a, float) for a in FACE_AXES[name])
        lo, hi = self.lo - self.inflate, self.hi + self.inflate
        c = (lo + hi) / 2
        half = (hi - lo) / 2
        corner = c + n * np.abs(n @ half) - u * np.abs(u @ half) - v * np.abs(v @ half)
        fw, fh = self.faces[name].size
        du = u * (2 * np.abs(u @ half) / fw)
        dv = v * (2 * np.abs(v @ half) / fh)
        R, t = self.matrix()
        return R @ corner + t, R @ du, R @ dv, R @ n

    def corners(self):
        """8 corners in world pixels (after rotations)."""
        lo, hi = self.lo - self.inflate, self.hi + self.inflate
        R, t = self.matrix()
        pts = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        return pts @ R.T + t


class Bone:
    def __init__(self, model, name, pivot=(0, 0, 0), parent=None, rot=None):
        self.model, self.name = model, name
        self.pivot = np.array(pivot, float)
        self.parent = parent
        self.parent_name = None                  # a bone of another part, joined up by bb.assemble
        self.rot = tuple(rot) if rot is not None else (0.0, 0.0, 0.0)
        self.uuid = _guid()
        self.children = []


class Model:
    """A Blockbench project under construction."""

    def __init__(self, name="model", height=32, format="free", density=1, snap=0.0):
        """height: how many pixels tall the subject is (drives the level of detail: 16-24 chunky Minecraft mob,
                   32-48 detailed character, 64+ fine voxel-art).
        format: free (default), bedrock, java_block, modded_entity ... see bb.FORMATS.
        density: texture texels per pixel of model size (1 = classic; 2 or 4 = finer textures on the same cubes).
        snap: round cube corners to this grid in pixels (1 or 0.5 keeps pixel art crisp; 0 = off)."""
        if format not in FORMATS:
            raise ValueError("unknown format %r (have: %s)" % (format, ", ".join(FORMATS)))
        self.name, self.height, self.format = name, float(height), format
        self.density, self.snap = int(density), float(snap)
        self.cubes, self.bones = [], []
        self.atlas = None
        self.root = None                         # set for a part (bb.part): the bone everything of it hangs from

    # -- building
    def bone(self, name, pivot=(0, 0, 0), parent=None, rot=None):
        """A bone (Blockbench group): cubes and child bones inside it rotate about `pivot`."""
        other = None
        if isinstance(parent, str) and self.root is not None and not any(x.name == parent for x in self.bones):
            other, parent = parent, None         # a bone another part makes: hang on our root until assembled
        if parent is None and self.root is not None:
            parent = self.root
        b = Bone(self, name, pivot, self._bone(parent), rot)
        b.parent_name = other
        if b.parent is not None:
            b.parent.children.append(b)
        self.bones.append(b)
        return b

    def _bone(self, b):
        if b is None or isinstance(b, Bone):
            return b
        for x in self.bones:
            if x.name == b:
                return x
        raise ValueError("no bone named %r" % (b,))

    def _snap(self, v):
        v = np.array(v, float)
        return np.round(v / self.snap) * self.snap if self.snap else v

    def cube(self, name, lo=None, hi=None, size=None, at=None, base=None, bone=None, rot=None, pivot=None,
             inflate=0.0, color=None, cutout=None):
        """Add a cuboid. Give opposite corners lo=(x0, y0, z0), hi=(x1, y1, z1); or size=(w, d, h) with at=centre
        or base=bottom-centre.
        rot: XYZ degrees about `pivot` (default: the cube's centre).  inflate: grow every side by this much without
        changing the texture layout (overlays such as hair over a head, clothes over a body).
        color: flat colour for all faces (paint details afterwards, or call m.paint_from_ref()).
        cutout: a RefMask, e.g. k.ref()["front"].mask(regions=[7, 11]): when the model is painted from the
            reference, texels outside that mask become transparent and the facing side shows that piece of the
            picture. Use it on thin plates for parts with a drawn outline: fins, ears, bows, tails, hair locks."""
        if lo is None or hi is None:
            if size is None or (at is None and base is None):
                raise ValueError("cube %r needs lo= and hi=, or size= with at= or base=" % name)
            s = np.array(size, float)
            c = np.array(at, float) if at is not None else np.array(base, float) + np.array([0, 0, s[2] / 2])
            lo, hi = c - s / 2, c + s / 2
        lo, hi = self._snap(lo), self._snap(hi)
        if np.any(hi - lo <= 0):
            raise ValueError("cube %r has no thickness on some axis: lo=%s hi=%s" % (name, tuple(lo), tuple(hi)))
        c = Cube(self, name, lo, hi, self._bone(bone) if bone is not None else self.root, rot, pivot, inflate)
        c.cut_mask = cutout
        if color is not None:
            c.paint(color)
        self.cubes.append(c)
        return c

    def beam(self, name, a, b, width, depth=None, bone=None, roll=0.0, forward=(0, 1, 0), inflate=0.0, color=None,
             cutout=None):
        """A cuboid running from point a to point b: limbs, struts, wing spars, blades, tail and finger segments,
        anything slanted. width x depth is its cross-section (depth defaults to width); its long axis is a -> b.
        forward: the direction its depth side faces as nearly as it can (default +Y, the model's back);
        roll: extra turn about its own axis in degrees. Returns the cube (paint it with .panel / .paint)."""
        a, b = np.array(a, float), np.array(b, float)
        axis = b - a
        length = float(np.linalg.norm(axis))
        if length < 1e-6:
            raise ValueError("beam %r: a and b are the same point" % name)
        z = axis / length
        x = np.cross(np.array(forward, float), z)
        if np.linalg.norm(x) < 1e-5:
            x = np.cross(np.eye(3)[int(np.abs(z).argmin())], z)
        x /= np.linalg.norm(x)
        R = np.column_stack((x, np.cross(z, x), z))
        if roll:
            R = R @ rot_matrix((0, 0, roll))
        s = np.array([float(width), float(width if depth is None else depth), length])
        mid = (a + b) / 2
        c = Cube(self, name, mid - s / 2, mid + s / 2, self._bone(bone) if bone is not None else self.root,
                 euler_zyx(R), mid, inflate)
        c.cut_mask = cutout
        if color is not None:
            c.paint(color)
        self.cubes.append(c)
        return c

    def mirror(self, item, name=None):
        """Mirrored copy of a cube (or a bone with everything in it) across x = 0. Names swap left/right, _l/_r."""
        def swap(n):
            for a, b in (("left", "right"), ("Left", "Right"), ("_l", "_r"), ("_L", "_R"), (".l", ".r"), (".L", ".R")):
                if a in n:
                    return n.replace(a, b)
                if b in n:
                    return n.replace(b, a)
            return n + "_mirror"
        flip = np.array([-1.0, 1.0, 1.0])
        if isinstance(item, Cube):
            c = item
            new = Cube(self, name or swap(c.name), (-c.hi[0], c.lo[1], c.lo[2]), (-c.lo[0], c.hi[1], c.hi[2]), c.bone,
                       (c.rot[0], -c.rot[1], -c.rot[2]), c.pivot * flip, c.inflate)
            new.color = c.color
            new.cut_mask, new.cut_flip = c.cut_mask, not c.cut_flip
            for f in FACES:
                src = c.faces[{"left": "right", "right": "left"}.get(f, f)]
                new.faces[f].px = src.px[:, ::-1].copy()
                new.faces[f].painted = src.painted
            self.cubes.append(new)
            return new
        b = item
        nb = self.bone(name or swap(b.name), b.pivot * flip, b.parent, (b.rot[0], -b.rot[1], -b.rot[2]))
        for c in [c for c in self.cubes if c.bone is b]:
            self.mirror(c).bone = nb
        for ch in list(b.children):
            twin = self.mirror(ch)
            if twin.parent is not None and twin in twin.parent.children:
                twin.parent.children.remove(twin)
            twin.parent = nb
            nb.children.append(twin)
        return nb

    # -- queries
    def bounds(self):
        pts = np.concatenate([c.corners() for c in self.cubes])
        return pts.min(0), pts.max(0)

    def triangles(self):
        """All cube faces as triangles in world pixels: (N, 3, 3)."""
        out = []
        for c in self.cubes:
            for f in FACES:
                if c.faces[f].hidden:
                    continue
                o, du, dv, _n = c.face_frame(f)
                w, h = c.faces[f].size
                a, b, cc, d = o, o + du * w, o + du * w + dv * h, o + dv * h
                out.append([a, b, cc])
                out.append([a, cc, d])
        return np.array(out, np.float32)

    # -- texture
    def pack(self):
        """Lay every face out in one texture atlas. Returns the RGBA image (h, w, 4)."""
        every = [f for c in self.cubes for f in c.faces.values()]
        faces = [f for f in every if not f.hidden] or every[:1]
        order = sorted(faces, key=lambda f: (-f.size[1], -f.size[0]))
        area = sum(f.size[0] * f.size[1] for f in faces) + 1
        side = 16
        while side * side < area or side < max(max(f.size) for f in faces):
            side *= 2
        while True:
            x, y, row = 1, 0, 1                          # texel (0, 0) is shared by all hidden faces
            ok = True
            for f in order:
                w, h = f.size
                if x + w > side:
                    x, y, row = 0, y + row, 0
                if y + h > side:
                    ok = False
                    break
                f.rect = (x, y, w, h)
                x += w
                row = max(row, h)
            if ok:
                break
            side *= 2
            if side > 8192:
                raise ValueError("the texture would exceed 8192 px: lower density or the number / size of cubes")
        img = np.zeros((side, side, 4), np.uint8)
        for f in faces:
            x, y, w, h = f.rect
            img[y:y + h, x:x + w] = f.px
        # faces buried inside the model share texel (0, 0): give it the model's average colour, never a stray one
        img[0, 0] = np.round(np.concatenate([f.px.reshape(-1, 4) for f in faces]).mean(0)).astype(np.uint8)
        img[0, 0, 3] = 255
        for f in every:
            if f.hidden and f not in faces:
                f.rect = (0, 0, 1, 1)
        self.atlas = img
        return img

    def paint_from_ref(self, ref, **kw):
        """Paint every face from the reference pictures (pixel art straight from the artwork). See bb.paint."""
        from . import paint
        return paint.paint_from_ref(self, ref, **kw)

    def shade(self, **kw):
        from . import paint
        return paint.shade(self, **kw)

    # -- output
    def to_json(self):
        import cv2
        img = self.pack()
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        side = img.shape[0]
        tex_uuid = _guid()
        box_uv = False
        elements = []
        for c in self.cubes:
            lo_b, hi_b = W2B @ c.lo, W2B @ c.hi
            frm, to = np.minimum(lo_b, hi_b), np.maximum(lo_b, hi_b)
            faces = {}
            for f in FACES:
                x, y, w, h = c.faces[f].rect
                faces[BB_FACE[f]] = {"uv": [x, y, x + w, y + h], "texture": 0}
            el = {"name": c.name, "box_uv": box_uv, "render_order": "default", "locked": False, "export": True,
                  "allow_mirror_modeling": True, "from": [round(float(v), 4) for v in frm],
                  "to": [round(float(v), 4) for v in to], "autouv": 0, "color": 0,
                  "origin": [round(float(v), 4) for v in W2B @ c.pivot], "faces": faces, "type": "cube", "uuid": c.uuid}
            if c.inflate:
                el["inflate"] = c.inflate
            r = to_bb_rot(c.rot)
            if any(abs(a) > 1e-6 for a in r):
                el["rotation"] = r
            elements.append(el)
        groups = []
        for b in self.bones:
            groups.append({"name": b.name, "uuid": b.uuid, "export": True, "locked": False, "selected": False,
                           "visibility": True, "origin": [round(float(v), 4) for v in W2B @ b.pivot],
                           "rotation": to_bb_rot(b.rot), "color": 0, "children": [], "reset": False, "shade": True,
                           "mirror_uv": False, "autouv": 0, "isOpen": True, "primary_selected": False})

        def node(b):
            g = dict(next(g for g in groups if g["uuid"] == b.uuid))      # inline copy: readable by Blockbench 4.x too
            g["children"] = [node(ch) for ch in b.children] + [c.uuid for c in self.cubes if c.bone is b]
            return g
        outliner = [node(b) for b in self.bones if b.parent is None] + [c.uuid for c in self.cubes if c.bone is None]
        tex = {"name": self.name + ".png", "path": "", "folder": "", "namespace": "", "id": "0", "group": "",
               "width": side, "height": side, "uv_width": side, "uv_height": side, "particle": False,
               "use_as_default": False, "layers_enabled": False, "sync_to_project": "", "render_mode": "default",
               "render_sides": "auto", "pbr_channel": "color", "frame_time": 1, "frame_order_type": "loop",
               "frame_order": "", "frame_interpolate": False, "visible": True, "internal": True, "saved": False,
               "uuid": tex_uuid, "source": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()}
        return {"meta": {"format_version": "5.0", "model_format": self.format, "box_uv": box_uv},
                "name": self.name, "model_identifier": "", "visible_box": [1, 1, 0], "variable_placeholders": "",
                "variable_placeholder_buttons": [], "timeline_setups": [],
                "unhandled_root_fields": {"forge3d": {"height_px": self.height, "density": self.density,
                                                      "cutout": bool(any((f.px[:, :, 3] < 128).any() for c in self.cubes
                                                                         for f in c.faces.values() if not f.hidden))}},
                "resolution": {"width": side, "height": side}, "elements": elements, "groups": groups,
                "outliner": outliner, "textures": [tex]}

    def save(self, path, texture=True):
        """Write <path>.bbmodel (and the texture as a .png beside it). Returns the path."""
        from ..kit import core
        path = core.out_path(path if path.endswith(".bbmodel") else path + ".bbmodel")
        doc = self.to_json()
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
        if texture:
            import cv2
            cv2.imwrite(os.path.splitext(path)[0] + ".png", cv2.cvtColor(self.atlas, cv2.COLOR_RGBA2BGRA))
        print("[f3d] saved %s: %d cubes, %d bones, texture %dx%d" % (path, len(self.cubes), len(self.bones),
                                                                    self.atlas.shape[1], self.atlas.shape[0]))
        SAVED.append(path)
        return path


# ---------------------------------------------------------------- a cuboid model built part by part
def plan(height=64, format="free", density=1, snap=0.0):
    """In a blockout script: this model is a Blockbench cuboid model, so every slot is filled by a cuboid part.

        from forge3d import bb, kit as k
        bb.plan(height=80, format="bedrock", density=2)        # 80 px tall, texture at 2 texels per pixel
        k.slot("chassis", k.box(...), note="...")

    `f3d blockout` then writes cuboid work orders: each part script builds its share with `m = bb.part()`,
    and the final script joins them with `m = bb.assemble()`; `m.save("out/name.bbmodel")`.
    height: pixels tall (16 px = one Minecraft block). 32-48 a mob, 64-96 a detailed boss or statue.
    format, density, snap: as bb.Model."""
    if format not in FORMATS:
        raise ValueError("unknown format %r (have: %s)" % (format, ", ".join(FORMATS)))
    PLAN.clear()
    PLAN.update(height=float(height), format=format, density=int(density), snap=float(snap))


def part(pivot=None):
    """In a part script of a cuboid model (run by `f3d part <name>`): this part's share of the shared Blockbench
    model. Add bones and cubes to it; do not save it, `f3d part` does and shows it in check.png.

    m.root      the part's own bone (named after the slot); bones and cubes you add hang from it by default.
                pivot= sets its joint (default: where the slot meets what it sits on).
    m.lo, m.hi  the slot's box in pixels;  m.height  pixels per subject height (a point (u, v) of the front
                picture is x = u * m.height, z = v * m.height).
    A bone may name a bone of another part as its parent (m.bone("pauldron_left", parent="upper_arm_left")):
    it is joined up when the model is assembled."""
    from ..kit import partkit, refkit
    act = partkit._state["active"]
    if act is None:
        raise RuntimeError("bb.part() is the Blockbench model of the part being built and only exists under "
                           "`f3d part <name>`. For a model written as one script use m = bb.Model(...)")
    if PART["model"] is not None:
        return PART["model"]
    doc = partkit.load_plan(act["dir"])
    cfg = doc.get("bb")
    if not cfg:
        raise RuntimeError("this blockout is not a cuboid model: put bb.plan(height=...) in the blockout script and "
                           "run `f3d blockout` again")
    try:
        world_h = refkit.ref().height
    except Exception:  # noqa: BLE001 - no reference pack in use
        world_h = 1.0
    m = Model(act["name"], height=cfg["height"], format=cfg["format"], density=cfg["density"], snap=cfg.get("snap", 0))
    rec = doc["slots"][act["name"]]
    m.scale = cfg["height"] / world_h
    m.lo, m.hi = np.array(rec["lo"], float) * m.scale, np.array(rec["hi"], float) * m.scale
    if pivot is None:
        if rec.get("on") and rec["on"][0] in doc["slots"]:
            o = doc["slots"][rec["on"][0]]
            pivot = np.clip((np.array(o["lo"], float) + np.array(o["hi"], float)) / 2 * m.scale, m.lo, m.hi)
        else:
            pivot = ((m.lo[0] + m.hi[0]) / 2, (m.lo[1] + m.hi[1]) / 2, m.lo[2])
    m.root = m.bone(act["name"], pivot=pivot)
    PART["model"] = m
    return m


def assemble(parts="parts", name=None, verbose=True):
    """Join the cuboid parts of a model into one Blockbench model with one bone tree and one texture:

        from forge3d import bb
        m = bb.assemble()
        m.save("out/name.bbmodel")

    Every part's bone hangs from the bone of the part it sits on (`on=` in the blockout), the rest from "root".
    Parts not built yet are left out and named. Run it with `f3d build model.py --ref refs`."""
    import pickle
    from ..kit import partkit
    d = partkit.find_parts_dir(parts)
    doc = partkit.load_plan(d)
    cfg = doc.get("bb")
    if not cfg:
        raise RuntimeError("this blockout is not a cuboid model (no bb.plan(...) in the blockout script); its parts "
                           "are joined with k.assemble()")
    m = Model(name or "model", height=cfg["height"], format=cfg["format"], density=cfg["density"],
              snap=cfg.get("snap", 0))
    root = m.bone("root", pivot=(0, 0, 0))
    roots, missing = {}, []
    for n in doc["order"]:
        f = os.path.join(d, n, PART_FILE)
        if not os.path.isfile(f):
            missing.append(n)
            continue
        with open(f, "rb") as fh:
            pm = pickle.load(fh)
        for c in pm.cubes:
            c.model = m
        for b in pm.bones:
            b.model = m
        m.cubes += pm.cubes
        m.bones += pm.bones
        roots[n] = pm.root
    for n, r in roots.items():
        on = [x for x in doc["slots"][n]["on"] if x in roots]
        r.parent = roots[on[0]] if on else root
        r.parent.children.append(r)
    seen = {}
    for b in m.bones:                              # bone names are unique in Blockbench entity formats
        if b.name in seen:
            seen[b.name] += 1
            b.name = "%s_%d" % (b.name, seen[b.name])
        else:
            seen[b.name] = 1
    by_name = {b.name: b for b in m.bones}
    for b in m.bones:
        p = by_name.get(b.parent_name) if b.parent_name else None
        if p is not None and p is not b:
            if b.parent is not None and b in b.parent.children:
                b.parent.children.remove(b)
            b.parent = p
            p.children.append(b)
    if verbose:
        print("[f3d] bb.assemble: %d part(s), %d cubes, %d bones%s" % (
            len(roots), len(m.cubes), len(m.bones),
            "" if not missing else "; not built yet and left out: " + ", ".join(missing)))
    return m
