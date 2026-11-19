"""Turn any model in the Blender scene into a Blockbench block model: voxelise it, then merge the voxels into as
few cuboids as possible. Colours come from the reference pictures (pixel art straight from the artwork) or from
the model's own materials."""
import numpy as np

from ..kit import core, raster
from .model import FACES, Model


def _surface_samples(objs, step):
    """Points on every triangle, at most `step` apart, with the material colour of their object."""
    from ..kit import colors as colmod
    dg = core.depsgraph()
    pts, cols = [], []
    for o in objs:
        v, t = core.world_tris(o, dg)
        if len(t) == 0:
            continue
        tri = v[t]
        vc = colmod.vertex_colors(o, len(v))                                # (n, 3) uint8 sRGB
        tc = vc[t].astype(np.float32)
        ext = np.linalg.norm(tri.max(1) - tri.min(1), axis=1)
        level = np.ceil(np.log2(np.maximum(ext / step, 1.0))).astype(np.int32).clip(0, 8)
        for lv in np.unique(level):
            sel = level == lv
            b = raster._bary_samples(int(2 ** lv))
            pts.append(np.einsum("sk,nkc->nsc", b, tri[sel]).reshape(-1, 3))
            cols.append(np.einsum("sk,nkc->nsc", b, tc[sel]).reshape(-1, 3))
    if not pts:
        raise ValueError("voxelize: the scene has no model geometry")
    return np.concatenate(pts), np.concatenate(cols)


def _boxes(occ):
    """Greedy decomposition of a boolean grid into axis-aligned boxes. -> [(x0, y0, z0, x1, y1, z1)] (exclusive)."""
    occ = occ.copy()
    nx, ny, nz = occ.shape
    out = []
    # start with the tallest columns so big interior blocks are claimed first
    for z in range(nz):
        for y in range(ny):
            row = occ[:, y, z]
            x = 0
            while x < nx:
                if not row[x]:
                    x += 1
                    continue
                x1 = x
                while x1 < nx and occ[x1, y, z]:
                    x1 += 1
                y1 = y + 1
                while y1 < ny and occ[x:x1, y1, z].all():
                    y1 += 1
                z1 = z + 1
                while z1 < nz and occ[x:x1, y:y1, z1].all():
                    z1 += 1
                occ[x:x1, y:y1, z:z1] = False
                out.append((x, y, z, x1, y1, z1))
                x = x1
    return out


def voxelize(objs=None, height=40, name="model", ref=None, colors=48, format="free", fill=True, vote=4,
             verbose=True):
    """Blender scene -> bb.Model made of merged voxels.

    height: voxels along the subject's height (24 = chunky, 40 = detailed, 64 = fine; cube count grows fast).
    ref: reference pack to take the colours from (recommended). Without it the model's own colours are used.
    colors: palette size for the texture (None = unlimited).   fill: make the inside solid.
    vote: how many of a voxel's eight octants must be solid for the voxel to exist (lower keeps thin parts such as
          strands and antennae, higher gives a leaner model).
    """
    from scipy import ndimage
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    bb = core.bbox(items)
    world_h = bb.max.z - min(0.0, bb.min.z)
    R = None
    if ref is not None:
        from ..kit import refkit
        R = refkit.as_ref(ref)
        world_h = R.height
    vs = world_h / float(height)                                           # voxel size in world units
    # occupancy at twice the resolution, then a vote per 2x2x2 block: a voxel counts when the solid really fills a
    # good part of it, so the model does not come out one voxel fatter than the source all around
    fine = vs / 2.0
    P, C = _surface_samples(items, fine * 0.45)
    fidx = np.floor(P / fine).astype(np.int64)
    lo = np.floor(fidx.min(0) / 2.0).astype(np.int64)                      # coarse-grid origin (keeps x = 0, z = 0 on voxel borders)
    fidx -= lo * 2
    fshape = tuple(int(v) for v in ((fidx.max(0) // 2 + 1) * 2))
    fflat = np.ravel_multi_index((fidx[:, 0], fidx[:, 1], fidx[:, 2]), fshape)
    fcount = np.bincount(fflat, minlength=int(np.prod(fshape)))
    focc = (fcount >= max(2, int(0.15 * np.median(fcount[fcount > 0])))).reshape(fshape)
    if fill:
        focc = ndimage.binary_fill_holes(focc)
    shape = tuple(v // 2 for v in fshape)
    votes = focc.reshape(shape[0], 2, shape[1], 2, shape[2], 2).sum(axis=(1, 3, 5))
    occ = votes >= vote
    idx = fidx // 2
    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]), shape)
    count = np.bincount(flat, minlength=int(np.prod(shape)))
    lab, n = ndimage.label(occ)
    if n > 1:                                                              # specks
        sizes = ndimage.sum(occ, lab, range(1, n + 1))
        keep = np.flatnonzero(sizes >= max(4, 0.002 * sizes.max())) + 1
        occ = np.isin(lab, keep)
    m = Model(name, height=height, format=format)
    root = m.bone(name, pivot=(0, 0, 0))
    boxes = _boxes(occ)
    pad = np.pad(occ, 1)
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(boxes):
        c = m.cube("v%04d" % i, lo=(x0 + lo[0], y0 + lo[1], z0 + lo[2]), hi=(x1 + lo[0], y1 + lo[1], z1 + lo[2]),
                   bone=root)
        # faces completely covered by neighbouring voxels need no texture
        px0, py0, pz0, px1, py1, pz1 = x0 + 1, y0 + 1, z0 + 1, x1 + 1, y1 + 1, z1 + 1
        cover = {"front": pad[px0:px1, py0 - 1, pz0:pz1], "back": pad[px0:px1, py1, pz0:pz1],
                 "left": pad[px0 - 1, py0:py1, pz0:pz1], "right": pad[px1, py0:py1, pz0:pz1],
                 "top": pad[px0:px1, py0:py1, pz1], "bottom": pad[px0:px1, py0:py1, pz0 - 1]}
        for f in FACES:
            if cover[f].all():
                c.faces[f].hidden = True
    if R is not None:
        m.paint_from_ref(R, colors=colors, cutout=False, verbose=verbose)     # voxels are solid: no see-through texels
    else:
        csum = np.zeros((int(np.prod(shape)), 3), np.float64)
        np.add.at(csum, flat, C.astype(np.float64) ** 2.2)
        mean = (csum / np.maximum(count, 1)[:, None]) ** (1 / 2.2)
        known = count.reshape(shape) > 0
        nearest = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        grid = mean.reshape(shape + (3,))[nearest[0], nearest[1], nearest[2]]
        for c in m.cubes:
            for f in FACES:
                face = c.faces[f]
                if face.hidden:
                    continue
                o, du, dv, nrm = c.face_frame(f)
                w, h = face.size
                ii, jj = np.meshgrid(np.arange(w), np.arange(h))
                centre = o + np.outer(ii.ravel() + 0.5, du) + np.outer(jj.ravel() + 0.5, dv) - nrm * 0.5
                vi = np.clip(np.floor(centre).astype(np.int64) - lo, 0, np.array(shape) - 1)
                face.px[:, :, :3] = np.clip(np.round(grid[vi[:, 0], vi[:, 1], vi[:, 2]]), 0, 255).astype(np.uint8).reshape(h, w, 3)
                face.painted = True
        if colors:
            from .paint import quantize
            faces = [f for c in m.cubes for f in c.faces.values() if not f.hidden]
            flatc = np.concatenate([f.px[:, :, :3].reshape(-1, 3) for f in faces])
            q = quantize(flatc, colors)
            k = 0
            for f in faces:
                nn = f.px.shape[0] * f.px.shape[1]
                f.px[:, :, :3] = q[k:k + nn].reshape(f.px.shape[0], f.px.shape[1], 3)
                k += nn
    if verbose:
        print("[f3d] voxelize: %d voxels -> %d cuboids (%d px tall, voxel = %.4g world units)" % (
            int(occ.sum()), len(m.cubes), height, vs))
    return m
