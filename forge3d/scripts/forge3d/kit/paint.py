"""Projection painting: colour the model with the reference pictures themselves.

Every reference view is projected straight onto the surfaces that face it and can see its camera; where views
overlap the one looking most squarely at the surface wins. Seen from a reference camera the model then shows
that reference picture. Three things keep the other angles clean:
- a picture is first tidied against the geometry it lands on (register.clean_picture): colours of a drawn part
  that spill over the edge of the surface patch behind or beside it are replaced by that patch's own colours;
- seen at a slant a surface gets the picture's flat-colour twin, never its dragged-out ink lines;
- what no view can see takes the flat colour of the nearest painted surface, measured along the surface with
  folds counting as far away, so a hair cap is filled from the hair around it and not from the ribbon on it.
"""
import math
import os

import bmesh
import bpy
import numpy as np

from . import core, raster, refkit, register

FILL_ATTR = "f3d_fill"
SEEN_ATTR = "f3d_seen"
FINAL_NODE = "f3d_final_colour"
SHADING = {"soft": 0.5, "lit": 0.0, "unlit": 1.0}
# Who paints a surface that several pictures can see. Drawings of one subject from different sides never agree in
# 3D (a face drawn from the front and again in profile puts the eye in two places), so views are not averaged:
# the first view in this list that sees a surface reasonably head-on owns it, the next ones get what is left.
OWNERSHIP = ("front", "back", "right", "left", "right_mirror", "left_mirror", "top", "bottom")


def _mesh_objs(objs):
    return [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]


def densify(obj, max_edge, budget=600000):
    """Split edges longer than max_edge (world units) so per-vertex data can follow occlusion boundaries.
    Stops before the mesh would exceed `budget` faces. Returns False when the budget cut it short."""
    me = obj.data
    s = obj.matrix_world.to_scale()
    scale = max(abs(s.x), abs(s.y), abs(s.z), 1e-9)
    complete = True
    for _ in range(6):
        bm = bmesh.new()
        bm.from_mesh(me)
        long_edges = [e for e in bm.edges if e.calc_length() * scale > max_edge]
        if not long_edges:
            bm.free()
            break
        cuts = int(min(8, max(1, math.ceil(max(e.calc_length() for e in long_edges) * scale / max_edge) - 1)))
        # each cut edge adds roughly 2 * cuts faces
        if len(bm.faces) + 2 * cuts * len(long_edges) > budget:
            room = max(0, budget - len(bm.faces)) // (2 * cuts)
            long_edges = sorted(long_edges, key=lambda e: -e.calc_length())[:room]
            complete = False
            if not long_edges:
                bm.free()
                break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=cuts, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
        bm.to_mesh(me)
        bm.free()
        if not complete:
            break
    me.update()
    return complete


def _world_geometry(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    nor = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("normal", nor)
    m = np.array(obj.matrix_world, dtype=np.float64)
    P = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    nm = np.linalg.inv(m[:3, :3]).T
    N = nor.reshape(-1, 3).astype(np.float64) @ nm.T
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    return P, N


def _adjacency(me):
    from scipy.sparse import coo_matrix
    n = len(me.vertices)
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    data = np.ones(len(e) * 2, np.float32)
    return coo_matrix((data, (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()


def _sample(img, x, y):
    """Bilinear sample of an (h, w, 3) uint8 image at continuous pixel coordinates. -> float32 (n, 3) in 0..1."""
    h, w = img.shape[:2]
    x, y = np.clip(x - 0.5, 0, w - 1.001), np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    a, b = img[y0, x0].astype(np.float32), img[y0, x0 + 1].astype(np.float32)
    c, d = img[y0 + 1, x0].astype(np.float32), img[y0 + 1, x0 + 1].astype(np.float32)
    return ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy) / 255.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _fill_unseen(me, P, normals, colors, known, height):
    """Unseen vertices take the colour of the nearest painted vertex, nearest along the surface, where crossing
    an inward fold costs as much as a long detour (parts meet in folds). The result is then evened out inside
    each colour area, not across their borders. colors: (n, 3) sRGB flat colours, valid where `known`."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    if known.all() or not known.any():
        return colors
    n = len(known)
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    d = P[e[:, 1]] - P[e[:, 0]]
    length = np.linalg.norm(d, axis=1)
    n0, n1 = normals[e[:, 0]], normals[e[:, 1]]
    turn = np.arccos(np.clip((n0 * n1).sum(1), -1.0, 1.0))
    inward = ((n1 - n0) * d).sum(1) < 0
    fold = turn * np.where(inward, 1.0, 0.1)
    cost = np.maximum(length + 0.12 * height * fold, 1e-9)
    G = coo_matrix((cost, (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    # only broad colour areas spread: a ribbon, an outline or the rim of an area is no evidence for what the
    # hidden surface next to it looks like. Sources are painted vertices at least 1.5 % of the height away from
    # any colour border.
    both = known[e[:, 0]] & known[e[:, 1]]
    border = both & (np.abs(colors[e[:, 0]] - colors[e[:, 1]]).max(1) > 0.1)
    sources = known
    if border.any():
        rim = np.zeros(n, bool)
        rim[e[border].ravel()] = True
        Gk = coo_matrix((length[both], (e[both, 0], e[both, 1])), shape=(n, n)).tocsr()
        away = dijkstra(Gk, directed=False, indices=np.flatnonzero(rim), min_only=True)
        broad = known & (away > 0.015 * height)
        if broad.sum() >= 0.2 * known.sum():
            sources = broad
    _dist, _pred, src = dijkstra(G, directed=False, indices=np.flatnonzero(sources), min_only=True, return_predecessors=True)
    out = colors.copy()
    U = ~known
    reach = U & (src >= 0)
    out[reach] = colors[src[reach]]
    lost = U & (src < 0)                                    # loose pieces no painted vertex connects to
    if lost.any():
        from scipy.spatial import cKDTree
        K = np.flatnonzero(sources)
        out[lost] = colors[K[cKDTree(P[K]).query(P[lost])[1]]]
    # even out inside each colour area: neighbours of a clearly different colour, or across a fold, hardly count
    dc = np.linalg.norm(out[e[:, 0]] - out[e[:, 1]], axis=1)
    wgt = np.exp(-(dc / 0.07) ** 2) * np.exp(-3.0 * fold) + 1e-4
    W = coo_matrix((np.concatenate([wgt, wgt]), (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
                   shape=(n, n)).tocsr()
    rows = np.flatnonzero(U)
    Wu = W[rows]
    den = np.maximum(np.asarray(Wu.sum(1)).ravel(), 1e-9)[:, None]
    for _ in range(12):
        out[rows] = 0.5 * out[rows] + 0.5 * (Wu @ out) / den
    # colour areas meet along mesh edges: two plain passes turn that sawtooth into a narrow soft edge
    Au = (W > 0).astype(np.float64)[rows]
    deg = np.maximum(np.asarray(Au.sum(1)).ravel(), 1.0)[:, None]
    for _ in range(2):
        out[rows] = 0.5 * out[rows] + 0.5 * (Au @ out) / deg
    return np.clip(out, 0.0, 1.0)


def _scraps(me, B, conf, limit):
    """Vertices in small islands (fewer than `limit` connected vertices) painted by any view but the first,
    which is the leading picture and keeps whatever it sees."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nv = B.shape[1]
    owner = np.where((conf.max(0) > 0.5) & (B.sum(0) > 1e-6), B.argmax(0), -1)
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    same = (owner[e[:, 0]] == owner[e[:, 1]]) & (owner[e[:, 0]] >= 0)
    g = coo_matrix((np.ones(int(same.sum()), np.int8), (e[same, 0], e[same, 1])), shape=(nv, nv))
    _cnt, lab = connected_components(g, directed=False)
    size = np.bincount(lab)
    return (owner >= 1) & (size[lab] < limit)


def _image(name, bgr):
    """A packed Blender image from a BGR uint8 array."""
    h, w = bgr.shape[:2]
    img = bpy.data.images.new(name, w, h, alpha=False)
    px = np.ones((h, w, 4), np.float32)
    px[:, :, :3] = bgr[::-1, :, ::-1].astype(np.float32) / 255.0
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def _material(sources, shading, name, positional=False):
    key = name or "f3d_painted_%s" % "_".join(src["key"] for src in sources)
    old = bpy.data.materials.get(key)
    if old is not None and old.users == 0:
        bpy.data.materials.remove(old)
    m = bpy.data.materials.new(key)             # a name still in use on other objects gets a numbered twin
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    N, link = nt.nodes.new, nt.links.new
    sum_c = None
    geo = N("ShaderNodeNewGeometry") if positional else None
    for src in sources:
        if positional:
            # the picture's coordinates straight from the surface position: needs no UV layer (a mesh has only
            # eight), but holds only while the object stays where it is, so it is baked right away
            v_, f_ = src["view"], src["view"].frame
            axes = (v_.u_vec * (np.array([-1.0, 1.0, 1.0]) if src["flip"] else 1.0), v_.v_vec)
            comp = N("ShaderNodeCombineXYZ")
            for ci, (axis, lo_, hi_) in enumerate(zip(axes, (f_[0], f_[1]), (f_[2], f_[3]))):
                dot = N("ShaderNodeVectorMath")
                dot.operation = "DOT_PRODUCT"
                link(geo.outputs["Position"], dot.inputs[0])
                dot.inputs[1].default_value = (float(axis[0]), float(axis[1]), float(axis[2]))
                mr = N("ShaderNodeMapRange")
                mr.clamp = False
                mr.inputs["From Min"].default_value = float(lo_)
                mr.inputs["From Max"].default_value = float(hi_)
                link(dot.outputs["Value"], mr.inputs["Value"])
                link(mr.outputs["Result"], comp.inputs[ci])
            uv_out = comp.outputs["Vector"]
        else:
            uv = N("ShaderNodeUVMap")
            uv.uv_map = "f3d_" + src["key"]
            uv_out = uv.outputs["UV"]
        tex = N("ShaderNodeTexImage")
        tex.image = _image("%s_%s" % (key, src["key"]), src["tex"])
        tex.extension = "EXTEND"
        tex.interpolation = "Cubic"
        link(uv_out, tex.inputs["Vector"])
        at = N("ShaderNodeAttribute")
        at.attribute_name = "f3d_w_" + src["key"]
        # seen at a slant, a picture's ink lines smear into streaks: there use its flat-colour twin
        flat = N("ShaderNodeTexImage")
        flat.image = _image("%s_%s_flat" % (key, src["key"]), src["flat"])
        flat.extension = "EXTEND"
        flat.interpolation = "Cubic"
        link(uv_out, flat.inputs["Vector"])
        sh = N("ShaderNodeAttribute")
        sh.attribute_name = "f3d_s_" + src["key"]
        pm = N("ShaderNodeMix")
        pm.data_type = "RGBA"
        link(sh.outputs["Fac"], pm.inputs["Factor"])
        link(flat.outputs["Color"], pm.inputs["A"])
        link(tex.outputs["Color"], pm.inputs["B"])
        picture = pm.outputs["Result"]
        sc = N("ShaderNodeVectorMath")
        sc.operation = "SCALE"
        link(picture, sc.inputs[0])
        link(at.outputs["Fac"], sc.inputs["Scale"])
        if sum_c is None:
            sum_c = sc.outputs["Vector"]
        else:
            add = N("ShaderNodeVectorMath")
            add.operation = "ADD"
            link(sum_c, add.inputs[0])
            link(sc.outputs["Vector"], add.inputs[1])
            sum_c = add.outputs["Vector"]
    fill = N("ShaderNodeAttribute")
    fill.attribute_name = FILL_ATTR
    seen = N("ShaderNodeAttribute")
    seen.attribute_name = SEEN_ATTR
    mix = N("ShaderNodeMix")
    mix.name = FINAL_NODE                     # the bake reads the finished colour from this node
    mix.data_type = "RGBA"
    link(seen.outputs["Fac"], mix.inputs["Factor"])
    link(fill.outputs["Color"], mix.inputs["A"])
    link(sum_c, mix.inputs["B"])
    color = mix.outputs["Result"]
    bsdf = N("ShaderNodeBsdfPrincipled")
    link(color, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.1
            break
    e = SHADING.get(shading, shading)
    if isinstance(e, bool) or not isinstance(e, (int, float)):
        raise ValueError("shading must be 'soft', 'lit', 'unlit' or a number 0..1")
    if e <= 0:
        link(bsdf.outputs[0], out.inputs["Surface"])
    else:
        em = N("ShaderNodeEmission")
        link(color, em.inputs["Color"])
        ms = N("ShaderNodeMixShader")
        ms.inputs["Fac"].default_value = float(e)
        link(bsdf.outputs[0], ms.inputs[1])
        link(em.outputs[0], ms.inputs[2])
        link(ms.outputs[0], out.inputs["Surface"])
    m["f3d_painted"] = True
    m["f3d_emit"] = float(e)
    m.diffuse_color = (0.6, 0.6, 0.65, 1.0)
    return m


def _sources(R, views, mirror):
    """Projection sources: the pack's views, plus the missing side as a mirror image of the one that exists."""
    names = [n for n in R.views if (views is None and (n in ("front", "back", "left", "right", "top", "bottom")
                                                        or n.startswith("az"))) or (views is not None and n in views)]
    out = [dict(key=n, view=R[n], flip=False, d=R[n].toward_camera) for n in names]
    if mirror:
        for have, want in (("right", "left"), ("left", "right")):
            if have in names and want not in R.views:
                d = R[have].toward_camera * np.array([-1.0, 1.0, 1.0])
                out.append(dict(key=want + "_mirror", view=R[have], flip=True, d=d))
    rank = {k: i for i, k in enumerate(OWNERSHIP)}
    out.sort(key=lambda src: rank.get(src["key"], len(rank)))
    return out


def _face_rules(sources):
    """A drawn face belongs to one picture. Seen from the side an eye is drawn where no front view puts it, so a
    head painted from both wears two pairs. When the front picture shows eyes: everything seen through its
    face is painted by the front picture alone, and eyes drawn in the other pictures are left out.
    -> (zones, erase, note) for paint_from_ref, from the faces and eyes `f3d ref` found in the pack."""
    lead = next((x["view"] for x in sources if x["key"] == "front" and x["view"].eyes), None)
    if lead is None:
        return [], [], None
    eyes = lead.eyes

    def holds(f, e):
        return f[0] < (e[0] + e[2]) / 2 < f[2] and f[1] < (e[1] + e[3]) / 2 < f[3]

    faces = [f for f in lead.faces if any(holds(f, e) for e in eyes)]
    loose = [e for e in eyes if not any(holds(f, e) for f in faces)]
    if loose:
        gw = 0.6 * max(e[2] - e[0] for e in loose)
        gh = 0.8 * max(e[3] - e[1] for e in loose)
        faces.append((min(e[0] for e in loose) - gw, min(e[1] for e in loose) - gh,
                      max(e[2] for e in loose) + gw, max(e[3] for e in loose) + gh))
    zones = [(lead.mask(box=f), "front") for f in faces]
    erase, dropped, done = [], [], set()
    for x in sources:
        v = x["view"]
        if v is lead or id(v) in done:
            continue
        done.add(id(v))
        for u0, v0, u1, v1 in v.eyes:
            gw, gh = 0.35 * (u1 - u0), v1 - v0
            erase.append(v.mask(box=(u0 - gw, v0 - 0.3 * gh, u1 + gw, v1 + 0.7 * gh)))
        if v.eyes:
            dropped.append(v.name)
    note = "the face (%d eye%s in the front picture) is painted by the front picture alone" % (
        len(eyes), "" if len(eyes) == 1 else "s")
    if dropped:
        note += "; the eye drawn in the %s picture was left out (it would be a second pair)" % " / ".join(dropped)
    return zones, erase, note


def _doubled(src, D, ppu, sc, base, subject, leaders, height, diag, veto):
    """Features a side picture draws on surfaces where a leading picture (front, back) shows plain colour.
    Drawings of one subject never agree in 3D: a profile puts the eye further back than the front view does, so
    projected, the head would carry it twice. Works in the side picture: every pixel is followed to the surface
    it lands on and from there into the leading picture; where that one sees the same surface (up to about 75
    degrees) in a clearly different broad colour, the side picture's feature is a double. Whole features only:
    the clash is grown over the feature's own colour area, and big areas are left alone (a different design,
    not a doubled detail). -> (mask in the side picture, number of features)."""
    import cv2
    v = src["view"]
    valid = D > register.EMPTY
    if not valid.any() or not leaders:
        return np.zeros(D.shape, bool), 0
    h, w = D.shape
    Dm = np.where(valid, D, np.nan)
    gy, gx = np.gradient(Dm)
    ok = valid & np.isfinite(gx) & np.isfinite(gy) & subject
    ys, xs = np.nonzero(ok)
    if len(ys) == 0:
        return np.zeros(D.shape, bool), 0
    u, wv = v.u_of((xs + 0.5) / sc), v.v_of((ys + 0.5) / sc)
    d = v.toward_camera
    P = np.outer(u, v.u_vec) + np.outer(wv, v.v_vec) + np.outer(D[ys, xs], d)
    # surface normal from the depth slopes (depth per world unit along u, and along v: rows run downward)
    N = np.outer(-gx[ys, xs] * ppu, v.u_vec) + np.outer(gy[ys, xs] * ppu, v.v_vec) + d[None, :]
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    if src["flip"]:
        P, N = P * np.array([-1.0, 1.0, 1.0]), N * np.array([-1.0, 1.0, 1.0])
    mine = base[ys, xs].astype(np.float32)
    # what the surroundings of each pixel look like in the side picture: its plain colour there
    kk = int(round(0.11 * height * ppu)) | 1
    kk = min(max(kk, 3), 255)
    plain = np.dstack([cv2.medianBlur(np.ascontiguousarray(base[:, :, ch]), kk) for ch in range(3)])[ys, xs].astype(np.float32)
    feature = np.abs(mine - plain).max(1) > 0.8 * veto * 255.0
    clash = np.zeros(len(ys), bool)
    for L in leaders:
        lv = L["view"]
        ix, iy, z, inb = L["to_px"](P)
        ixc, iyc = np.clip(ix, 0, L["depth"].shape[1] - 1), np.clip(iy, 0, L["depth"].shape[0] - 1)
        facing = N @ L["d"]
        slope = np.sqrt(np.clip(1.0 - facing ** 2, 0.0, 1.0)) / np.maximum(facing, 0.1)
        sees = inb & (facing > 0.2) & (z >= L["depth"][iyc, ixc] - (0.004 * diag + 2.5 * L["px"] * slope))
        px, py = lv.x_of(P @ lv.u_vec) * L["scale"], lv.y_of(P @ lv.v_vec) * L["scale"]
        theirs = _sample(L["base"], px, py) * 255.0
        # the leading picture shows there what the side picture has all around the feature: plain surface
        clash |= sees & feature & (np.abs(mine - theirs).max(1) > veto * 255.0) & \
            (np.abs(theirs - plain).max(1) < 0.6 * veto * 255.0)
    img = np.zeros((h, w), np.uint8)
    img[ys[clash], xs[clash]] = 1
    ro = max(1, int(round(0.005 * height * ppu)))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1)))
    cnt, lab, stats, _c = cv2.connectedComponentsWithStats(img, connectivity=8)
    out = np.zeros((h, w), bool)
    area_all = float(subject.sum())
    found = 0
    reach = max(2, int(round(0.06 * height * ppu)))
    for i in range(1, cnt):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < (0.012 * height * ppu) ** 2:
            continue
        comp = lab == i
        col = np.median(base[comp].astype(np.float32), axis=0)
        like = (np.abs(base.astype(np.float32) - col).max(2) < 34) & subject
        near = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))) > 0
        _n, lab2 = cv2.connectedComponents((like & near).astype(np.uint8), connectivity=8)
        ids = np.unique(lab2[comp])
        whole = np.isin(lab2, ids[ids > 0])
        if whole.sum() > 0.035 * area_all:                 # a broad area: a difference in design, not a doubled detail
            continue
        out |= whole
        found += 1
    if out.sum() > 0.08 * area_all:
        return np.zeros((h, w), bool), 0
    return out, found


def paint_from_ref(objs=None, ref=None, views=None, fill="auto", bias=None, sharpness=12.0, shading="soft",
                   mirror=True, detail=0.02, guard=0.008, grazing=0.3, own=0.4, zones=None, erase=None,
                   occluders=None, tidy=True, veto=0.22, graze_hint=True, calm=40, name=None, verbose=True,
                   faces=True, agree=0.2, _positional=False, _hint_gap=0.02):
    """Colour object(s) by projecting the reference pictures onto them (faces, prints, gradients, trim: everything
    the pictures show). Call it after the geometry is final.

    objs: object(s) to paint (default: the whole model). Other model parts still cast occlusion on them.
    views: which reference views to use (default: every view in the pack).
    fill: colour for surfaces no view can see: 'auto' = the flat colour of the nearest painted surface (nearest
          along the surface; folds count as far, so each part continues in its own colour), '#rrggbb' = that
          flat colour.
    bias: {view: priority}; default front 1.6, mirrored side 0.7, others 1.0 (the front picture wins where views
          overlap).
    sharpness: how strongly the most head-on view dominates where views overlap. High values (default) give each
               surface to one picture, which avoids ghosted double features; lower values (3-5) blend softly.
    shading: 'soft' (half self-lit: keeps the artwork's own shading readable, default), 'lit' (fully shaded by
             scene lights), 'unlit' (exactly the picture colours), or a number 0..1 = self-lit share.
    mirror: when the pack has only one side view, use its mirror image for the other side (right for subjects
            that are roughly left/right symmetric; pass False for asymmetric ones).
    detail: resolution of the view-blend weights as a fraction of the subject height; coarse faces are subdivided
            down to this. The pictures themselves are sampled per pixel, so a smaller value does not sharpen the
            artwork, it only refines where one view hands over to another (and costs faces).
    erase: [RefMask, ...]: parts of pictures to leave out. Turnaround art draws a bow, an ear or a fin facing the
           viewer in every view, which no single 3D shape can do: projected, the model wears it two or three
           times. Keep it in the picture that shows it best (usually the front) and erase it from the others:
           erase=[S.mask(box=(0.04, 0.51, 0.26, 0.68))] with S = k.ref().side. The area is repainted with the
           main colour around it. A mirrored side view is erased along with its original.
    tidy: before projecting, clean each picture against the geometry it lands on: seen from the view's camera
          the surface falls into patches (a fringe, the face behind it, a sleeve, the skirt behind that); where
          the drawn fringe spills a little over onto the face patch, that strip is replaced by the face's own
          flat colour, and the other way round. Removes the strips of wrong colour along part edges; wide
          disagreements are left as drawn (k.conform softens the relief there instead). True = strips up to
          about 2.4 % of the height across; a number sets half that width as a fraction of the height (0.02
          for a mesh that fits its picture loosely); False turns it off if it eats a detail you need.
    guard: how far (fraction of the height) to stay away from occlusion edges on top of that.
    grazing: surfaces turned further away from a view than this cosine (0.3 = 73 degrees) ignore that view.
    own: pictures of one subject from different sides never agree exactly in 3D, so views are not averaged. In
         the order front, back, sides, a view takes every surface it sees with a cosine above this (0.4 = within
         about 66 degrees of head-on) and later views only get what is left: a face keeps the one pair of eyes
         drawn in the front picture. 0 turns this off; values below 0.25 are raised to 0.25, because a picture
         projected at a grazing angle smears.
    zones: [(region, views), ...]: inside the region only the named view(s) may paint. The cure when a feature
           shows up twice. region is either
           - a RefMask, the easy way: zones=[(F.mask(box=(-0.2, 0.43, 0.2, 0.72)), "front")] gives everything seen
             through that part of the front picture (a face with its cheeks) to the front view. It reaches 0.1 x
             height behind the first surface; a third tuple item changes that depth: (mask, "front", 0.2);
           - a world-space box ((x0, y0, z0), (x1, y1, z1)).
    faces: a face is painted by the front picture alone and eyes drawn in the other pictures are left out,
           using the faces and eyes `f3d ref` found (see <view>.eyes). This is what keeps a head from wearing
           two pairs of eyes; pass False only for a subject whose side pictures show different eyes.
    agree: the side pictures may only add to what the front and back pictures say, not contradict it. Where the
           front (or back) picture can see a surface at all, even at a slant, it knows that part's broad colour;
           a side picture that shows another colour there (0..1 difference above `agree`) drew a neighbouring
           part over it (an apron reaching round onto the skirt) and is ignored on that spot. 0 = off.
    veto: the front and back pictures overrule the side pictures. Drawings of one subject never agree in 3D: a
          profile draws the eye further back than the front view puts it, so both would land on the head. Where
          a side picture shows a feature on a surface that the front (or back) picture also sees, at up to about
          75 degrees, and shows as plain there (colour difference above `veto`, 0..1), that whole feature is
          removed from the side picture before projecting. 0 = off.
    The defaults are tuned; change one knob at a time and only for a defect you can see.
    Returns the shared material. Painted colours survive k.export (it bakes them into a texture).
    """
    R = refkit.as_ref(ref) if ref is not None else refkit.ref()
    sources = _sources(R, views, mirror)
    if not sources:
        raise ValueError("paint_from_ref: the reference pack has none of the requested views")
    items = _mesh_objs(objs)
    if not items:
        raise ValueError("paint_from_ref: nothing to paint")
    from . import modify
    modify.apply(items)
    capped = [o.name for o in items if not densify(o, detail * max(R.height, 1e-6))]
    if capped and verbose:
        print("[f3d] paint_from_ref: detail=%g would need more than 600k faces on %s; refined as far as that budget "
              "allows (the texture itself is sampled per pixel either way)" % (detail, ", ".join(capped[:3])))
    if 0 < own < 0.25:
        print("[f3d] paint_from_ref: own=%g raised to 0.25 (below that the owning picture is stretched over surfaces "
              "it only grazes; use zones=[(box, view)] to hand a region to one view)" % own)
        own = 0.25
    zone_list = []
    face_note, n_face = None, 0
    zones = list(zones or [])
    if faces:
        fz, fe, face_note = _face_rules(sources)
        zones, erase, n_face = zones + fz, list(erase or []) + fe, len(fz)
    from . import ironing as ironmod
    ironed = [(m, n) for m, n in ironmod.zones() if any(x["view"] is m.view and not x["flip"] for x in sources)]
    for entry in list(zones or []) + ironed:
        zbox, zviews = entry[0], entry[1]
        zviews = [zviews] if isinstance(zviews, str) else list(zviews)
        if isinstance(zbox, refkit.RefMask):
            zone_list.append(("mask", zbox, zviews, float(entry[2]) if len(entry) > 2 else 0.1))
        else:
            a, b = np.array(zbox[0], float), np.array(zbox[1], float)
            zone_list.append(("box", (np.minimum(a, b), np.maximum(a, b)), zviews, 0.0))
    pri = {"front": 1.6, "left_mirror": 0.7, "right_mirror": 0.7}
    # priorities are meant as "this much more head-on": make them independent of the sharpness exponent
    pri = {k_: v_ ** (sharpness / 4.0) for k_, v_ in pri.items()}
    pri.update({k_: float(v_) ** (sharpness / 4.0) for k_, v_ in (bias or {}).items()})
    occ = _mesh_objs(occluders) if occluders is not None else [o for o in core.scene_objects() if o.type == "MESH"]
    for o in items:
        if o not in occ:
            occ.append(o)
    dg = core.depsgraph()
    vs, ts, off = [], [], 0
    for o in occ:
        v, t = core.world_tris(o, dg)
        if len(t):
            vs.append(v)
            ts.append(t + off)
            off += len(v)
    V = np.concatenate(vs)
    T = np.concatenate(ts)
    tris = V[T]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    lo, hi = V.min(0), V.max(0)
    flipv = np.array([-1.0, 1.0, 1.0])
    for src in sources:                                    # one depth buffer per source: who is in front of whom
        v = src["view"]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        cu, cv_ = (corners * (flipv if src["flip"] else 1.0)) @ v.u_vec, corners @ v.v_vec
        fr = (float(cu.min()) - 0.02 * diag, float(cv_.min()) - 0.02 * diag, float(cu.max()) + 0.02 * diag,
              float(cv_.max()) + 0.02 * diag)
        ppu = 448.0 / max(fr[2] - fr[0], fr[3] - fr[1])
        geo = tris * flipv if src["flip"] else tris
        D, to_px = raster.depth_buffer(geo, v.toward_camera, v.u_vec, v.v_vec, fr, ppu)
        r = max(2, int(round(guard * R.height * ppu)))
        import cv2
        near = cv2.dilate(D, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        src.update(depth=D, near=near, to_px=to_px, px=1.0 / ppu, r=r)
        # the picture, enlarged to working size and tidied against the surface patches it lands on
        sc = register.view_scale(v)
        w2, h2 = int(round(v.size_px[0] * sc)), int(round(v.size_px[1] * sc))
        tex2 = cv2.resize(v.texture, (w2, h2), interpolation=cv2.INTER_CUBIC)
        from .. import refs as refmod
        flat2, line = refmod.cel_flat(tex2, 0.0045 * max(h2, w2))
        subject2 = cv2.resize(v.mask().array.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0

        def wipe(gone, tex2, flat2, line):
            """Repaint an area with the main colour found around it: one calm colour, no invented drawing."""
            rr = max(3, int(round(0.03 * R.height * v.px_per_unit * sc)))
            lbl_n, lbl = cv2.connectedComponents(gone.astype(np.uint8), connectivity=8)
            flat2, tex2 = flat2.copy(), tex2.copy()
            for li in range(1, lbl_n):
                g = lbl == li
                ring = (cv2.dilate(g.astype(np.uint8), np.ones((2 * rr + 1, 2 * rr + 1), np.uint8)) > 0) & ~gone & subject2
                if not ring.any():
                    continue
                q = (flat2[ring] // 12).astype(np.int64)
                code = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
                vals, counts = np.unique(code, return_counts=True)
                main = flat2[ring][code == vals[counts.argmax()]].mean(0).astype(np.uint8)
                flat2[g] = main
                tex2[g] = main
            return tex2, flat2, line & ~gone

        for em in (erase or []):
            if not isinstance(em, refkit.RefMask):
                raise ValueError("paint_from_ref: erase takes RefMasks, e.g. erase=[k.ref().side.mask(box=(u0, v0, u1, v1))]")
            if em.view is not v:
                continue
            gone = cv2.resize(em.grow(0.006 * R.height).array.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
            if gone.any():
                tex2, flat2, line = wipe(gone, tex2, flat2, line)
        replaced = 0.0
        D2, _tp, ppu2 = register.view_buffer(geo, v, sc)
        lab, _cnt, _steep = register.geo_patches(D2, ppu2, R.height)
        src["doubled"] = 0
        if veto and src["key"] not in ("front", "back"):
            leaders = [x for x in sources if x["key"] in ("front", "back") and "base" in x]
            gone, count = _doubled(src, D2, ppu2, sc, register.base_picture(lab > 0, flat2, ppu2, R.height), subject2,
                                   leaders, R.height, diag, veto)
            if count:
                rg = max(1, int(round(0.006 * R.height * ppu2)))
                gone = cv2.dilate(gone.astype(np.uint8), np.ones((2 * rg + 1, 2 * rg + 1), np.uint8)) > 0
                tex2, flat2, line = wipe(gone & subject2, tex2, flat2, line)
                src["doubled"] = count
        if tidy:
            pic, _foreign = register.drawn_layout(lab, register.to_lab(flat2), ppu2, R.height)
            tex2, flat2, replaced = register.clean_picture(lab, pic, tex2, flat2, line, ppu2, R.height,
                                                           0.012 if tidy is True else float(tidy))
        src.update(tex=tex2, flat=flat2, base=register.base_picture(lab > 0, flat2, ppu2, R.height), scale=sc,
                   replaced=replaced)
    mat = _material(sources, shading, name, _positional)
    stats = {}
    for o in items:
        me = o.data
        P, Nn = _world_geometry(o)
        nv = len(P)
        adj = _adjacency(me)
        deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1.0)
        loops = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loops)
        zone_sel = []                                      # (vertex selection, views allowed inside it)
        for kind, shape, zviews, zdepth in zone_list:
            if kind == "box":
                zone_sel.append((np.all((P >= shape[0]) & (P <= shape[1]), axis=1), zviews))
                continue
            # a picture region: everything seen through that mask from its view, down to zdepth behind the
            # first surface (so a face mask takes the face and its cheeks, not the hair behind the head)
            zv = shape.view
            zsrc = next((x for x in sources if x["view"] is zv and not x["flip"]), None)
            if zsrc is None:
                raise ValueError("paint_from_ref: a zone mask comes from the %r view, which is not being painted" % zv.name)
            zu, zw = P @ zv.u_vec, P @ zv.v_vec
            mx = np.floor(zv.x_of(zu)).astype(int)
            my = np.floor(zv.y_of(zw)).astype(int)
            mh, mw = shape.array.shape
            inm = (mx >= 0) & (mx < mw) & (my >= 0) & (my < mh)
            inm &= shape.array[np.clip(my, 0, mh - 1), np.clip(mx, 0, mw - 1)]
            jx, jy, jz, jin = zsrc["to_px"](P)
            D = zsrc["depth"]
            first = D[np.clip(jy, 0, D.shape[0] - 1), np.clip(jx, 0, D.shape[1] - 1)]
            zone_sel.append((inm & jin & (jz >= first - zdepth * R.height), zviews))
        if face_note:                                      # the automatic face zones come first in the list
            at = me.attributes.get("f3d_face")
            if at is not None:
                me.attributes.remove(at)
            mark = np.zeros(nv, bool)
            for zsel, _zv in zone_sel[len(zones) - n_face:len(zones)]:
                mark |= zsel
            me.attributes.new("f3d_face", "FLOAT", "POINT").data.foreach_set("value", mark.astype(np.float32))
        ns = len(sources)
        B = np.zeros((ns, nv), np.float32)                 # blend weight per source
        conf = np.zeros((ns, nv), np.float32)              # can this source be trusted here at all
        C = np.zeros((ns, nv, 3), np.float32)
        S = np.zeros((ns, nv), np.float32)                 # 1 = original picture, 0 = its line-free twin
        F_ = np.zeros((ns, nv, 3), np.float32)             # each part's base colour: what unseen neighbours are filled from
        free = np.ones(nv, np.float32)                     # share of each vertex not yet owned by an earlier view
        H = np.zeros((ns, nv), np.float32)                 # colour hints: views shut out of a zone still know
        Hc = np.zeros((ns, nv, 3), np.float32)             # what colour the part has where the owner cannot see
        sees = np.zeros((ns, nv), bool)                    # the view has this spot in sight, at whatever angle
        # which picture a surface belongs to is decided by its broad orientation, not by every fold: the walls
        # of a pleat on the front of a skirt belong to the front picture, not to the profile they happen to face
        Nb = Nn.copy()
        for _ in range(int(calm)):
            Nb = 0.5 * Nb + 0.5 * (adj @ Nb) / deg[:, None]
        nb_len = np.linalg.norm(Nb, axis=1, keepdims=True)
        Nb = np.where(nb_len > 0.4, Nb / np.maximum(nb_len, 1e-9), Nn)       # thin parts: both sides cancel, keep the real one
        for i, src in enumerate(sources):
            v, d = src["view"], src["d"]
            facing = Nn @ d
            broad = np.maximum(Nb @ d, facing)
            Pm = P * flipv if src["flip"] else P
            ix, iy, z, inb = src["to_px"](Pm)
            ixc, iyc = np.clip(ix, 0, src["depth"].shape[1] - 1), np.clip(iy, 0, src["depth"].shape[0] - 1)
            slope = np.sqrt(np.clip(1.0 - facing ** 2, 0.0, 1.0)) / np.maximum(facing, 0.1)
            tol = 0.003 * diag + 1.5 * src["px"] * slope
            vis = inb & (facing > 0.03) & (z >= src["depth"][iyc, ixc] - tol)
            # occlusion edges: the pictures never line up with the geometry to the pixel, and a surface just
            # behind an edge may catch the colours of whatever stands in front of it
            edge_free = z >= src["near"][iyc, ixc] - (tol + src["r"] * src["px"] * slope + 0.012 * diag)
            # beside a real occlusion edge shrink the visible set by one ring: a vertex there would smear the
            # occluder's colours. Not inside rough relief (frills, folds), where every bump hides a crumb of the
            # next one: there the shrinking would eat the whole area
            vis &= edge_free | ((adj @ vis.astype(np.float32)) >= deg - 0.5)
            u, w = Pm @ v.u_vec, Pm @ v.v_vec
            f = v.frame
            inside = (u >= f[0]) & (u <= f[2]) & (w >= f[1]) & (w <= f[3])
            sees[i] = vis & inside & edge_free
            t = np.clip((broad - grazing) / 0.25, 0.0, 1.0)        # skip grazing angles: there a pixel smears into a streak
            c = np.where(vis & inside, t * t * (3 - 2 * t), 0.0)
            for zsel, zviews in zone_sel:
                if src["key"].split("_")[0] not in zviews and src["key"] not in zviews:
                    H[i] = np.maximum(H[i], np.where(zsel, c * np.clip(facing, 0.0, 1.0) ** 2, 0.0))
                    c = np.where(zsel, 0.0, c)
            px, py = v.x_of(u) * src["scale"], v.y_of(w) * src["scale"]
            F_[i] = Hc[i] = _sample(src["base"], px, py)[:, ::-1]
            # a surface this view sees only at a slant is not painted from it, but the view still knows which
            # colour is there: better than the nearest painted colour, which in a busy area (frills, folds) is
            # as often the neighbour's
            # (also where a bump of the same relief hides it by a hair: a crevice of a frill is frill-coloured)
            if graze_hint:
                gz = inb & inside & (c <= 0) & (broad > 0.08) & (z >= src["near"][iyc, ixc] - _hint_gap * R.height)
                H[i] = np.maximum(H[i], np.where(gz, 0.3 + np.clip(broad, 0.0, 1.0), 0.0))
            conf[i] = c
            # beside an occlusion edge this view is a last resort: any other view that sees the spot wins
            safe = np.where(edge_free, 1.0, 0.04)
            B[i] = c * np.clip(broad, 0.0, 1.0) ** sharpness * pri.get(src["key"], 1.0) * safe * (0.02 + 0.98 * free)
            if own > 0:
                t2 = np.clip((broad - own) / 0.15, 0.0, 1.0)
                free = free * (1.0 - c * (t2 * t2 * (3 - 2 * t2)) * np.where(edge_free, 1.0, 0.0))
            t3 = np.clip((facing - 0.45) / 0.25, 0.0, 1.0)
            S[i] = (t3 * t3 * (3 - 2 * t3)) * np.where(edge_free, 1.0, 0.0)
            sharp_c = _sample(src["tex"], px, py)[:, ::-1]                      # BGR -> RGB, sRGB values
            flat_c = _sample(src["flat"], px, py)[:, ::-1]
            C[i] = flat_c + (sharp_c - flat_c) * S[i][:, None]
            if not _positional:
                uvl = me.uv_layers.get("f3d_" + src["key"]) or me.uv_layers.new(name="f3d_" + src["key"])
                if uvl is None:
                    raise RuntimeError("paint_from_ref: '%s' has no free UV layer for view %s (Blender allows 8)" % (o.name, src["key"]))
                uv = np.stack([(u - f[0]) / (f[2] - f[0]), (w - f[1]) / (f[3] - f[1])], -1).astype(np.float32)
                uvl.data.foreach_set("uv", uv[loops].ravel())
        if agree > 0:
            leads = [i for i, x in enumerate(sources) if x["key"] in ("front", "back")]
            for j, x in enumerate(sources):
                if j in leads or not leads:
                    continue
                clash = np.zeros(nv, bool)
                for i in leads:
                    clash |= sees[i] & (np.abs(F_[i] - F_[j]).max(1) > agree)
                clash &= conf[j] > 0
                if clash.any():
                    B[j, clash] = 0.0
                    conf[j, clash] = 0.0
                    x["clash"] = x.get("clash", 0) + int(clash.sum())
        # a view should paint whole areas. A scrap of a secondary view inside another view's area (the side
        # picture catching the wall of a fringe on the front of a head) is never right: hand it to the fill
        scrap = _scraps(me, B, conf, max(40, int(0.0012 * nv)))
        if scrap.any():
            B[:, scrap] = 0.0
            conf[:, scrap] = 0.0
        tot = B.sum(0)
        seen = conf.max(0)
        known = (seen > 0.5) & (tot > 1e-6)
        Wn = np.where(tot > 1e-6, B / np.maximum(tot, 1e-6), 0.0).astype(np.float32)
        for i, src in enumerate(sources):
            for key, vals in (("f3d_w_" + src["key"], Wn[i]), ("f3d_s_" + src["key"], S[i])):
                at = me.attributes.get(key)
                if at is not None:
                    me.attributes.remove(at)
                me.attributes.new(key, "FLOAT", "POINT").data.foreach_set("value", vals.astype(np.float32))
        col = (Wn[:, :, None] * C).sum(0)
        flat_col = (Wn[:, :, None] * F_).sum(0)
        # inside a zone the other views may not draw, but where the owner cannot see (the side of an ironed
        # face) they still say which colour the part has there: skin, not the hair next to it
        covered = float(known.mean())
        hint = (H.max(0) > 0.25) & ~known
        if hint.any():
            flat_col[hint] = ((H[:, :, None] * Hc).sum(0) / np.maximum(H.sum(0), 1e-6)[:, None])[hint]
            known = known | hint
        if isinstance(fill, str) and fill != "auto":
            lin = _srgb_to_linear(np.where(hint[:, None], flat_col, col))
            lin[~known] = np.array(core.rgba(fill)[:3], np.float32)
        elif not known.any():
            base = core.rgba(R.palette[0] if R.palette else "#808080")[:3]
            lin = np.tile(np.array(base, np.float32), (nv, 1))
        else:
            lin = _srgb_to_linear(_fill_unseen(me, P, Nn, flat_col, known, R.height))
        sk = np.where(tot > 1e-6, seen, 0.0)[:, None]
        final = _srgb_to_linear(col) * sk + lin * (1.0 - sk)                 # what the shader shows, per vertex
        for key, dtype, values in ((FILL_ATTR, "FLOAT_COLOR", np.concatenate([lin, np.ones((nv, 1), np.float32)], 1)),
                                   (SEEN_ATTR, "FLOAT", np.where(tot > 1e-6, seen, 0.0)),
                                   ("f3d_rgb", "FLOAT_COLOR", np.concatenate([final, np.ones((nv, 1), np.float32)], 1))):
            at = me.attributes.get(key)
            if at is not None:
                me.attributes.remove(at)
            me.attributes.new(key, dtype, "POINT").data.foreach_set("color" if dtype == "FLOAT_COLOR" else "value",
                                                                    values.astype(np.float32).ravel())
        lead_w = np.zeros(nv, np.float32)
        lead_seen = np.zeros(nv, np.float32)
        for i, src in enumerate(sources):
            if src["key"] in ("front", "back"):
                lead_w += Wn[i] * S[i]
                lead_seen = np.maximum(lead_seen, conf[i])
        at = me.attributes.get("f3d_lead_seen")
        if at is not None:
            me.attributes.remove(at)
        me.attributes.new("f3d_lead_seen", "FLOAT", "POINT").data.foreach_set("value", lead_seen.astype(np.float32))
        at = me.attributes.get("f3d_keep")
        if at is not None:
            me.attributes.remove(at)
        me.attributes.new("f3d_keep", "FLOAT", "POINT").data.foreach_set(
            "value", np.where(tot > 1e-6, np.clip(lead_w, 0.0, 1.0) * seen, 0.0).astype(np.float32))
        me.materials.clear()
        me.materials.append(mat)
        me.update()
        stats[o.name] = covered
    if verbose:
        worst = sorted(stats.items(), key=lambda kv: kv[1])[:4]
        print("[f3d] paint_from_ref: %d object(s) painted from %s. Directly covered by a picture: %s; the rest "
              "(surfaces between and behind the views) continues the nearest painted colour, which is normal" % (
                  len(items), "/".join(src["key"] for src in sources),
                  ", ".join("%s %.0f%%" % (k, v * 100) for k, v in worst)))
        if face_note:
            print("[f3d] paint_from_ref: " + face_note)
        cl = ["%s" % x["key"] for x in sources if x.get("clash", 0) > 0.01 * sum(len(o.data.vertices) for o in items)]
        if cl:
            print("[f3d] paint_from_ref: where the %s picture contradicts the colour the front / back picture sees "
                  "on the same surface it was left out (agree=0 keeps it)" % " / ".join(cl))
        if tidy:
            print("[f3d] paint_from_ref: tidied the pictures against the geometry (colours on the wrong side of a "
                  "part edge replaced: %s)" % ", ".join("%s %.0f%%" % (x["key"], x["replaced"] * 100) for x in sources))
        dbl = ["%s %d" % (x["key"], x["doubled"]) for x in sources if x.get("doubled")]
        if dbl:
            print("[f3d] paint_from_ref: dropped features the side picture draws a second time where the front / back "
                  "picture already has that surface plain (%s); veto=0 keeps them" % ", ".join(dbl))
    return mat


# ---------------------------------------------------------------- one part, its own drawing
def _main_colour(pix):
    """Most common colour of (n, 3) uint8 pixels (coarsely binned), as float BGR."""
    q = (pix // 12).astype(np.int64)
    code = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, counts = np.unique(code, return_counts=True)
    return pix[code == vals[counts.argmax()]].astype(np.float32).mean(0)


def paint_part(objs, views="front", keep=None, base="auto", facing=0.35, mirror=False, shading="soft", ref=None,
               name=None, verbose=True):
    """Colour a part you built with its own share of the reference drawing: the picture where the part faces a
    view, one clean base colour everywhere else. This is what keeps a model built from parts clean from every
    side: a face carries the eyes and the mouth and nothing of the fringe hanging over it, hair carries its
    highlights and no skin, and no picture is ever stretched around a side it does not show.

    objs: the part (one object or several that share the drawing, e.g. both sleeves).
    views: 'front' or a list in order of priority, e.g. ['back', 'right'] for a hair mass.
    keep: False = no drawing at all, only the base colour or gradient (clean and plain; right for a part whose
          own relief already shows its form, such as a generated mesh of hair). Otherwise a
          RefMask (or a list, each for its own view): the pixels of the picture that belong to this part. The
          rest of the picture is replaced by the base colour before projecting. Build it from colours, regions
          and boxes, e.g. for a face under a fringe (S = skin tone read off the picture):
              F.mask(color="#fbe9e2", tol=22) | F.mask(box=(-0.19, 0.50, -0.05, 0.60)) | F.mask(box=...)   # skin + eyes
          Default: everything the view shows, which is right for a part nothing overlaps (an apron, a shield).
    base: colour of the part where no picture applies: '#rrggbb'; 'auto' = the main colour of the kept pixels
          under the part; 'gradient' = that colour as it changes with height in the picture (ombre hair, a
          shaded gown), continued all the way round; or a list of colours, bottom to top, for an ombre you
          read off the picture yourself: base=["#7295d2", "#607aba", "#4e62a4"].
    facing: surfaces turned further from a view than this cosine show the base colour (0.35 = about 70 degrees).
    mirror: also project a side view, mirrored, onto the far side (symmetric parts painted from one profile).
    shading: as in paint_from_ref.
    Other parts never cast occlusion here: each part is painted as if it stood alone, so call it per part.
    Returns the material. Survives k.export (baked into the texture) and can be retouched with k.dab.
    """
    import cv2
    from .. import refs as refmod
    R = refkit.as_ref(ref) if ref is not None else refkit.ref()
    names = [views] if isinstance(views, str) else list(views)
    for n in names:
        if n not in R.views:
            raise ValueError("paint_part: the reference pack has no %r view (it has: %s)" % (n, ", ".join(R.views)))
    keeps = {}
    if keep is False:
        keep, facing = None, 2.0
    for m in ([] if keep is None else [keep] if isinstance(keep, refkit.RefMask) else list(keep)):
        if not isinstance(m, refkit.RefMask):
            raise ValueError("paint_part: keep takes RefMasks, e.g. keep=k.ref()['front'].mask(color='#fbe9e2')")
        if m.view.name not in names:
            raise ValueError("paint_part: a keep mask comes from the %r view, which is not in views=%r" % (m.view.name, names))
        keeps[m.view.name] = (keeps[m.view.name] | m) if m.view.name in keeps else m
    items = _mesh_objs(objs)
    if not items:
        raise ValueError("paint_part: nothing to paint")
    from . import modify
    modify.apply(items)
    for o in items:
        densify(o, 0.02 * max(R.height, 1e-6))
    dg = core.depsgraph()
    tris = np.concatenate([(lambda V, T: V[T])(*core.world_tris(o, dg)) for o in items])
    flipv = np.array([-1.0, 1.0, 1.0])
    sources = []
    for n in names:
        v = R[n]
        sources.append(dict(key=n, view=v, flip=False, d=v.toward_camera))
        if mirror and n in ("right", "left"):
            sources.append(dict(key=n + "_mirror", view=v, flip=True, d=v.toward_camera * flipv))
    grad = None                                            # (heights ascending, colours BGR) for base='gradient'
    flat_base = None
    if isinstance(base, (list, tuple)) and base and all(isinstance(c, str) for c in base):
        cols = []
        for c in base:
            r, g, b = core.rgba(c)[:3]
            cols.append([core.linear_to_srgb(b) * 255.0, core.linear_to_srgb(g) * 255.0, core.linear_to_srgb(r) * 255.0])
        cols = np.array(cols, np.float32)
        zs = tris[:, :, 2]
        grad = (np.linspace(float(zs.min()), float(zs.max()), len(cols)), cols) if len(cols) > 1 else None
        flat_base = cols.mean(0)
        base = "stops"
    for src in sources:
        v = src["view"]
        sc = register.view_scale(v)
        w2, h2 = int(round(v.size_px[0] * sc)), int(round(v.size_px[1] * sc))
        tex2 = cv2.resize(v.texture, (w2, h2), interpolation=cv2.INTER_CUBIC)
        flat2, _line = refmod.cel_flat(tex2, 0.0045 * max(h2, w2))
        K = keeps[v.name].array if v.name in keeps else v.mask().array
        K = cv2.resize(K.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
        foot = raster.silhouette((tris * flipv if src["flip"] else tris).astype(np.float32), v.rec, sc) > 0
        if foot.shape != K.shape:
            foot = cv2.resize(foot.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST) > 0
        own = K & foot
        if own.sum() < 50:
            own = K
        if flat_base is None:                              # the leading view decides the base colour
            if isinstance(base, str) and base not in ("auto", "gradient"):
                c = core.rgba(base)[:3]
                flat_base = np.array([core.linear_to_srgb(c[2]), core.linear_to_srgb(c[1]), core.linear_to_srgb(c[0])], np.float32) * 255.0
            elif own.any():
                flat_base = _main_colour(flat2[own])
            else:
                flat_base = np.array(refkit.hex_to_bgr(R.palette[0] if R.palette else "#808080"), np.float32)
            if base == "gradient" and v.name in ("front", "back", "left", "right") and own.any():
                per_row = own.sum(1)                       # sparse rows (a stray highlight, a tip) say nothing
                rows = np.flatnonzero(per_row >= max(5, 0.15 * per_row.max()))
                if len(rows) >= 4:
                    cols = np.stack([np.median(flat2[y][own[y]].astype(np.float32), axis=0) for y in rows])
                    sig = max(1.0, 0.055 * R.height * v.px_per_unit * sc)   # an ombre, not every shadow band
                    full = np.stack([np.interp(np.arange(h2), rows, cols[:, c]) for c in range(3)], 1)
                    full = cv2.GaussianBlur(full.reshape(h2, 1, 3).astype(np.float32), (0, 0), sigmaX=0.01, sigmaY=sig).reshape(h2, 3)
                    z = np.array([v.v_of((y + 0.5) / sc) for y in range(h2)])
                    grad = (z[::-1].copy(), full[::-1].copy())
        if grad is not None and v.name in ("front", "back", "left", "right"):
            z_rows = np.array([v.v_of((y + 0.5) / sc) for y in range(h2)])
            base_img = np.stack([np.interp(z_rows, grad[0], grad[1][:, c]) for c in range(3)], 1)[:, None, :].repeat(w2, 1)
        else:
            base_img = np.broadcast_to(flat_base, (h2, w2, 3))
        # the part's own drawing over its base colour; one pixel in from the mask edge, so no ink outline of a
        # neighbouring part comes along
        a = cv2.erode(K.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 0.8)[:, :, None]
        src.update(scale=sc,
                   tex=np.clip(tex2 * a + base_img * (1 - a), 0, 255).astype(np.uint8),
                   flat=np.clip(flat2 * a + base_img * (1 - a), 0, 255).astype(np.uint8))
    key = name or "f3d_part_%s" % (items[0].name if items[0].name.isascii() else "%08x" % (hash(items[0].name) & 0xffffffff))
    mat = _material(sources, shading, key)

    def base_at(z):
        if grad is None:
            return np.broadcast_to(flat_base[::-1] / 255.0, (len(z), 3))
        return np.stack([np.interp(z, grad[0], grad[1][:, c]) for c in (2, 1, 0)], 1) / 255.0

    covered = []
    for o in items:
        me = o.data
        P, Nn = _world_geometry(o)
        nv = len(P)
        loops = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loops)
        ns = len(sources)
        B = np.zeros((ns, nv), np.float32)
        S = np.zeros((ns, nv), np.float32)
        C = np.zeros((ns, nv, 3), np.float32)
        free = np.ones(nv, np.float32)
        for i, src in enumerate(sources):
            v = src["view"]
            cosv = Nn @ src["d"]
            Pm = P * flipv if src["flip"] else P
            u, w = Pm @ v.u_vec, Pm @ v.v_vec
            f = v.frame
            inside = (u >= f[0]) & (u <= f[2]) & (w >= f[1]) & (w <= f[3])
            t = np.clip((cosv - facing) / 0.2, 0.0, 1.0)
            c = np.where(inside, t * t * (3 - 2 * t), 0.0).astype(np.float32)
            B[i] = c * free
            free = free * (1.0 - c)
            t3 = np.clip((cosv - 0.45) / 0.25, 0.0, 1.0)
            S[i] = t3 * t3 * (3 - 2 * t3)
            px, py = v.x_of(u) * src["scale"], v.y_of(w) * src["scale"]
            sharp_c, flat_c = _sample(src["tex"], px, py)[:, ::-1], _sample(src["flat"], px, py)[:, ::-1]
            C[i] = flat_c + (sharp_c - flat_c) * S[i][:, None]
            uvl = me.uv_layers.get("f3d_" + src["key"]) or me.uv_layers.new(name="f3d_" + src["key"])
            if uvl is None:
                raise RuntimeError("paint_part: '%s' has no free UV layer for view %s (Blender allows 8)" % (o.name, src["key"]))
            uv = np.stack([(u - f[0]) / (f[2] - f[0]), (w - f[1]) / (f[3] - f[1])], -1).astype(np.float32)
            uvl.data.foreach_set("uv", uv[loops].ravel())
        seen = np.clip(B.sum(0), 0.0, 1.0)
        Wn = np.where(seen > 1e-6, B / np.maximum(B.sum(0), 1e-6), 0.0).astype(np.float32)
        for i, src in enumerate(sources):
            for akey, vals in (("f3d_w_" + src["key"], Wn[i]), ("f3d_s_" + src["key"], S[i])):
                at = me.attributes.get(akey)
                if at is not None:
                    me.attributes.remove(at)
                me.attributes.new(akey, "FLOAT", "POINT").data.foreach_set("value", vals.astype(np.float32))
        lin = _srgb_to_linear(base_at(P[:, 2]).astype(np.float32))
        final = _srgb_to_linear((Wn[:, :, None] * C).sum(0)) * seen[:, None] + lin * (1.0 - seen[:, None])
        for akey, dtype, values in ((FILL_ATTR, "FLOAT_COLOR", np.concatenate([lin, np.ones((nv, 1), np.float32)], 1)),
                                    (SEEN_ATTR, "FLOAT", seen),
                                    ("f3d_rgb", "FLOAT_COLOR", np.concatenate([final, np.ones((nv, 1), np.float32)], 1))):
            at = me.attributes.get(akey)
            if at is not None:
                me.attributes.remove(at)
            me.attributes.new(akey, dtype, "POINT").data.foreach_set("color" if dtype == "FLOAT_COLOR" else "value",
                                                                     values.astype(np.float32).ravel())
        me.materials.clear()
        me.materials.append(mat)
        me.update()
        covered.append(float((seen > 0.5).mean()))
    if verbose:
        b = flat_base[::-1]
        print("[f3d] paint_part(%s): drawing from %s on %.0f%% of the surface, base %s elsewhere" % (
            ", ".join(o.name for o in items[:3]), "/".join(x["key"] for x in sources), 100.0 * float(np.mean(covered)),
            "a height gradient" if grad is not None else "#%02x%02x%02x" % (int(b[0]), int(b[1]), int(b[2]))))
    return mat


# ---------------------------------------------------------------- baking
def is_painted(obj):
    return any(m is not None and m.get("f3d_painted") for m in getattr(obj.data, "materials", []))


def _atlas_dirs():
    """Ten directions to unfold a surface along: every 45 degrees round the vertical, plus up and down.
    -> [(direction, u axis, v axis)]"""
    out = []
    for i in range(8):
        a = math.radians(45.0 * i)
        out.append((np.array([math.sin(a), -math.cos(a), 0.0]), np.array([math.cos(a), math.sin(a), 0.0]),
                    np.array([0.0, 0.0, 1.0])))
    out.append((np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])))
    out.append((np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])))
    return out


def _unwrap(items, size=2048, pad=3):
    """One shared UV atlas ('f3d_bake') for all the objects, made of flat views of the surface: every face is
    laid out as seen from the one of ten directions that its neighbourhood faces, connected faces of one
    direction form a chart, and the charts are packed at one scale. Unlike an angle-based unwrap of a dense
    organic mesh (thousands of crumbs, most of the texture empty) this keeps the charts large and the picture
    sharp."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy import ndimage
    dirs = _atlas_dirs()
    D = np.stack([d for d, _u, _v in dirs])
    charts = []                                             # per object: dict
    sizes = []
    for o in items:
        me = o.data
        uvl = me.uv_layers.get("f3d_bake") or me.uv_layers.new(name="f3d_bake")
        if uvl is None:
            raise RuntimeError("'%s' has no free UV layer for baking (Blender allows 8 per mesh)" % o.name)
        me.uv_layers.active = uvl
        uvl.active_render = True
        P, Nn = _world_geometry(o)
        nv, nf, nl = len(me.vertices), len(me.polygons), len(me.loops)
        adj = _adjacency(me)
        deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1.0)[:, None]
        Nb = Nn.copy()
        for _ in range(30):
            Nb = 0.5 * Nb + 0.5 * (adj @ Nb) / deg
        ln = np.linalg.norm(Nb, axis=1, keepdims=True)
        Nb = np.where(ln > 0.4, Nb / np.maximum(ln, 1e-9), Nn)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        le = np.empty(nl, np.int32)
        me.loops.foreach_get("edge_index", le)
        ls = np.empty(nf, np.int32)
        me.polygons.foreach_get("loop_start", ls)
        lt = np.empty(nf, np.int32)
        me.polygons.foreach_get("loop_total", lt)
        lf = np.repeat(np.arange(nf), lt)                   # face of every loop (loops are stored face by face)
        fn = np.empty(nf * 3, np.float32)
        me.polygons.foreach_get("normal", fn)
        m3 = np.array(o.matrix_world, dtype=np.float64)[:3, :3]
        fn = fn.reshape(-1, 3).astype(np.float64) @ np.linalg.inv(m3)
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        fb = np.zeros((nf, 3))
        np.add.at(fb, lf, Nb[lv])
        fb /= np.maximum(np.linalg.norm(fb, axis=1, keepdims=True), 1e-12)
        own = fn @ D.T                                      # how squarely each face itself looks along each direction
        score = fb @ D.T + 0.35 * own
        score[own < 0.25] = -9.0                            # never lay a face out edge-on
        lab = score.argmax(1)
        edge_on = score.max(1) < -5
        lab[edge_on] = own[edge_on].argmax(1)
        # faces sharing an edge and a direction belong to one chart
        order = np.argsort(le, kind="stable")
        e_sorted, f_sorted = le[order], lf[order]
        pair = e_sorted[1:] == e_sorted[:-1]
        fa, fb_ = f_sorted[:-1][pair], f_sorted[1:][pair]
        same = lab[fa] == lab[fb_]
        g = coo_matrix((np.ones(int(same.sum()), np.int8), (fa[same], fb_[same])), shape=(nf, nf))
        cnt, chart = connected_components(g, directed=False)
        # crumbs join the chart next door when they can be seen from its direction at all
        for _ in range(2):
            csize = np.bincount(chart, minlength=cnt)
            small = csize[chart] < 8
            cand = small[fa] & ~small[fb_] & (own[fa, lab[fb_]] > 0.12)
            lab[fa[cand]] = lab[fb_[cand]]
            cand = small[fb_] & ~small[fa] & (own[fb_, lab[fa]] > 0.12)
            lab[fb_[cand]] = lab[fa[cand]]
            same = lab[fa] == lab[fb_]
            g = coo_matrix((np.ones(int(same.sum()), np.int8), (fa[same], fb_[same])), shape=(nf, nf))
            cnt, chart = connected_components(g, directed=False)
        U = np.stack([u for _d, u, _v in dirs])[lab[lf]]
        V = np.stack([v for _d, _u, v in dirs])[lab[lf]]
        pu, pv = (P[lv] * U).sum(1), (P[lv] * V).sum(1)
        cl = chart[lf]
        idx = np.arange(cnt)
        u0, u1 = ndimage.minimum(pu, cl, idx), ndimage.maximum(pu, cl, idx)
        v0, v1 = ndimage.minimum(pv, cl, idx), ndimage.maximum(pv, cl, idx)
        charts.append(dict(obj=o, uvl=uvl, pu=pu, pv=pv, cl=cl, u0=u0, v0=v0, w=u1 - u0, h=v1 - v0, cnt=cnt))
        sizes.append((cnt, nf))
    W = np.concatenate([c["w"] for c in charts])
    H = np.concatenate([c["h"] for c in charts])
    n = len(W)
    order = np.argsort(-H, kind="stable")
    scale = None
    side = math.sqrt(float(((W + 1e-4) * (H + 1e-4)).sum()) * 1.2)       # world units across the atlas, first guess
    for _ in range(4):
        gap = pad * side / float(size)                      # world size of the gap between charts
        x = y = row = 0.0
        X, Y = np.zeros(n), np.zeros(n)
        for i in order:
            w, h = W[i] + gap, H[i] + gap
            if x + w > side and x > 0:
                x, y, row = 0.0, y + row, 0.0
            X[i], Y[i] = x, y
            x += w
            row = max(row, h)
        total = y + row
        widest = float((W + gap).max())
        if total <= side and widest <= side:
            scale = 1.0 / side
            break
        side = max(widest, side * math.sqrt(max(total / side, 1.0)) * 1.03)
    if scale is None:
        scale = 1.0 / max(side, total)
    gap = pad * side / float(size)
    off = 0
    for c in charts:
        k_ = c["cl"] + off
        uv = np.stack([(X[k_] + gap / 2 + c["pu"] - c["u0"][c["cl"]]) * scale,
                       (Y[k_] + gap / 2 + c["pv"] - c["v0"][c["cl"]]) * scale], -1).astype(np.float32)
        c["uvl"].data.foreach_set("uv", np.clip(uv, 0.0, 1.0).ravel())
        c["obj"].data.update()
        off += c["cnt"]
    return dict(charts=int(n), texels_per_unit=float(size) * scale)


def _bake_image(items, size, margin=6, source="final"):
    """Bake the painted colour (source='final') or a per-vertex float attribute (source=its name) of the objects
    into a new image through their 'f3d_bake' UV layer. -> float32 array (size, size, 4), rows bottom-up as
    Blender stores them, alpha 1 where something was baked."""
    sc = bpy.context.scene
    img = bpy.data.images.new("f3d_bake_tmp", int(size), int(size), alpha=True)
    img.generated_color = (0.0, 0.0, 0.0, 0.0)
    saved = dict(engine=sc.render.engine, samples=sc.cycles.samples, device=sc.cycles.device)
    rewired, swapped, temp_mat = [], [], None
    if source == "final":
        mats = {}
        for o in items:
            for m in o.data.materials:
                if m is not None and m.get("f3d_painted"):
                    mats[m.name] = m
        for m in mats.values():
            nt = m.node_tree
            out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
            mix = nt.nodes.get(FINAL_NODE)
            if mix is None:
                raise RuntimeError("bake: material '%s' was not made by paint_from_ref" % nt.id_data.name)
            prev = out.inputs["Surface"].links[0].from_socket
            em = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(mix.outputs["Result"], em.inputs["Color"])
            nt.links.new(em.outputs[0], out.inputs["Surface"])
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = img
            for n in nt.nodes:
                n.select = False
            tex.select = True
            nt.nodes.active = tex
            rewired.append((nt, out, prev, em, tex))
    else:
        temp_mat = bpy.data.materials.new("f3d_bake_attr")
        temp_mat.use_nodes = True
        nt = temp_mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        at = nt.nodes.new("ShaderNodeAttribute")
        at.attribute_name = source
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(at.outputs["Fac"], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.select = True
        nt.nodes.active = tex
        for o in items:
            swapped.append((o, [m for m in o.data.materials]))
            o.data.materials.clear()
            o.data.materials.append(temp_mat)
    from .staging import quiet, safe_shader_names
    try:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = 1
        sc.cycles.device = "CPU"
        for o in bpy.context.scene.objects:
            o.select_set(False)
        for o in items:
            o.data.uv_layers.active = o.data.uv_layers["f3d_bake"]
            o.select_set(True)
        bpy.context.view_layer.objects.active = items[0]
        with quiet(), safe_shader_names():
            bpy.ops.object.bake(type="EMIT", margin=int(margin), use_clear=True)
    finally:
        for nt, out, prev, em, tex in rewired:
            nt.links.new(prev, out.inputs["Surface"])
            nt.nodes.remove(em)
            nt.nodes.remove(tex)
        for o, old in swapped:
            o.data.materials.clear()
            for m in old:
                o.data.materials.append(m)
        if temp_mat is not None:
            bpy.data.materials.remove(temp_mat)
        sc.render.engine = saved["engine"]
        sc.cycles.samples = saved["samples"]
        sc.cycles.device = saved["device"]
    px = np.empty(int(size) * int(size) * 4, np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(int(size), int(size), 4)


def _close_gaps(px, items):
    """Texels no face was baked into: faces smaller than a texel get their own vertex colour there, and every
    remaining empty texel continues the nearest filled one, so no black crumbs show along seams and slivers."""
    from scipy import ndimage
    size = px.shape[0]
    have = px[:, :, 3] > 0.5
    for o in items:
        me = o.data
        uvl = me.uv_layers.get("f3d_bake")
        col = me.attributes.get("f3d_rgb")
        if uvl is None or col is None or len(me.polygons) == 0:
            continue
        nl = len(me.loops)
        uv = np.empty(nl * 2, np.float32)
        uvl.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)
        lv = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        rgb = np.empty(len(me.vertices) * 4, np.float32)
        col.data.foreach_get("color", rgb)
        rgb = rgb.reshape(-1, 4)[:, :3]
        x = np.clip((uv[:, 0] * size).astype(int), 0, size - 1)
        y = np.clip((uv[:, 1] * size).astype(int), 0, size - 1)
        miss = ~have[y, x]
        if miss.any():
            # the image holds display (sRGB) values of the linear colours
            lin = np.clip(rgb[lv[miss]], 0.0, 1.0)
            px[y[miss], x[miss], :3] = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
            px[y[miss], x[miss], 3] = 1.0
            have[y[miss], x[miss]] = True
    if have.any() and not have.all():
        idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
        px[:, :, :3] = px[idx[0], idx[1], :3]
    px[:, :, 3] = 1.0
    return px


def _baked_material(items, px, path, emit, cleanup, name="f3d_baked"):
    """Give the objects one ordinary material showing the baked picture `px` through their 'f3d_bake' UVs."""
    size = px.shape[0]
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.pixels.foreach_set(np.ascontiguousarray(px, np.float32).ravel())
    if path:
        img.filepath_raw = core.out_path(path)
        img.file_format = "PNG"
        img.save()
    img.pack()
    baked = bpy.data.materials.new(name)
    if baked.node_tree is None:
        baked.use_nodes = True
    nt = baked.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "f3d_bake"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.1
            break
    if emit > 0:
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        ms = nt.nodes.new("ShaderNodeMixShader")
        ms.inputs["Fac"].default_value = emit
        nt.links.new(bsdf.outputs[0], ms.inputs[1])
        nt.links.new(em.outputs[0], ms.inputs[2])
        nt.links.new(ms.outputs[0], out.inputs["Surface"])
    else:
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    baked["f3d_baked"] = True
    baked["f3d_emit"] = emit
    old = set()
    for o in items:
        me = o.data
        old.update(m for m in me.materials if m is not None)
        me.materials.clear()
        me.materials.append(baked)
        if cleanup:
            for uv in [u for u in me.uv_layers if u.name.startswith("f3d_") and u.name != "f3d_bake"]:
                me.uv_layers.remove(uv)
            for at in [a for a in me.attributes if a.name.startswith(("f3d_w_", "f3d_s_")) or a.name in (FILL_ATTR, SEEN_ATTR)]:
                me.attributes.remove(at)
    for m in old:
        if m.users == 0 and m.get("f3d_painted"):
            bpy.data.materials.remove(m)
    return baked


def bake(objs=None, size=2048, path=None, margin=6, cleanup=True, verbose=True):
    """Turn projection paint into an ordinary image texture on a UV atlas (what other programs and game engines
    understand). k.export() calls this automatically; call it yourself to control the size or keep the image.

    objs: painted objects (default: every painted object).  size: texture side in pixels.
    path: where to save the PNG (default: packed inside the .blend only).
    Returns the baked material (shared by all the objects), or None when nothing is painted.
    """
    items = [o for o in _mesh_objs(objs) if is_painted(o)]
    if not items:
        return None
    _unwrap(items, size)
    emit = max(float(m.get("f3d_emit", 0.5)) for o in items for m in o.data.materials if m is not None and m.get("f3d_painted"))
    px = _close_gaps(_bake_image(items, size, margin, "final"), items)
    baked = _baked_material(items, px, path, emit, cleanup)
    if verbose:
        print("[f3d] bake: %d object(s) -> one %dx%d texture%s" % (len(items), size, size, " (%s)" % path if path else ""))
    return baked


def plain_material_for_export(objs):
    """Swap baked materials for a plain Principled twin that exporters understand. Returns an undo function."""
    swaps = []
    twins = {}
    for o in objs:
        me = o.data
        for i, m in enumerate(getattr(me, "materials", [])):
            if m is None or not (m.get("f3d_baked") or m.get("f3d_flat")):
                continue
            if m.name not in twins:
                t = m.copy()
                t.name = m.name + "_export"
                nt = t.node_tree
                out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
                bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
                nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
                twins[m.name] = t
            swaps.append((me, i, m))
            me.materials[i] = twins[m.name]

    def undo():
        for me, i, m in swaps:
            me.materials[i] = m
        for t in twins.values():
            if t.users == 0:
                bpy.data.materials.remove(t)
    return undo


def undercoat(inner, outer, verbose=True):
    """What a layer covers takes its colour: the scalp under a wig, a chest under a coat. A gap in the cover (a
    parting, a seam between locks) then shows more of the cover instead of a bright spot of what lies beneath.
    Only surfaces the cover hides from nearly every side change, and never a face (a forehead under a fringe
    keeps its skin). Both must be painted (k.layer does this for the layers given as over=).
    Returns the number of vertices recoloured."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    outers = [o for o in _mesh_objs(outer) if o.data.attributes.get("f3d_rgb") is not None]
    if not outers:
        return 0
    dg = core.depsgraph()
    trees = []
    for o in outers:
        V, T = core.world_tris(o, dg)
        rgb = np.empty(len(o.data.vertices) * 4, np.float32)
        o.data.attributes["f3d_rgb"].data.foreach_get("color", rgb)
        trees.append((BVHTree.FromPolygons([Vector(p) for p in V.tolist()], T.tolist()), T, rgb.reshape(-1, 4)[:, :3]))
    dirs = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    total = 0
    for o in _mesh_objs(inner):
        me = o.data
        fa, sa, ra = me.attributes.get(FILL_ATTR), me.attributes.get(SEEN_ATTR), me.attributes.get("f3d_rgb")
        if fa is None or sa is None:
            continue
        P, Nn = _world_geometry(o)
        nv = len(P)
        bb = P.max(0) - P.min(0)
        eps = 0.002 * float(np.linalg.norm(bb))
        covered = np.zeros(nv, bool)
        colour = np.zeros((nv, 3), np.float32)
        face = np.zeros(nv, np.float32)
        if me.attributes.get("f3d_face") is not None:       # a face keeps its own colour whatever hangs over it
            me.attributes["f3d_face"].data.foreach_get("value", face)
        # only what lies close under the cover can be covered by it
        from scipy.spatial import cKDTree
        near = np.zeros(nv, bool)
        reach = 0.12 * float(np.linalg.norm(bb))
        for o2 in outers:
            P2, _n2 = _world_geometry(o2)
            near |= cKDTree(P2).query(P, distance_upper_bound=reach)[0] < reach
        for vi in np.flatnonzero((face < 0.5) & near):
            p, nrm = P[vi], Nn[vi]
            start = Vector(p + nrm * eps)
            first = None
            for tree, T, rgb in trees:
                h_ = tree.ray_cast(start, Vector(nrm))
                if h_[0] is not None and (first is None or h_[3] < first[0]):
                    first = (h_[3], rgb, T[h_[2]])
            if first is None:
                continue
            look = dirs[dirs @ nrm > 0.25]
            free, allow = 0, 0.3 * len(look)                # a gap in the cover lets a few directions through
            for d in look:
                if not any(tree.ray_cast(start, Vector(d))[0] is not None for tree, _T, _rgb in trees):
                    free += 1
                    if free > allow:
                        break
            if free <= allow:
                covered[vi] = True
                colour[vi] = first[1][first[2]].mean(0)
        # ... and what pokes through a hole in the cover although the original shows the cover there from every
        # side it can be seen from (a scalp through a gap in a generated wig): it lies right at the cover and
        # belongs under it
        for o2 in outers:
            lname = o2.get("f3d_layer")
            if not lname:
                continue
            try:
                from .layerkit import layer_ref
                L = layer_ref(lname)
            except (FileNotFoundError, ValueError):
                continue
            P2, _n2 = _world_geometry(o2)
            rgb2 = np.empty(len(P2) * 4, np.float32)
            o2.data.attributes["f3d_rgb"].data.foreach_get("color", rgb2)
            rgb2 = rgb2.reshape(-1, 4)[:, :3]
            dist, idx = cKDTree(P2).query(P)
            close = (dist < 0.02 * float(np.linalg.norm(bb))) & (face < 0.5) & ~covered
            if not close.any():
                continue
            under = close.copy()
            said = np.zeros(nv, bool)
            for v in L:
                if v.name not in ("front", "back", "right", "left"):
                    continue
                m = v.shown()
                if m is None:
                    under[:] = False
                    break
                for flip in ((False, True) if v.name in ("right", "left") and not any(
                        w.name == {"right": "left", "left": "right"}[v.name] for w in L) else (False,)):
                    d = v.toward_camera * (np.array([-1.0, 1.0, 1.0]) if flip else 1.0)
                    Pm = P * np.array([-1.0, 1.0, 1.0]) if flip else P
                    x = np.floor(v.x_of(Pm @ v.u_vec)).astype(int)
                    y = np.floor(v.y_of(Pm @ v.v_vec)).astype(int)
                    ok = (x >= 0) & (x < m.shape[1]) & (y >= 0) & (y < m.shape[0])
                    over = np.zeros(nv, bool)
                    over[ok] = m[y[ok], x[ok]]
                    under &= over | (Nn @ d < 0.25)         # a view the surface does not face says nothing
                    said |= over & (Nn @ d >= 0.25)
            under &= said | (Nn[:, 2] > 0.6)                # ... but someone must say so, unless it faces up
            if under.any():
                covered |= under
                colour[under] = rgb2[idx[under]]
        if not covered.any():
            continue
        adj = _adjacency(me)
        w = covered.astype(np.float32)
        col = colour.copy()
        for _ in range(2):                                  # soften the rim by a ring or two
            den = np.maximum(np.asarray(adj @ w).ravel(), 1e-6)
            col = np.where(covered[:, None], colour, (adj @ (col * w[:, None])) / den[:, None])
            w = np.maximum(w, 0.5 * np.asarray(adj @ w).ravel() / np.maximum(np.asarray(adj.sum(1)).ravel(), 1.0))
        w = np.clip(w, 0.0, 1.0)
        seen = np.empty(nv, np.float32)
        sa.data.foreach_get("value", seen)
        fill = np.empty(nv * 4, np.float32)
        fa.data.foreach_get("color", fill)
        fill = fill.reshape(-1, 4)
        fill[:, :3] = fill[:, :3] * (1 - w[:, None]) + col * w[:, None]
        fa.data.foreach_set("color", fill.ravel())
        sa.data.foreach_set("value", seen * (1 - w))
        if ra is not None:
            rgb = np.empty(nv * 4, np.float32)
            ra.data.foreach_get("color", rgb)
            rgb = rgb.reshape(-1, 4)
            rgb[:, :3] = rgb[:, :3] * (1 - w[:, None]) + col * w[:, None]
            ra.data.foreach_set("color", rgb.ravel())
        me.update()
        total += int(covered.sum())
        if verbose:
            print("[f3d] undercoat(%s under %s): %d of %d vertices are covered from every side and took the cover's "
                  "colour" % (o.name, ", ".join(x.name for x in outers), int(covered.sum()), nv))
    return total


def dab(objs, color, at=None, radius=0.1, box=None, facing=None, unseen_only=False, soft=0.5):
    """Retouch projection paint: recolour part of painted object(s) with a flat colour. For the spots the pictures
    cannot settle (the crown of a head, undersides, a smear at a seam).

    at + radius: paint within a sphere (world units), or  box=((x0, y0, z0), (x1, y1, z1)).
    facing: only surfaces looking this way: 'up', 'down', 'front', 'back', 'left', 'right' or a vector.
    unseen_only: touch only what no reference view could see.   soft: 0..1 feathering of the sphere's edge.
    Call after k.paint_from_ref and before k.export / k.bake.
    """
    dirs = {"up": (0, 0, 1), "down": (0, 0, -1), "front": (0, -1, 0), "back": (0, 1, 0), "left": (-1, 0, 0),
            "right": (1, 0, 0)}
    lin = np.array(core.rgba(color)[:3], np.float32)
    touched = 0
    for o in _mesh_objs(objs):
        me = o.data
        fa, sa, ra = me.attributes.get(FILL_ATTR), me.attributes.get(SEEN_ATTR), me.attributes.get("f3d_rgb")
        if fa is None or sa is None:
            raise ValueError("dab: '%s' is not painted (call k.paint_from_ref first, and dab before baking)" % o.name)
        P, Nn = _world_geometry(o)
        nv = len(P)
        w = np.ones(nv, np.float32)
        if at is not None:
            d = np.linalg.norm(P - np.array(at, float), axis=1) / max(radius, 1e-9)
            w *= np.clip((1.0 - d) / max(soft, 1e-3), 0.0, 1.0).astype(np.float32)
        if box is not None:
            lo, hi = np.minimum(box[0], box[1]), np.maximum(box[0], box[1])
            w *= np.all((P >= lo) & (P <= hi), axis=1)
        if facing is not None:
            v = np.array(dirs[facing] if isinstance(facing, str) else facing, float)
            v /= np.linalg.norm(v)
            w *= np.clip((Nn @ v - 0.2) / 0.4, 0.0, 1.0).astype(np.float32)
        seen = np.empty(nv, np.float32)
        sa.data.foreach_get("value", seen)
        if unseen_only:
            w *= 1.0 - seen
        if not (w > 0).any():
            continue
        fill = np.empty(nv * 4, np.float32)
        fa.data.foreach_get("color", fill)
        fill = fill.reshape(-1, 4)
        fill[:, :3] = fill[:, :3] * (1 - w[:, None]) + lin * w[:, None]
        fa.data.foreach_set("color", fill.ravel())
        sa.data.foreach_set("value", seen * (1 - w))
        if ra is not None:
            rgb = np.empty(nv * 4, np.float32)
            ra.data.foreach_get("color", rgb)
            rgb = rgb.reshape(-1, 4)
            rgb[:, :3] = rgb[:, :3] * (1 - w[:, None]) + lin * w[:, None]
            ra.data.foreach_set("color", rgb.ravel())
        me.update()
        touched += int((w > 0.01).sum())
    if touched == 0:
        print("[f3d] warning: dab(%s) matched no surface (check at / box / facing)" % (color,))
    return objs
