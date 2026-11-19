"""Load a .bbmodel into the current Blender scene (for contact sheets and comparisons)."""
import base64
import json
import os
import tempfile

import bpy
import numpy as np

from ..kit import core, refkit
from .model import W2B, rot_matrix

# Blockbench face -> its four corners as (x, y, z) picks of from(0)/to(1), in the order
# [top-left, top-right, bottom-right, bottom-left] of the face's texture rectangle (checked against Blockbench 5.2)
CORNERS = {
    "east":  [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    "west":  [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    "up":    [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    "down":  [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
    "south": [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    "north": [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
}


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if text.startswith("<lz>"):
        raise ValueError("%s is a compressed .bbmodel; in Blockbench turn off Settings > Export > 'Compress project "
                         "files' and save again" % path)
    doc = json.loads(text)
    if "meta" not in doc:
        raise ValueError("%s is not a Blockbench project (.bbmodel)" % path)
    return doc


def _bb_rot(rot):
    """Blockbench [x, y, z] degrees (three.js order ZYX) -> matrix in Blockbench axes."""
    return rot_matrix(rot or (0, 0, 0))


def _parents(doc):
    """element uuid -> list of group transforms (R, origin) from the innermost group outward."""
    groups = {g["uuid"]: g for g in doc.get("groups", [])}
    chain = {}

    def walk(nodes, stack):
        for n in nodes:
            if isinstance(n, str):
                chain[n] = list(reversed(stack))
            elif isinstance(n, dict):
                g = dict(groups.get(n.get("uuid"), {}))
                g.update({k: v for k, v in n.items() if k in ("origin", "rotation", "name") and k not in g})
                item = (_bb_rot(g.get("rotation")), np.array(g.get("origin", (0, 0, 0)), float), g.get("name", "group"))
                walk(n.get("children", []), stack + [item])
    walk(doc.get("outliner", []), [])
    return chain


def _texture_images(doc, path):
    out = []
    for i, t in enumerate(doc.get("textures", [])):
        img = None
        src = t.get("source") or ""
        if src.startswith("data:"):
            raw = base64.b64decode(src.split(",", 1)[1])
            tmp = os.path.join(tempfile.mkdtemp(prefix="f3d_bb_"), "tex%d.png" % i)
            with open(tmp, "wb") as f:
                f.write(raw)
            img = bpy.data.images.load(tmp)
            img.pack()
        else:
            for cand in (t.get("relative_path") and os.path.join(os.path.dirname(path), t["relative_path"]), t.get("path")):
                if cand and os.path.isfile(cand):
                    img = bpy.data.images.load(cand)
                    break
        out.append((img, float(t.get("uv_width") or doc.get("resolution", {}).get("width", 16)),
                    float(t.get("uv_height") or doc.get("resolution", {}).get("height", 16))))
    return out


def _material(img, name):
    """Blockbench look: nearest-neighbour texels, mostly self-lit with a little shading."""
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.9
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.05
            break
    em = nt.nodes.new("ShaderNodeEmission")
    ms = nt.nodes.new("ShaderNodeMixShader")
    ms.inputs["Fac"].default_value = 0.5
    nt.links.new(bsdf.outputs[0], ms.inputs[1])
    nt.links.new(em.outputs[0], ms.inputs[2])
    if img is not None:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.interpolation = "Closest"
        tex.extension = "EXTEND"
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        cut = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(tex.outputs["Alpha"], cut.inputs["Fac"])
        nt.links.new(tr.outputs[0], cut.inputs[1])
        nt.links.new(ms.outputs[0], cut.inputs[2])
        nt.links.new(cut.outputs[0], out.inputs["Surface"])
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "DITHERED"
    else:
        bsdf.inputs["Base Color"].default_value = (0.6, 0.6, 0.62, 1)
        em.inputs["Color"].default_value = (0.6, 0.6, 0.62, 1)
        nt.links.new(ms.outputs[0], out.inputs["Surface"])
    return m


def load(path, height=None):
    """Create Blender objects for a .bbmodel. height: scale the model to this many world units tall (default: the
    active reference pack's height when the file was written by forge3d, else 16 px = 1 m). Returns the objects."""
    doc = read(path)
    chain = _parents(doc)
    textures = _texture_images(doc, path)
    res = doc.get("resolution", {})
    mats = [_material(img, "bb_tex_%d" % i) for i, (img, _w, _h) in enumerate(textures)] or [_material(None, "bb_plain")]
    cubes = [e for e in doc.get("elements", []) if e.get("type", "cube") == "cube" and e.get("visibility", True) is not False]
    verts, faces, uvs, midx, owner = [], [], [], [], []
    for ci, e in enumerate(cubes):
        frm, to = np.array(e["from"], float), np.array(e["to"], float)
        inf = float(e.get("inflate", 0) or 0)
        frm, to = frm - inf, to + inf
        R = _bb_rot(e.get("rotation"))
        org = np.array(e.get("origin", (0, 0, 0)), float)
        ext = (frm, to)
        for fname, picks in CORNERS.items():
            fd = (e.get("faces") or {}).get(fname)
            if not fd or fd.get("texture", 0) is None or fd.get("texture") is False:
                continue
            uv = fd.get("uv", [0, 0, 0, 0])
            if abs(uv[2] - uv[0]) < 1e-9 or abs(uv[3] - uv[1]) < 1e-9:
                continue
            ti = fd.get("texture", 0)
            ti = ti if isinstance(ti, int) and 0 <= ti < len(textures) else 0
            uw, uh = (textures[ti][1], textures[ti][2]) if textures else (res.get("width", 16), res.get("height", 16))
            quad = [(uv[0], uv[1]), (uv[2], uv[1]), (uv[2], uv[3]), (uv[0], uv[3])]         # TL, TR, BR, BL
            for _ in range(int(round((fd.get("rotation", 0) or 0) / 90.0)) % 4):
                quad = [quad[3], quad[0], quad[1], quad[2]]
            base = len(verts)
            for (px, py, pz), (u, v) in zip(picks, quad):
                p = np.array([ext[px][0], ext[py][1], ext[pz][2]])
                p = R @ (p - org) + org
                for Rg, og, _n in chain.get(e.get("uuid"), []):
                    p = Rg @ (p - og) + og
                verts.append(p)
                uvs.append((u / uw, 1.0 - v / uh))
            faces.append((base, base + 1, base + 2, base + 3))
            midx.append(ti)
            owner.append(ci)
    n_cube_faces = len(faces)
    mesh_els = [e for e in doc.get("elements", []) if e.get("type") == "mesh" and e.get("visibility", True) is not False]
    for mi, e in enumerate(mesh_els):
        R = _bb_rot(e.get("rotation"))
        org = np.array(e.get("origin", (0, 0, 0)), float)
        for fd in (e.get("faces") or {}).values():
            keys = [k for k in fd.get("vertices", []) if k in e["vertices"]]
            if len(keys) < 3:
                continue
            ti = fd.get("texture", 0)
            ti = ti if isinstance(ti, int) and 0 <= ti < len(textures) else 0
            uw, uh = (textures[ti][1], textures[ti][2]) if textures else (res.get("width", 16), res.get("height", 16))
            base = len(verts)
            for k in keys:
                p = R @ np.array(e["vertices"][k], float) + org      # mesh vertices are relative to the origin
                for Rg, og, _n in chain.get(e.get("uuid"), []):
                    p = Rg @ (p - og) + og
                verts.append(p)
                u, v = (fd.get("uv") or {}).get(k, (0.0, 0.0))
                uvs.append((u / uw, 1.0 - v / uh))
            faces.append(tuple(range(base, base + len(keys))))
            midx.append(ti)
            owner.append(len(cubes) + mi)
    names = [c["name"] for c in cubes] + [e.get("name", "mesh") for e in mesh_els]
    if not verts:
        raise ValueError("%s has no visible cubes or meshes" % path)
    V = np.array(verts) @ W2B                                   # Blockbench -> kit axes (W2B is orthonormal)
    meta = (doc.get("unhandled_root_fields") or {}).get("forge3d") or {}
    if height is None:
        try:
            ref_h = refkit.ref().height
        except Exception:  # noqa: BLE001 - no active pack
            ref_h = None
        if ref_h and meta.get("height_px"):
            scale = ref_h / float(meta["height_px"])
        elif ref_h:
            scale = ref_h / max(1e-9, float(V[:, 2].max() - V[:, 2].min()))
        else:
            scale = 1.0 / 16.0
    else:
        scale = height / max(1e-9, float(V[:, 2].max() - V[:, 2].min()))
    V = V * scale
    name = doc.get("name") or os.path.splitext(os.path.basename(path))[0]
    cutouts = bool(meta.get("cutout"))
    if "cutout" not in meta:                     # files from elsewhere: look for see-through texels inside face rectangles
        for ti, (img, uw, uh) in enumerate(textures):
            if img is None or img.channels != 4 or cutouts:
                continue
            iw, ih = img.size
            px = np.empty(iw * ih * 4, np.float32)
            img.pixels.foreach_get(px)
            alpha = px[3::4].reshape(ih, iw)[::-1]
            for e in cubes:
                for fd in (e.get("faces") or {}).values():
                    uv = fd.get("uv") if fd else None
                    if not uv or not isinstance(uv, list) or fd.get("texture", 0) != ti:
                        continue
                    x0, x1 = sorted((int(uv[0] * iw / uw), int(uv[2] * iw / uw)))
                    y0, y1 = sorted((int(uv[1] * ih / uh), int(uv[3] * ih / uh)))
                    if x1 > x0 and y1 > y0 and (alpha[y0:y1, x0:x1] < 0.5).any():
                        cutouts = True
                        break
                if cutouts:
                    break
    objs = []
    split = len(names) <= 400
    owner = np.array(owner)
    for key in (sorted(set(owner.tolist())) if split else [None]):
        sel = [i for i in range(len(faces)) if key is None or owner[i] == key]
        vmap, vs, fs = {}, [], []
        for i in sel:
            f = []
            for vi in faces[i]:
                if vi not in vmap:
                    vmap[vi] = len(vs)
                    vs.append(tuple(V[vi]))
                f.append(vmap[vi])
            fs.append(f)
        me = bpy.data.meshes.new(names[key] if key is not None else name)
        me.from_pydata(vs, [], fs)
        uvl = me.uv_layers.new(name="UVMap")
        k = 0
        for i in sel:
            for vi in faces[i]:
                uvl.data[k].uv = uvs[vi]
                k += 1
        for m in mats:
            me.materials.append(m)
        for pi, i in enumerate(sel):
            me.polygons[pi].material_index = min(midx[i], len(mats) - 1)
        me.update()
        if key is not None and key >= len(cubes):
            me.shade_smooth()
        o = bpy.data.objects.new(me.name, me)
        core.link(o)
        if cutouts:
            o["f3d_cutout"] = True
        if key is not None:
            o["f3d_float"] = True          # cubes of a block model legitimately just touch or overlap
        objs.append(o)
    bpy.context.view_layer.update()
    return objs
