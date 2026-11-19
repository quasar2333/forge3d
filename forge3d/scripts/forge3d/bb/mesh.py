"""Write the Blender-side model as Blockbench mesh elements: a .bbmodel that keeps the model's real shape and
texture (Blockbench's Generic Model format supports free-form meshes next to cubes)."""
import base64
import json
import os

import bmesh
import bpy
import numpy as np

from ..kit import colors, core
from .model import SAVED, W2B, _guid


def _png_data_url(rgba):
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(), buf.tobytes()


def _image_rgba(img):
    w, h = img.size
    px = np.empty(w * h * img.channels, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, img.channels)[::-1]
    if img.colorspace_settings.name not in ("sRGB", "Filmic sRGB"):
        px = np.concatenate([np.where(px[:, :, :3] <= 0.0031308, px[:, :, :3] * 12.92,
                                      1.055 * np.clip(px[:, :, :3], 0, None) ** (1 / 2.4) - 0.055), px[:, :, 3:]], 2)
    out = np.full((h, w, 4), 255, np.uint8)
    out[:, :, :min(4, img.channels)] = np.clip(np.round(px[:, :, :4] * 255), 0, 255).astype(np.uint8)
    return out


def _uv_layer_for(mat, me):
    if mat is not None and mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.bl_idname == "ShaderNodeUVMap" and n.uv_map in me.uv_layers:
                return me.uv_layers[n.uv_map]
    return next((l for l in me.uv_layers if l.active_render), me.uv_layers.active)


def _texture(name, rgba, index):
    url, _raw = _png_data_url(rgba)
    h, w = rgba.shape[:2]
    return {"name": name, "path": "", "folder": "", "namespace": "", "id": str(index), "group": "", "width": w,
            "height": h, "uv_width": w, "uv_height": h, "particle": False, "use_as_default": False,
            "layers_enabled": False, "sync_to_project": "", "render_mode": "default", "render_sides": "auto",
            "pbr_channel": "color", "frame_time": 1, "frame_order_type": "loop", "frame_order": "",
            "frame_interpolate": False, "visible": True, "internal": True, "saved": False, "uuid": _guid(),
            "source": url}


def save_scene(path, objs=None, height=64, faces=60000, texture=2048, format="free", name=None):
    """Save the Blender-side model as a .bbmodel made of mesh elements (one per object, grouped by parent).

    The highest-fidelity Blockbench product: the model keeps its real shape and its texture, and every object
    stays a separate, named, editable mesh in Blockbench. Needs a format with meshes (Generic Model, 'free').
    height: how many Blockbench pixels tall the model is (16 = one block).
    faces: total face budget; denser models are reduced to it (Blockbench gets sluggish beyond ~100k faces).
    texture: side of the baked texture for projection-painted objects.
    Objects with plain materials get their colours from a small palette texture.
    """
    from ..kit import paint, refkit
    if format != "free":
        raise ValueError("mesh models need the Generic Model format (format='free'); for %r build cuboids with "
                         "bb.Model or bb.voxelize" % format)
    path = core.out_path(path if path.endswith(".bbmodel") else path + ".bbmodel")
    paint.bake(objs, size=texture)
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    if not items:
        raise ValueError("save_scene: no model geometry")
    dg = core.depsgraph()
    bb = core.bbox(items)
    try:
        world_h = refkit.ref().height
    except Exception:  # noqa: BLE001 - no reference pack in use
        world_h = max(bb.max.z - min(0.0, bb.min.z), 1e-9)
    s = float(height) / world_h
    total = sum(len(o.evaluated_get(dg).data.polygons) for o in items)
    ratio = min(1.0, faces / float(total)) if faces and total else 1.0
    textures, tex_index, palette = [], {}, {}
    elements, groups, outliner = [], {}, []
    n_faces = 0
    for oi, o in enumerate(items):
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        if ratio < 0.999 and len(me.polygons) > 12:
            tmp = bpy.data.objects.new("f3d_tmp", me)
            bpy.context.scene.collection.objects.link(tmp)
            tmp.modifiers.new("d", "DECIMATE").ratio = max(ratio, 0.002)
            dg2 = core.depsgraph()
            me2 = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg2), preserve_all_data_layers=True, depsgraph=dg2)
            bpy.data.objects.remove(tmp, do_unlink=True)
            bpy.data.meshes.remove(me)
            me = me2
        bm = bmesh.new()
        bm.from_mesh(me)
        ngons = [f for f in bm.faces if len(f.verts) > 4]
        if ngons:
            bmesh.ops.triangulate(bm, faces=ngons)
        bm.to_mesh(me)
        bm.free()
        mats = list(me.materials) or [None]
        mw = np.array(o.matrix_world, dtype=np.float64)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        P = (co.reshape(-1, 3).astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]) * s
        centre = (P.min(0) + P.max(0)) / 2 if len(P) else np.zeros(3)
        local = (P - centre) @ W2B.T
        verts = {"v%d" % i: [round(float(c), 4) for c in p] for i, p in enumerate(local)}
        flip = np.linalg.det(mw[:3, :3]) < 0
        fdict = {}
        uv_cache = {}
        for f in me.polygons:
            mat = mats[min(f.material_index, len(mats) - 1)]
            key = mat.name if mat is not None else None
            if key not in tex_index:
                col, img = colors._base_color(mat)
                if img is not None and _uv_layer_for(mat, me) is not None:
                    tex_index[key] = (len(textures), None)
                    textures.append(("%s.png" % (img.name.rsplit(".", 1)[0] or "texture"), _image_rgba(img)))
                else:
                    rgb = tuple(int(v) for v in colors._lin_to_srgb8(np.array(col if col is not None else (0.6, 0.6, 0.6))))
                    palette.setdefault(rgb, len(palette))
                    tex_index[key] = ("palette", rgb)
            ti, swatch = tex_index[key]
            loops = list(f.loop_indices)
            vkeys = ["v%d" % me.loops[li].vertex_index for li in loops]
            if flip:
                loops, vkeys = loops[::-1], vkeys[::-1]
            if swatch is None:
                if key not in uv_cache:
                    uv_cache[key] = _uv_layer_for(mat, me)
                layer = uv_cache[key]
                if layer is None:
                    # this mesh wears a textured material another mesh registered, but has no UV layer for it
                    # (materials of the same name from separately built parts): give it the plain colour
                    col = colors._base_color(mat)[0]
                    rgb = tuple(int(v) for v in colors._lin_to_srgb8(np.array(col if col is not None else (0.6, 0.6, 0.6))))
                    palette.setdefault(rgb, len(palette))
                    ti, swatch = "palette", rgb
            if swatch is None:
                h, w = textures[ti][1].shape[:2]
                uv = {k: [round(layer.data[li].uv.x * w, 3), round((1.0 - layer.data[li].uv.y) * h, 3)]
                      for k, li in zip(vkeys, loops)}
                fdict["f%d" % f.index] = {"vertices": vkeys, "uv": uv, "texture": ti}
            else:
                fdict["f%d" % f.index] = {"vertices": vkeys, "uv": None, "texture": "palette", "swatch": swatch}
        n_faces += len(fdict)
        el = {"name": o.name, "type": "mesh", "uuid": _guid(), "color": oi % 8, "export": True, "visibility": True,
              "locked": False, "render_order": "default", "allow_mirror_modeling": True, "shading": "smooth",
              "origin": [round(float(c), 4) for c in W2B @ centre], "rotation": [0, 0, 0], "vertices": verts,
              "faces": fdict}
        elements.append(el)
        gname = o.parent.name if o.parent is not None and o.parent.type == "EMPTY" else None
        if gname is None:
            outliner.append(el["uuid"])
        else:
            if gname not in groups:
                piv = np.array(o.parent.matrix_world.translation) * s
                groups[gname] = {"name": gname, "uuid": _guid(), "export": True, "locked": False, "selected": False,
                                 "visibility": True, "origin": [round(float(c), 4) for c in W2B @ piv],
                                 "rotation": [0, 0, 0], "color": 0, "children": [], "reset": False, "shade": True,
                                 "mirror_uv": False, "autouv": 0, "isOpen": True, "primary_selected": False}
            groups[gname]["children"].append(el["uuid"])
        bpy.data.meshes.remove(me)
    if n_faces == 0:
        raise RuntimeError("save_scene: the model produced no faces (nothing was written)")
    tex_json = [_texture(nm, rgba, i) for i, (nm, rgba) in enumerate(textures)]
    if palette:                                              # flat-coloured materials: one swatch each
        cols = int(np.ceil(np.sqrt(len(palette))))
        side = 4
        while side < cols * 4:
            side *= 2
        img = np.zeros((side, side, 4), np.uint8)
        img[:, :, 3] = 255
        pal_index = len(tex_json)
        centre_of = {}
        for rgb, i in palette.items():
            r, c = divmod(i, cols)
            img[r * 4:r * 4 + 4, c * 4:c * 4 + 4, :3] = rgb
            centre_of[rgb] = (c * 4 + 2.0, r * 4 + 2.0)
        tex_json.append(_texture("palette.png", img, pal_index))
        for el in elements:
            for f in el["faces"].values():
                if f["texture"] == "palette":
                    cx, cy = centre_of[f.pop("swatch")]
                    f["uv"] = {k: [cx + 0.6 * ((i % 2) * 2 - 1), cy + 0.6 * ((i // 2) * 2 - 1)]
                               for i, k in enumerate(f["vertices"])}
                    f["texture"] = pal_index
    for el in elements:
        for f in el["faces"].values():
            f.pop("swatch", None)
    first = tex_json[0] if tex_json else {"width": 16, "height": 16}
    doc = {"meta": {"format_version": "5.0", "model_format": "free", "box_uv": False},
           "name": name or os.path.splitext(os.path.basename(path))[0], "model_identifier": "", "visible_box": [1, 1, 0],
           "variable_placeholders": "", "variable_placeholder_buttons": [], "timeline_setups": [],
           "unhandled_root_fields": {"forge3d": {"height_px": float(height), "world_height": world_h, "meshes": True}},
           "resolution": {"width": first["width"], "height": first["height"]}, "elements": elements,
           "groups": [{k: v for k, v in g.items() if k != "children"} for g in groups.values()],
           "outliner": list(groups.values()) + outliner, "textures": tex_json}
    with open(path, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    for i, (nm, rgba) in enumerate(textures[:1]):
        with open(os.path.splitext(path)[0] + ".png", "wb") as fh:
            fh.write(_png_data_url(rgba)[1])
    SAVED.append(path)
    print("[f3d] saved %s: %d mesh element(s), %d faces%s, %d texture(s), %d px tall" % (
        path, len(elements), n_faces, " (reduced from %d)" % total if ratio < 0.999 else "", len(tex_json), height))
    return path
