"""The colours a model actually shows, per vertex (for voxelising, palettes and checks)."""
import bpy
import numpy as np

from . import core


def _lin_to_srgb8(c):
    c = np.clip(np.asarray(c, np.float32), 0.0, 1.0)
    s = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(s * 255), 0, 255).astype(np.uint8)


def _image_pixels(img):
    w, h = img.size
    px = np.empty(w * h * img.channels, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, img.channels)


def _base_color(mat):
    """(colour linear rgb or None, image or None) of a material's Principled base colour."""
    if mat is None:
        return (0.6, 0.6, 0.6), None
    if mat.node_tree:
        bsdf = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if bsdf is not None:
            sock = bsdf.inputs["Base Color"]
            if sock.links:
                node = sock.links[0].from_node
                seen = 0
                while node is not None and node.bl_idname != "ShaderNodeTexImage" and seen < 6:
                    nxt = next((i.links[0].from_node for i in node.inputs if i.links), None)
                    node, seen = nxt, seen + 1
                if node is not None and node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
                    return None, node.image
            return tuple(sock.default_value[:3]), None
    return tuple(mat.diffuse_color[:3]), None


def vertex_colors(obj, n=None):
    """sRGB uint8 colour per vertex of the evaluated object: projection paint, image texture or flat material."""
    obj = core.get(obj)
    dg = core.depsgraph()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    try:
        nv = len(me.vertices)
        at = me.attributes.get("f3d_rgb")
        if at is not None and at.domain == "POINT" and len(at.data) == nv:
            c = np.empty(nv * 4, np.float32)
            at.data.foreach_get("color", c)
            return _lin_to_srgb8(c.reshape(-1, 4)[:, :3])
        out = np.full((nv, 3), 153, np.uint8)
        if not me.materials or nv == 0:
            return out
        loops = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", loops)
        pmat = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("material_index", pmat)
        ltot = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_total", ltot)
        lmat = np.repeat(pmat, ltot)
        uv = None
        if me.uv_layers.active is not None:
            uv = np.empty(len(me.loops) * 2, np.float32)
            me.uv_layers.active.data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
        for i, m in enumerate(me.materials):
            sel = lmat == i
            if not sel.any():
                continue
            col, img = _base_color(m)
            if img is not None and uv is not None:
                px = _image_pixels(img)
                h, w = px.shape[:2]
                x = np.clip((uv[sel, 0] % 1.0) * w, 0, w - 1).astype(int)
                y = np.clip((uv[sel, 1] % 1.0) * h, 0, h - 1).astype(int)
                rgb = px[y, x, :3]
                if img.colorspace_settings.name not in ("sRGB", "Filmic sRGB"):
                    rgb = _lin_to_srgb8(rgb)
                else:
                    rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
                out[loops[sel]] = rgb
            elif col is not None:
                out[loops[sel]] = _lin_to_srgb8(np.array(col))
        return out
    finally:
        ev.to_mesh_clear()
