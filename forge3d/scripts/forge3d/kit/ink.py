"""Line art: the dark outline a drawing has around every piece, as real geometry.

The outline is an inverted hull: a copy of the surface pushed out by the line width, facing inward, in the ink
colour. Seen from outside only its far side shows, as a line around every separate piece from any direction. It
is one extra object, so it can be hidden, deleted or exported like any other."""
import bpy
import numpy as np

from . import core

TAG = "f3d_ink"
DEFAULT_COLOR = "#1d2140"
DEFAULT_WIDTH = 0.004          # of the model's height
THIN = 0.6                     # a piece thinner than the line gets a line of this share of its own thickness


def material(color):
    key = "ink_%s" % core.hex_of(color).lstrip("#")
    m = bpy.data.materials.get(key)
    if m is not None:
        return m
    from .materials import _principled
    m = bpy.data.materials.new(key)
    nt, bsdf = _principled(m)
    col = core.rgba(color)
    bsdf.inputs["Base Color"].default_value = col
    bsdf.inputs["Roughness"].default_value = 1.0
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.0
            break
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = col
    # the hull faces inward: its near side is seen from behind and must vanish. Culling does that in EEVEE and
    # in exported files; the Backfacing switch does the same for Cycles, which ignores culling
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(clear.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    m.use_backface_culling = True
    m.diffuse_color = col
    m["f3d_flat"] = True                    # exporters get the plain Principled twin (see paint.plain_material_for_export)
    return m


def _hull(verts, tris, width):
    """Welded vertices pushed out along their normals, each connected piece by min(width, THIN x its thickness).
    -> (verts, tris with reversed winding) or None."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(tris) == 0:
        return None
    scale = float(np.abs(verts).max()) or 1.0
    key = np.round(verts / (scale * 2e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    v = verts[first].astype(np.float64)
    t = inv.reshape(-1)[tris]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    if len(t) == 0:
        return None
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(b - a, c - a)                                    # length = 2 x area
    n = np.zeros_like(v)
    for i in range(3):
        for j in range(3):
            n[:, j] += np.bincount(t[:, i], weights=fn[:, j], minlength=len(v))
    ln = np.linalg.norm(n, axis=1)
    n /= np.maximum(ln, 1e-20)[:, None]
    nv = len(v)
    g = coo_matrix((np.ones(len(t) * 2), (np.concatenate([t[:, 0], t[:, 1]]), np.concatenate([t[:, 1], t[:, 2]]))),
                   shape=(nv, nv))
    count, label = connected_components(g, directed=False)
    tl = label[t[:, 0]]
    area = np.bincount(tl, weights=np.linalg.norm(fn, axis=1) / 2.0, minlength=count)
    cen = np.stack([np.bincount(label, weights=v[:, i], minlength=count) for i in range(3)], 1)
    cen /= np.maximum(np.bincount(label, minlength=count), 1)[:, None]
    oc = cen[tl]
    vol = np.abs(np.bincount(tl, weights=np.einsum("ij,ij->i", a - oc, np.cross(b - oc, c - oc)) / 6.0, minlength=count))
    thick = 2.0 * vol / np.maximum(area, 1e-20)                    # tube: its radius; sheet: its thickness
    w = np.minimum(width, THIN * thick)[label]
    return (v + n * w[:, None]).astype(np.float32), t[:, ::-1].astype(np.int32)


def _mesh(name, verts, tris):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.add(len(tris) * 3)
    me.loops.foreach_set("vertex_index", tris.ravel())
    me.polygons.add(len(tris))
    me.polygons.foreach_set("loop_start", np.arange(0, len(tris) * 3, 3, dtype=np.int32))
    try:
        me.polygons.foreach_set("loop_total", np.full(len(tris), 3, np.int32))
    except (AttributeError, RuntimeError, TypeError):
        pass                                                       # derived from loop_start in current Blender
    me.update(calc_edges=True)
    return me


def clear():
    """Remove the line art."""
    for o in [o for o in bpy.data.objects if o.get(TAG)]:
        me = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


def outline(objs=None, width=None, color=None, name="ink outline"):
    """Give the model the line art of a drawing: a dark outline around every separate piece, from any direction.

    Call it last in the build script (after k.assemble / painting), for a model after a reference that is drawn
    with outlines; `f3d ref` says whether it is. Pieces that are separate objects or separate shells get their own
    contour, exactly as the drawing separates them with a line: hair locks, ruffles, bows, fingers. What was fused
    into one skin has no line inside it.
    width: line width in world units (default: measured from the reference, else 0.4 % of the model's height).
           Thin pieces (threads, embroidery, beads, wisps) automatically get a finer line.
    color: ink colour (default: measured from the reference, else a dark indigo).
    Returns the outline object. Calling it again replaces it. It is skipped by lint and by the exports that
    cannot hide its near side (.obj, .stl, .ply, .bbmodel); .glb, .fbx and .usd carry it.
    """
    clear()
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    if not items:
        raise ValueError("outline: no model geometry")
    found = None
    if width is None or color is None:
        try:
            from . import refkit
            found = refkit.ref().ink
        except Exception:  # noqa: BLE001 - no reference pack in use
            found = None
    bb = core.bbox(items)
    h = max(bb.size.z, 1e-9)
    if width is None:
        width = min(max(found["width"], 0.0025 * h), 0.007 * h) if found else DEFAULT_WIDTH * h
    if color is None:
        color = found["color"] if found else DEFAULT_COLOR
    dg = core.depsgraph()
    vs, ts, off = [], [], 0
    for o in items:
        v, t = core.world_tris(o, dg)
        hull = _hull(v, t, float(width))
        if hull is None:
            continue
        vs.append(hull[0])
        ts.append(hull[1] + off)
        off += len(hull[0])
    if not vs:
        raise ValueError("outline: the model has no faces")
    me = _mesh(name, np.concatenate(vs), np.concatenate(ts))
    me.materials.append(material(color))
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    obj[TAG] = True
    obj["f3d_ink_width"] = float(width)
    bpy.context.view_layer.update()
    return obj
