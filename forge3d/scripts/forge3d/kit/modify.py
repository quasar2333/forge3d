"""Modifiers, booleans, duplication patterns, joining. Needs only bpy + numpy."""
import math
import re

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import core
from .shapes import finish, shade


def _mesh_objs(objs):
    return [o for o in core.expand(objs) if o.type == "MESH"]


def bevel(objs, width=0.01, segs=3, angle=30.0, harden=True):
    """Bevel modifier on edges sharper than `angle`. The cheapest way to make hard-surface parts catch light."""
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Bevel", "BEVEL")
        m.width = width
        m.segments = int(segs)
        m.limit_method = "ANGLE"
        m.angle_limit = math.radians(angle)
        m.miter_outer = "MITER_ARC"
        if harden and hasattr(m, "harden_normals"):
            m.harden_normals = True
    return objs


def subdiv(objs, levels=2, simple=False):
    """Subdivision surface (smooths and rounds; add loops or bevels first where edges must stay tight)."""
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Subdivision", "SUBSURF")
        m.levels = m.render_levels = int(levels)
        if simple:
            m.subdivision_type = "SIMPLE"
    return objs


def solidify(objs, thickness=0.01, offset=-1.0):
    """Give a surface thickness. offset -1 grows inward, +1 outward, 0 both ways."""
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Solidify", "SOLIDIFY")
        m.thickness = thickness
        m.offset = offset
    return objs


def displace(objs, strength=0.02, scale=0.5, kind="CLOUDS", midlevel=0.5):
    """Noise displacement for rocks, terrain, cloth wrinkles, organic irregularity. Needs enough vertices
    (subdiv or remesh first). kind: CLOUDS, VORONOI, MUSGRAVE-like 'STUCCI', 'WOOD', 'MARBLE'."""
    tex = bpy.data.textures.new("f3d_displace", kind)
    if hasattr(tex, "noise_scale"):
        tex.noise_scale = scale
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Displace", "DISPLACE")
        m.texture = tex
        m.strength = strength
        m.mid_level = midlevel
        m.texture_coords = "GLOBAL"
    return objs


def _simple_deform(objs, method, axis, **props):
    for o in _mesh_objs(objs):
        m = o.modifiers.new(method.title(), "SIMPLE_DEFORM")
        m.deform_method = method
        m.deform_axis = axis.upper()
        for k, v in props.items():
            setattr(m, k, v)
    return objs


def bend(objs, angle=45.0, axis="Z"):
    """Bend around an axis by `angle` degrees (needs segments along the bend direction)."""
    return _simple_deform(objs, "BEND", axis, angle=math.radians(angle))


def twist(objs, angle=45.0, axis="Z"):
    """Twist along an axis by `angle` degrees."""
    return _simple_deform(objs, "TWIST", axis, angle=math.radians(angle))


def taper(objs, factor=0.5, axis="Z"):
    """Taper along an axis: positive widens toward +axis, negative narrows."""
    return _simple_deform(objs, "TAPER", axis, factor=factor)


def remesh(objs, voxel=0.02, smooth=True):
    """Voxel remesh: fuses overlapping shells into one watertight skin (good after joining organic parts)."""
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Remesh", "REMESH")
        m.mode = "VOXEL"
        m.voxel_size = voxel
        m.use_smooth_shade = bool(smooth)
    return objs


def smooth_mesh(objs, factor=0.5, iterations=5):
    """Relax the surface (removes voxel stair-steps and lumps)."""
    for o in _mesh_objs(objs):
        m = o.modifiers.new("Smooth", "SMOOTH")
        m.factor = factor
        m.iterations = int(iterations)
    return objs


def decimate(objs, ratio=None, faces=None):
    """Reduce polygon count to a ratio (0-1) or to roughly `faces` triangles per object."""
    for o in _mesh_objs(objs):
        r = ratio
        if faces is not None:
            _, tris = core.world_tris(o)
            r = min(1.0, faces / max(1, len(tris)))
        if r is None or r >= 1.0:
            continue
        m = o.modifiers.new("Decimate", "DECIMATE")
        m.ratio = max(0.001, r)
    return objs


def apply(objs=None):
    """Bake all modifiers into real geometry (do this before export-sensitive steps or mirroring by data)."""
    items = _mesh_objs(objs) if objs is not None else [o for o in core.scene_objects() if o.type == "MESH"]
    dg = core.depsgraph()
    for o in items:
        if not o.modifiers:
            continue
        new = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        old = o.data
        o.modifiers.clear()
        o.data = new
        new.name = old.name
        if old.users == 0:
            bpy.data.meshes.remove(old)
    bpy.context.view_layer.update()
    return objs


def apply_scale(objs=None):
    """Bake object scale into the mesh (bevel widths and exports behave after non-uniform scaling)."""
    items = _mesh_objs(objs) if objs is not None else [o for o in core.scene_objects() if o.type == "MESH"]
    for o in items:
        s = o.scale.copy()
        if (s - Vector((1, 1, 1))).length < 1e-7:
            continue
        if o.data.users > 1:
            o.data = o.data.copy()
        o.data.transform(Matrix.Diagonal((s.x, s.y, s.z, 1.0)))
        if s.x * s.y * s.z < 0:
            o.data.flip_normals()
        o.scale = (1, 1, 1)
        for ch in o.children:
            ch.matrix_parent_inverse = Matrix.Diagonal((s.x, s.y, s.z, 1.0)) @ ch.matrix_parent_inverse
    bpy.context.view_layer.update()
    return objs


def boolean(target, cutter, op="DIFFERENCE", keep_cutter=False, solver="EXACT"):
    """Boolean `cutter` into `target` (DIFFERENCE, UNION or INTERSECT) and bake it. The cutter is removed
    unless keep_cutter=True. Both meshes should be closed; give the cutter a little overshoot past the surface."""
    target = core.get(target)
    cutters = _mesh_objs(cutter)
    for c in cutters:
        m = target.modifiers.new("Boolean", "BOOLEAN")
        m.operation = op.upper()
        m.object = c
        try:
            m.solver = solver
        except TypeError:
            pass
    apply(target)
    for c in cutters:
        if keep_cutter:
            c.hide_render = True
            c.hide_viewport = True
            c["f3d_helper"] = True
        else:
            bpy.data.objects.remove(c, do_unlink=True)
    return target


def cut(target, cutter, **kw):
    """Subtract `cutter` from `target`."""
    return boolean(target, cutter, "DIFFERENCE", **kw)


def union(target, other, **kw):
    """Fuse `other` into `target`."""
    return boolean(target, other, "UNION", **kw)


def intersect(target, other, **kw):
    """Keep only the overlap of the two."""
    return boolean(target, other, "INTERSECT", **kw)


_SWAPS = [(r"\.L$", ".R"), (r"\.R$", ".L"), (r"_L$", "_R"), (r"_R$", "_L"), (r"\.l$", ".r"), (r"\.r$", ".l"),
          (r"Left", "Right"), (r"Right", "Left"), (r"left", "right"), (r"right", "left")]


def _mirror_name(name):
    for pat, rep in _SWAPS:
        if re.search(pat, name):
            return re.sub(pat, rep, name, count=1)
    return name + ".mirror"


def mirror(objs, axis="X", plane=0.0):
    """Mirrored copy of object(s) across the plane `axis = plane` (default the X = 0 symmetry plane).

    Returns the new object (or list). Names swap .L/.R, _L/_R, Left/Right. Build one side, then mirror it.
    """
    a = "XYZ".index(axis.upper())
    single = isinstance(objs, (bpy.types.Object, str))
    refl = Matrix.Identity(4)
    refl[a][a] = -1.0
    refl[a][3] = 2.0 * plane
    dg = core.depsgraph()
    out = []
    for o in _mesh_objs(objs):
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        me.transform(refl @ o.matrix_world)
        me.flip_normals()
        new = bpy.data.objects.new(_mirror_name(o.name), me)
        core.link(new)
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            c = (co.min(0) + co.max(0)) / 2
            me.vertices.foreach_set("co", (co - c).ravel())
            me.update()
            new.location = Vector(c.tolist())
        for key in o.keys():
            if key.startswith("f3d_"):
                new[key] = o[key]
        out.append(new)
    bpy.context.view_layer.update()
    return out[0] if single and len(out) == 1 else out


def array(obj, count, offset=(1, 0, 0)):
    """`count` copies in a row, each shifted by `offset` from the previous (cheap linked instances).
    Returns all of them including the original."""
    obj = core.get(obj)
    off = Vector(offset)
    out = [obj]
    for i in range(1, int(count)):
        c = core.dup(obj, name="%s.%02d" % (obj.name, i), linked=True)
        core.translate(c, off * i)
        out.append(c)
    return out


def radial(obj, count, axis="Z", center=(0, 0, 0), angle=360.0):
    """`count` copies evenly spaced around an axis through `center` (spokes, petals, bolts, legs, teeth).
    Returns all of them including the original."""
    obj = core.get(obj)
    step = angle / int(count) if abs(angle - 360.0) < 1e-6 else angle / max(1, int(count) - 1)
    out = [obj]
    for i in range(1, int(count)):
        c = core.dup(obj, name="%s.%02d" % (obj.name, i), linked=True)
        rot = [0.0, 0.0, 0.0]
        rot["XYZ".index(axis.upper())] = step * i
        core.rotate(c, rot, about=center)
        out.append(c)
    return out


def join(objs, name=None, smooth=None):
    """Merge several objects into one mesh (modifiers are applied, materials kept)."""
    items = _mesh_objs(objs)
    if not items:
        raise ValueError("join: nothing to join")
    dg = core.depsgraph()
    bm = bmesh.new()
    mats = []
    for o in items:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        mw = o.matrix_world
        me.transform(mw)
        if mw.determinant() < 0:
            me.flip_normals()
        remap = {}
        for i, m in enumerate(me.materials):
            if m not in mats:
                mats.append(m)
            remap[i] = mats.index(m)
        before = len(bm.faces)
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        for f in bm.faces[before:]:
            f.material_index = remap.get(f.material_index, 0)
        bpy.data.meshes.remove(me)
    name = name or items[0].name
    for o in items:
        bpy.data.objects.remove(o, do_unlink=True)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    obj = bpy.data.objects.new(name, me)
    core.link(obj)
    n = len(me.vertices)
    if n:
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        c = (co.min(0) + co.max(0)) / 2
        me.vertices.foreach_set("co", (co - c).ravel())
        me.update()
        obj.location = Vector(c.tolist())
    if smooth is not None:
        shade(obj, smooth)
    bpy.context.view_layer.update()
    return obj
