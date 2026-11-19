"""Saving, exporting and importing."""
import os

import bpy

from . import core


def save(path):
    """Save the scene as a .blend."""
    path = core.out_path(path)
    bpy.ops.wm.save_as_mainfile(filepath=path, check_existing=False, compress=True)
    return path


def _select(objs):
    for o in bpy.context.scene.objects:
        o.select_set(False)
    items = core.scene_objects() if objs is None else core.expand(objs)
    for o in items:
        o.select_set(True)
    if items:
        bpy.context.view_layer.objects.active = items[0]
    return items


def export(path, objs=None, texture=4096):
    """Export the model (never the stage helpers). Format from the extension:
    .glb / .gltf (engines, web, AR), .fbx (Unity, Unreal), .obj, .stl (3D printing), .usd / .usdc / .usdz, .ply,
    .bbmodel (Blockbench mesh model; bb.save_scene has the options).
    Modifiers are applied in the file. Projection paint (k.paint_from_ref) is baked into one `texture` x `texture`
    image first; other procedural material textures do not survive export, flat PBR values do.
    """
    path = core.out_path(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bbmodel":
        from ..bb import mesh
        return mesh.save_scene(path, objs, texture=texture)
    from . import paint
    paint.bake(objs, size=texture)
    items = _select(objs)
    if not items:
        raise ValueError("export: no model geometry to export")
    if objs is None and ext in (".glb", ".gltf", ".fbx", ".usd", ".usdc", ".usda", ".usdz"):
        for o in core.scene_objects(ink=True):      # line art (k.outline) travels with formats that cull back faces
            if o.get("f3d_ink"):
                o.select_set(True)
                items.append(o)
    undo = paint.plain_material_for_export(items)
    from .staging import quiet
    try:
        _write(path, ext)
    finally:
        undo()
    return path


def _write(path, ext):
    from .staging import quiet
    with quiet():
        if ext in (".glb", ".gltf"):
            bpy.ops.export_scene.gltf(filepath=path, export_format="GLB" if ext == ".glb" else "GLTF_SEPARATE",
                                      use_selection=True, export_apply=True, export_yup=True)
        elif ext == ".fbx":
            bpy.ops.export_scene.fbx(filepath=path, use_selection=True, use_mesh_modifiers=True,
                                     apply_scale_options="FBX_SCALE_UNITS", mesh_smooth_type="FACE",
                                     path_mode="COPY", embed_textures=True)
        elif ext == ".obj":
            bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, apply_modifiers=True)
        elif ext == ".stl":
            bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True, apply_modifiers=True)
        elif ext == ".ply":
            bpy.ops.wm.ply_export(filepath=path, export_selected_objects=True, apply_modifiers=True)
        elif ext in (".usd", ".usdc", ".usda", ".usdz"):
            bpy.ops.wm.usd_export(filepath=path, selected_objects_only=True)
        else:
            raise ValueError("unsupported export format %r (glb, gltf, fbx, obj, stl, ply, usd, usdz)" % ext)
    return path


def import_mesh(path, name=None):
    """Import a model file (.glb, .gltf, .obj, .fbx, .stl, .ply, .usd*, .blend). Returns the new root objects."""
    path = core.in_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    before = set(o.name for o in bpy.data.objects)
    from .staging import quiet
    with quiet():
        if ext in (".glb", ".gltf"):
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)
        elif ext == ".fbx":
            if hasattr(bpy.ops.wm, "fbx_import"):
                bpy.ops.wm.fbx_import(filepath=path)
            else:
                bpy.ops.import_scene.fbx(filepath=path)
        elif ext == ".stl":
            bpy.ops.wm.stl_import(filepath=path)
        elif ext == ".ply":
            bpy.ops.wm.ply_import(filepath=path)
        elif ext in (".usd", ".usdc", ".usda", ".usdz"):
            bpy.ops.wm.usd_import(filepath=path)
        elif ext == ".blend":
            with bpy.data.libraries.load(path) as (src, dst):
                dst.objects = list(src.objects)
            for o in dst.objects:
                if o is not None:
                    bpy.context.scene.collection.objects.link(o)
        else:
            raise ValueError("unsupported import format %r" % ext)
    new = [o for o in bpy.data.objects if o.name not in before]
    for o in new:
        o.select_set(False)
    top = [o for o in new if o.parent is None or o.parent not in new]
    if name and len(top) == 1:
        top[0].name = name
    bpy.context.view_layer.update()
    return top
