"""Load a model source (build script, .blend, mesh file, .bbmodel) into the current Blender session."""
import os
import runpy
import sys
import traceback

import bpy

from . import apihints
from .kit import core

MESH_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usd", ".usdc", ".usda", ".usdz")


class BuildError(Exception):
    pass


def run_script(path, quiet_errors=False, fresh=True):
    """Execute a build script from an empty scene (fresh=False: in the scene as it is). Raises BuildError with a
    readable report on failure."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise BuildError("build script not found: %s" % path)
    if fresh:
        core.reset()
    from .bb import model as bbmodel
    del bbmodel.SAVED[:]
    bbmodel.PLAN.clear()
    bbmodel.PART["model"] = None
    argv = sys.argv
    sys.argv = [path]
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise BuildError("script exited with code %r" % (e.code,))
    except BaseException as e:  # noqa: BLE001 - report every failure of user code
        if quiet_errors:
            raise BuildError("%s: %s" % (type(e).__name__, e))
        raise BuildError(format_error(path, e))
    finally:
        sys.argv = argv
    bpy.context.view_layer.update()
    from .bb import model as bbmodel
    if bbmodel.SAVED and not core.scene_objects():
        from .bb import to_blender
        to_blender.load(bbmodel.SAVED[-1])
        bpy.context.view_layer.update()


def format_error(script, exc):
    tb = traceback.extract_tb(exc.__traceback__)
    lines = ["BUILD FAILED: %s: %s" % (type(exc).__name__, exc)]
    user = [f for f in tb if os.path.abspath(f.filename) == script]
    for f in user:
        lines.append("  %s:%d  %s" % (os.path.basename(f.filename), f.lineno, (f.line or "").strip()))
    kit_frames = [f for f in tb if os.sep + "forge3d" + os.sep in f.filename]
    if kit_frames:
        f = kit_frames[-1]
        lines.append("  raised inside forge3d %s:%d in %s()" % (os.path.basename(f.filename), f.lineno, f.name))
    if not user:
        lines.extend("  " + ln for ln in traceback.format_exception_only(type(exc), exc))
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for h in apihints.hints_for(text):
        lines.append("  HINT: " + h)
    return "\n".join(lines)


def load_source(path, ref=None):
    """Populate the scene from any supported source. Returns the kind: 'script', 'blend', 'mesh', 'bbmodel'.
    ref: reference pack folder that k.ref() returns when the script names none."""
    path = os.path.abspath(path)
    from .kit import refkit
    refkit.set_default(ref)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        run_script(path)
        return "script"
    if not os.path.isfile(path):
        raise BuildError("file not found: %s" % path)
    if ext == ".blend":
        bpy.ops.wm.open_mainfile(filepath=path)
        sc = bpy.context.scene
        if not sc.objects and bpy.data.objects:          # a part file (written by `f3d part`) holds bare objects
            for o in bpy.data.objects:
                sc.collection.objects.link(o)
            bpy.context.view_layer.update()
        return "blend"
    if ext in MESH_EXTS:
        from .kit import io
        core.reset()
        io.import_mesh(path)
        return "mesh"
    if ext == ".bbmodel":
        from .bb import to_blender
        core.reset()
        to_blender.load(path)
        return "bbmodel"
    raise BuildError("unsupported source %r (use a build script .py, .blend, .bbmodel or a mesh file)" % ext)
