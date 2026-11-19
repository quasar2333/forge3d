"""Lighting, cameras and rendering. Needs only bpy + numpy."""
import math
import os
import sys
from contextlib import contextmanager

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import core

# view name -> (azimuth, elevation) in degrees. Azimuth 0 = camera in front of the model (at -Y, looking +Y),
# 90 = camera on the model's +X side. Elevation 90 = straight down.
VIEWS = {
    "front": (0, 0), "back": (180, 0), "right": (90, 0), "left": (-90, 0), "top": (0, 90), "bottom": (0, -90),
    "iso": (35, 22), "iso_right": (35, 22), "iso_left": (-35, 22), "iso_back": (145, 22), "iso_back_left": (-145, 22),
    "three_quarter": (35, 12), "hero": (32, 14), "high": (35, 48),
}


@contextmanager
def quiet():
    """Silence Blender's render chatter on stdout."""
    if os.environ.get("F3D_VERBOSE") or not bpy.app.background:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
        os.close(null)


@contextmanager
def safe_shader_names():
    """EEVEE's Metal backend (macOS) aborts the whole process while compiling the shader of a material or world
    whose name is not plain ASCII and that uses textures or attributes. Give such datablocks an ASCII name for the
    duration of a render and put the real name back afterwards."""
    renamed = []
    for coll in (bpy.data.materials, bpy.data.worlds, bpy.data.node_groups):
        for idb in coll:
            if not idb.name.isascii():
                renamed.append((idb, idb.name))
                idb.name = "f3d_tmp_%04d" % len(renamed)
    try:
        yield
    finally:
        for idb, name in renamed:
            try:
                idb.name = name
            except ReferenceError:
                pass


def set_engine(name, scene=None):
    """'EEVEE', 'CYCLES' or 'WORKBENCH' -> whatever this Blender calls it."""
    sc = scene or bpy.context.scene
    key = name.upper().replace("BLENDER_", "")
    cands = {"EEVEE": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"), "EEVEE_NEXT": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"),
             "CYCLES": ("CYCLES",), "WORKBENCH": ("BLENDER_WORKBENCH",)}.get(key)
    if cands is None:
        raise ValueError("engine must be EEVEE, CYCLES or WORKBENCH")
    for c in cands:
        try:
            sc.render.engine = c
            return c
        except TypeError:
            continue
    raise RuntimeError("render engine %s is not available" % name)


def set_view_transform(name="Standard", scene=None):
    sc = scene or bpy.context.scene
    for cand in (name, "Standard"):
        try:
            sc.view_settings.view_transform = cand
            break
        except TypeError:
            continue
    try:
        sc.view_settings.look = "None"
    except TypeError:
        pass


def hdri_path(name):
    if os.path.isfile(core.in_path(name)):
        return core.in_path(name)
    for base in (bpy.utils.resource_path("LOCAL"), bpy.utils.resource_path("SYSTEM")):
        p = os.path.join(base, "datafiles", "studiolights", "world", name + ".exr")
        if os.path.isfile(p):
            return p
    return None


def world(hdri="studio", strength=1.0, bg=None, rotation=0.0, ambient=0.0, name="f3d_world"):
    """Image-based lighting from a bundled HDRI (studio, city, courtyard, forest, interior, night, sunrise,
    sunset) or a .hdr/.exr path. bg: flat colour the camera sees instead of the HDRI (None shows the HDRI).
    ambient: extra uniform white light from all directions (softens shadows; 1.0 lights a surface at its albedo)."""
    sc = bpy.context.scene
    w = bpy.data.worlds.new(name)
    if w.node_tree is None:
        w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    light = nt.nodes.new("ShaderNodeBackground")
    light.inputs["Strength"].default_value = strength
    path = hdri_path(hdri) if hdri else None
    if path:
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(path, check_existing=True)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(rotation))
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], light.inputs["Color"])
    else:
        light.inputs["Color"].default_value = (1, 1, 1, 1)
    lit = light.outputs[0]
    if ambient > 0:
        amb = nt.nodes.new("ShaderNodeBackground")
        amb.inputs["Color"].default_value = (1, 1, 1, 1)
        amb.inputs["Strength"].default_value = ambient
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(light.outputs[0], add.inputs[0])
        nt.links.new(amb.outputs[0], add.inputs[1])
        lit = add.outputs[0]
    if bg is None:
        nt.links.new(lit, out.inputs["Surface"])
    else:
        cam = nt.nodes.new("ShaderNodeBackground")
        cam.inputs["Color"].default_value = core.rgba(bg)
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(lit, mix.inputs[1])
        nt.links.new(cam.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    sc.world = w
    return w


def direction(az, el):
    """Unit vector from the subject toward a camera / light at the given azimuth and elevation (degrees)."""
    a, e = math.radians(az), math.radians(el)
    return Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))


def look_at_matrix(pos, target, roll=0.0, az=0.0):
    pos, target = Vector(pos), Vector(target)
    f = target - pos
    if f.length < 1e-9:
        f = Vector((0, 1, 0))
    f.normalize()
    up = Vector((0, 0, 1))
    if abs(f.z) > 0.999:
        a = math.radians(az)
        up = Vector((-math.sin(a), math.cos(a), 0)) * (1 if f.z < 0 else -1)
    z = -f
    x = up.cross(z)
    x.normalize()
    y = z.cross(x)
    rot = Matrix((x, y, z)).transposed()
    if roll:
        rot = rot @ Matrix.Rotation(math.radians(roll), 3, "Z")
    m = rot.to_4x4()
    m.translation = pos
    return m


def light(kind="AREA", at=(2, -2, 3), look_at=(0, 0, 0), power=500.0, size=1.0, color="#ffffff", name="Light",
          helper=False):
    """Add a light aimed at a point. kind: AREA (soft, default), SUN, POINT, SPOT. power: watts (sun: strength)."""
    data = bpy.data.lights.new(name, kind.upper())
    data.energy = power
    data.color = core.rgba(color)[:3]
    if kind.upper() == "AREA":
        data.shape = "DISK"
        data.size = size
    elif kind.upper() == "SUN":
        data.angle = math.radians(2.0 if size == 1.0 else size)
    elif hasattr(data, "shadow_soft_size"):
        data.shadow_soft_size = size * 0.25
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.matrix_world = look_at_matrix(at, look_at)
    if helper:
        obj["f3d_helper"] = True
        obj["f3d_stage"] = True
    return obj


# preset -> world + lights. Light tuples: (az, el, distance in subject radii, power factor, size in radii, colour)
STAGES = {
    "studio":   dict(hdri="studio", strength=0.3, bg="#e6e7ea", view="Standard",
                     lights=[(-40, 42, 4.5, 1.0, 3.0, "#fff4e6"), (55, 18, 5.0, 0.25, 4.0, "#e6f0ff"),
                             (160, 35, 4.5, 0.6, 2.0, "#ffffff")]),
    "soft":     dict(hdri="studio", strength=0.18, ambient=0.42, bg="#f3f1ec", view="Standard",
                     lights=[(-25, 60, 4.5, 0.45, 5.0, "#ffffff"), (140, 30, 5.0, 0.2, 3.0, "#ffffff")]),
    "dramatic": dict(hdri="night", strength=1.5, bg="#14161b", view="AgX",
                     lights=[(-65, 30, 4.0, 1.6, 1.6, "#ffe1c4"), (150, 25, 4.5, 1.4, 1.5, "#8fb7ff")]),
    "outdoor":  dict(hdri="courtyard", strength=1.0, bg=None, view="AgX",
                     lights=[(-50, 50, 6.0, 0.5, 1.0, "#fff1d6")]),
    "sunset":   dict(hdri="sunset", strength=1.3, bg=None, view="AgX",
                     lights=[(-70, 16, 6.0, 0.8, 1.0, "#ffb070")]),
    "flat":     dict(hdri=None, strength=0.0, ambient=0.75, bg="#dfe3e8", view="Standard",
                     lights=[(-30, 50, 5.0, 0.2, 6.0, "#ffffff")]),
    "product_dark": dict(hdri="studio", strength=0.2, bg="#1b1d22", view="AgX",
                     lights=[(-35, 55, 4.2, 1.1, 3.5, "#ffffff"), (150, 30, 4.5, 1.1, 1.8, "#ffffff")]),
}
# Area-light watts per (metre of distance) squared for a key light. Calibrated in the Standard view transform:
# at factor 1.0 a matte surface facing the key renders at about 0.65 x its albedo, leaving headroom for the HDRI.
KEY_WATTS = 5.5


def clear_stage():
    for o in list(bpy.data.objects):
        if o.get("f3d_stage"):
            bpy.data.objects.remove(o, do_unlink=True)


def _ground(bb, radius, color, seamless=True):
    me = bpy.data.meshes.new("f3d_ground")
    r = radius * 40
    me.from_pydata([(-r, -r, 0), (r, -r, 0), (r, r, 0), (-r, r, 0)], [], [(0, 1, 2, 3)])
    obj = bpy.data.objects.new("f3d_ground", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (bb.center.x, bb.center.y, bb.min.z)
    obj["f3d_helper"] = True
    obj["f3d_stage"] = True
    m = bpy.data.materials.new("f3d_ground")
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
    out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
    bsdf.inputs["Base Color"].default_value = core.rgba(color)
    bsdf.inputs["Roughness"].default_value = 0.9
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.1
            break
    tc = nt.nodes.new("ShaderNodeTexCoord")
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    nt.links.new(tc.outputs["Object"], ln.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = radius * 3.5
    mr.inputs["From Max"].default_value = radius * 12
    nt.links.new(ln.outputs["Value"], mr.inputs["Value"])
    if seamless:
        # far from the subject the floor melts into the flat backdrop colour, so there is never a horizon line
        mr.inputs["To Min"].default_value = 0.0
        mr.inputs["To Max"].default_value = 1.0
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = core.rgba(color)
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(mr.outputs["Result"], mix.inputs["Fac"])
        nt.links.new(bsdf.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    else:
        mr.inputs["To Min"].default_value = 1.0
        mr.inputs["To Max"].default_value = 0.0
        nt.links.new(mr.outputs["Result"], bsdf.inputs["Alpha"])
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
    me.materials.append(m)
    return obj


def stage(preset="studio", objs=None, bg=None, ground=True, hdri=None, strength=None, exposure=1.0, view=None):
    """Light the scene for a presentation render. Replaces any previous stage.

    preset: studio (neutral product look), soft (bright, gentle, good for cute / stylised / low-poly),
            dramatic (dark background, strong key + cool rim), product_dark, outdoor, sunset, flat (even light
            for pixel-art / blocky models).
    bg: background colour override.  ground: floor that catches the contact shadow.
    exposure: multiply all light.  view: 'Standard' (colours as specified) or 'AgX' (filmic, photographic).
    Call after the model is built; follow with k.camera() and k.render().
    """
    if preset not in STAGES:
        raise ValueError("unknown stage preset %r (have: %s)" % (preset, ", ".join(STAGES)))
    cfg = STAGES[preset]
    clear_stage()
    items = core.scene_objects() if objs is None else core.expand(objs)
    bb = core.bbox(items)
    radius = max(bb.diag / 2, 1e-3)
    bgc = bg if bg is not None else cfg["bg"]
    world(cfg["hdri"] if hdri is None else hdri, (cfg["strength"] if strength is None else strength) * exposure, bgc,
          ambient=cfg.get("ambient", 0.0) * exposure)
    for i, (az, el, dist, pw, size, col) in enumerate(cfg["lights"]):
        d = dist * radius
        light("AREA", bb.center + direction(az, el) * d, bb.center, KEY_WATTS * pw * exposure * d * d,
              size * radius, col, name="f3d_light_%d" % i, helper=True)
    if ground:
        _ground(bb, radius, bgc if bgc is not None else "#8a8a86", seamless=bgc is not None)
    sc = bpy.context.scene
    set_view_transform(view or cfg["view"])
    set_engine("EEVEE")
    if hasattr(sc.eevee, "use_shadows"):
        sc.eevee.use_shadows = True
    if hasattr(sc.eevee, "use_raytracing"):
        sc.eevee.use_raytracing = True
    return sc


def _sample_points(objs, limit=6000):
    dg = core.depsgraph()
    chunks = [core.world_verts(o, dg) for o in objs]
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        return np.zeros((1, 3), np.float32)
    pts = np.vstack(chunks)
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def frame_camera(cam, pts, az, el, margin=1.1, roll=0.0, aspect=1.0, ortho_scale=None):
    """Position `cam` at the given angles so that `pts` fill the frame."""
    pts = np.asarray(pts, dtype=np.float64)
    lo, hi = pts.min(0), pts.max(0)
    centre = (lo + hi) / 2
    radius = max(float(np.linalg.norm(hi - lo)) / 2, 1e-6)
    d = direction(az, el)
    m = look_at_matrix(Vector(centre) + d * radius * 10, centre, roll, az)
    rot = np.array(m.to_3x3())
    rel = (pts - centre) @ rot            # camera-space coords: x right, y up, z toward the camera
    data = cam.data
    data.sensor_fit = "HORIZONTAL"
    if data.type == "ORTHO":
        w = rel[:, 0].max() - rel[:, 0].min()
        h = rel[:, 1].max() - rel[:, 1].min()
        mid = np.array([(rel[:, 0].max() + rel[:, 0].min()) / 2, (rel[:, 1].max() + rel[:, 1].min()) / 2, 0.0])
        data.ortho_scale = ortho_scale if ortho_scale is not None else max(w, h * aspect, 1e-6) * margin
        pos = centre + rot @ mid + np.array(d) * (radius * 4 + rel[:, 2].max())
        data.clip_start = 0.001
        data.clip_end = radius * 12 + 10
    else:
        half_h = math.atan(data.sensor_width / 2 / data.lens)
        tan_h = math.tan(half_h)
        tan_v = tan_h / aspect
        mid = np.zeros(3)
        dist = radius * 3
        for _ in range(4):
            q = rel - mid
            need = q[:, 2] + np.maximum(np.abs(q[:, 0]) / tan_h, np.abs(q[:, 1]) / tan_v) * margin
            dist = float(need.max())
            depth = np.maximum(dist - q[:, 2], 1e-6)
            px, py = q[:, 0] / depth, q[:, 1] / depth
            shift = np.array([(px.max() + px.min()) / 2, (py.max() + py.min()) / 2, 0.0]) * dist
            mid = mid + shift
        pos = centre + rot @ mid + np.array(d) * dist
        data.clip_start = max(0.001, dist * 0.01)
        data.clip_end = dist * 20 + radius * 10
    out = m.copy()
    out.translation = Vector(pos.tolist())
    cam.matrix_world = out
    bpy.context.view_layer.update()
    return cam


def camera(view="hero", az=None, el=None, lens=70.0, ortho=False, objs=None, margin=1.15, roll=0.0,
           at=None, look_at=None, res=None, name="Camera"):
    """Create the scene camera, framed on the model.

    view: front, back, left, right, top, iso, iso_left, iso_back, three_quarter, hero, high; or give az / el
          (az 0 = straight in front, 90 = from the model's +X side; el = degrees above the horizon).
    lens: focal length in mm (35 wide, 70 natural product shot, 120 flat / telephoto).  ortho: no perspective.
    at + look_at: place the camera by hand instead.
    The framing follows the render resolution automatically, so the order of k.camera / k.render does not matter.
    """
    sc = bpy.context.scene
    old = bpy.data.objects.get(name)
    if old is not None and old.type == "CAMERA":
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new(name)
    data.type = "ORTHO" if ortho else "PERSP"
    data.lens = lens
    cam = bpy.data.objects.new(name, data)
    sc.collection.objects.link(cam)
    if res is not None:
        sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1])
    if at is not None:
        target = Vector(look_at) if look_at is not None else core.bbox(objs).center
        cam.matrix_world = look_at_matrix(at, target, roll)
    else:
        if az is None and el is None:
            if view not in VIEWS:
                raise ValueError("unknown view %r (have: %s)" % (view, ", ".join(VIEWS)))
            a, e = VIEWS[view]
        else:
            a, e = az or 0.0, el or 0.0
        cam["f3d_frame"] = [float(a), float(e), float(margin), float(roll)]
        if objs is not None:
            cam["f3d_frame_objs"] = [o.name for o in core.expand(objs)]
        _reframe(cam)
    sc.camera = cam
    return cam


def _reframe(cam):
    """(Re)fit an auto-framed camera to the current render aspect ratio."""
    spec = cam.get("f3d_frame")
    if spec is None:
        return
    sc = bpy.context.scene
    aspect = sc.render.resolution_x / sc.render.resolution_y
    names = cam.get("f3d_frame_objs")
    items = [bpy.data.objects[n] for n in names if n in bpy.data.objects] if names else core.scene_objects()
    frame_camera(cam, _sample_points(items), spec[0], spec[1], spec[2], spec[3], aspect)


def render(path, res=None, engine="EEVEE", samples=None, transparent=False, cam=None):
    """Render the scene camera to a PNG. engine: 'EEVEE' (fast, default) or 'CYCLES' (path traced; use for
    glass, liquids, soft global illumination, final hero shots). Returns the path."""
    sc = bpy.context.scene
    if cam is not None:
        sc.camera = core.get(cam)
    if sc.camera is None:
        camera()
    if res is not None:
        sc.render.resolution_x, sc.render.resolution_y = int(res[0]), int(res[1])
    sc.render.resolution_percentage = 100
    _reframe(sc.camera)
    eng = set_engine(engine)
    if eng == "CYCLES":
        sc.cycles.samples = int(samples or 96)
        try:
            sc.cycles.use_denoising = True
        except Exception:
            pass
    elif hasattr(sc.eevee, "taa_render_samples"):
        sc.eevee.taa_render_samples = int(samples or 64)
    sc.render.film_transparent = bool(transparent)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    path = core.out_path(path)
    sc.render.filepath = path
    with quiet(), safe_shader_names():
        bpy.ops.render.render(write_still=True)
    return path


# ---------------------------------------------------------------- evaluation renders (used by the f3d CLI)
@contextmanager
def evaluation_setup(mode="color", hide_helpers=True):
    """Temporarily swap in neutral lighting / shading for objective look-at-it renders, then restore the scene."""
    sc = bpy.context.scene
    vl = bpy.context.view_layer
    saved = dict(engine=sc.render.engine, world=sc.world, film=sc.render.film_transparent,
                 vt=sc.view_settings.view_transform, look=sc.view_settings.look, cam=sc.camera,
                 rx=sc.render.resolution_x, ry=sc.render.resolution_y, pct=sc.render.resolution_percentage,
                 fmt=sc.render.image_settings.file_format, cm=sc.render.image_settings.color_mode,
                 override=vl.material_override, path=sc.render.filepath)
    sh = sc.display.shading
    saved_sh = {k: getattr(sh, k) for k in ("light", "color_type", "single_color", "show_cavity", "show_shadows",
                                             "show_object_outline", "show_specular_highlight") if hasattr(sh, k)}
    hidden, temp = [], []
    for o in sc.objects:
        if o.hide_render:
            continue
        if o.type == "LIGHT" or (hide_helpers and o.get("f3d_helper")) or (
                o.get("f3d_ink") and mode in ("clay", "silhouette", "parts")):     # workbench cannot cull the hull
            o.hide_render = True
            hidden.append(o)
    try:
        sc.render.film_transparent = True
        sc.render.image_settings.file_format = "PNG"
        sc.render.image_settings.color_mode = "RGBA"
        sc.render.resolution_percentage = 100
        set_view_transform("Standard")
        if mode == "albedo":
            # shadeless: uniform white light from everywhere, so a surface renders at its own colour and a
            # comparison with flat artwork measures the colours, not the lighting
            set_engine("EEVEE")
            if hasattr(sc.eevee, "taa_render_samples"):
                sc.eevee.taa_render_samples = 8
            if hasattr(sc.eevee, "use_raytracing"):
                sc.eevee.use_raytracing = False
            if hasattr(sc.eevee, "use_shadows"):
                saved["shadows"] = sc.eevee.use_shadows
                sc.eevee.use_shadows = False
            world(None, 0.0, None, ambient=1.0, name="f3d_eval_world")
            temp.append(sc.world)
        elif mode == "color":
            set_engine("EEVEE")
            if hasattr(sc.eevee, "taa_render_samples"):
                sc.eevee.taa_render_samples = 16
            if hasattr(sc.eevee, "use_raytracing"):
                sc.eevee.use_raytracing = False
            world("studio", 0.2, None, ambient=0.4, name="f3d_eval_world")
            temp.append(sc.world)
            items = core.scene_objects()
            bb = core.bbox(items)
            r = max(bb.diag / 2, 1e-3)
            d = 4.5 * r
            temp.append(light("AREA", bb.center + direction(-35, 45) * d, bb.center, KEY_WATTS * 0.45 * d * d, 3.5 * r,
                              name="f3d_eval_key"))
        else:
            set_engine("WORKBENCH")
            sh.color_type = "SINGLE"
            sh.show_specular_highlight = False
            if mode == "silhouette":
                sh.light = "FLAT"
                sh.single_color = (1, 1, 1)
                sh.show_cavity = sh.show_shadows = sh.show_object_outline = False
            elif mode == "parts":
                sh.light = "FLAT"
                sh.color_type = "RANDOM"
                sh.show_cavity = sh.show_shadows = sh.show_object_outline = False
            else:  # clay
                sh.light = "STUDIO"
                sh.single_color = (0.82, 0.8, 0.77)
                sh.show_cavity = True
                if hasattr(sh, "cavity_type"):
                    sh.cavity_type = "BOTH"
                sh.show_shadows = False
                sh.show_object_outline = True
        yield sc
    finally:
        for o in temp:
            if isinstance(o, bpy.types.World):
                bpy.data.worlds.remove(o)
            else:
                bpy.data.objects.remove(o, do_unlink=True)
        for o in hidden:
            o.hide_render = False
        if "shadows" in saved:
            sc.eevee.use_shadows = saved["shadows"]
        sc.render.engine = saved["engine"]
        sc.world = saved["world"]
        sc.render.film_transparent = saved["film"]
        sc.view_settings.view_transform = saved["vt"]
        try:
            sc.view_settings.look = saved["look"]
        except TypeError:
            pass
        sc.camera = saved["cam"]
        sc.render.resolution_x, sc.render.resolution_y = saved["rx"], saved["ry"]
        sc.render.resolution_percentage = saved["pct"]
        sc.render.image_settings.file_format = saved["fmt"]
        sc.render.image_settings.color_mode = saved["cm"]
        sc.render.filepath = saved["path"]
        vl.material_override = saved["override"]
        for k, v in saved_sh.items():
            try:
                setattr(sh, k, v)
            except Exception:
                pass


def temp_camera(az, el, pts, ortho=True, lens=85.0, margin=1.08, roll=0.0, aspect=1.0, ortho_scale=None):
    data = bpy.data.cameras.new("f3d_eval_cam")
    data.type = "ORTHO" if ortho else "PERSP"
    data.lens = lens
    cam = bpy.data.objects.new("f3d_eval_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    cam["f3d_helper"] = True
    frame_camera(cam, pts, az, el, margin, roll, aspect, ortho_scale)
    return cam


def remove_camera(cam):
    data = cam.data
    bpy.data.objects.remove(cam, do_unlink=True)
    if data.users == 0:
        bpy.data.cameras.remove(data)


def render_tile(cam, path, size=(512, 512)):
    """Render through `cam` inside an evaluation_setup() block. Returns the path of the RGBA PNG."""
    sc = bpy.context.scene
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = int(size[0]), int(size[1])
    sc.render.filepath = path
    with quiet(), safe_shader_names():
        bpy.ops.render.render(write_still=True)
    return path
