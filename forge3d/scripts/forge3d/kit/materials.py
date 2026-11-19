"""Materials: physically sensible presets plus a few procedural textures. Needs only bpy."""
import math

import bpy
from mathutils import Vector

from . import core

# name -> Principled settings. `color` is only a fallback when the caller gives none.
PRESETS = {
    "plastic":        dict(rough=0.42),
    "glossy_plastic": dict(rough=0.15, coat=0.4, coat_rough=0.05),
    "matte":          dict(rough=0.95, specular=0.2),
    "rubber":         dict(rough=0.85, specular=0.3),
    "paint":          dict(rough=0.35, coat=0.3, coat_rough=0.1),
    "car_paint":      dict(rough=0.3, metal=0.35, coat=1.0, coat_rough=0.03),
    "ceramic":        dict(rough=0.18, coat=0.6, coat_rough=0.03),
    "metal":          dict(metal=1.0, rough=0.35, color="#b8bcc2"),
    "steel":          dict(metal=1.0, rough=0.3, color="#8f9499"),
    "brushed_metal":  dict(metal=1.0, rough=0.38, aniso=0.6, color="#b8bcc2", texture="brushed"),
    "chrome":         dict(metal=1.0, rough=0.04, color="#e6e8ea"),
    "aluminum":       dict(metal=1.0, rough=0.25, color="#e9ebec"),
    "iron":           dict(metal=1.0, rough=0.55, color="#6f7174"),
    "gold":           dict(metal=1.0, rough=0.18, color="#ffc355"),
    "copper":         dict(metal=1.0, rough=0.25, color="#f3a383"),
    "brass":          dict(metal=1.0, rough=0.28, color="#e3b770"),
    "glass":          dict(rough=0.02, transmission=1.0, ior=1.45, color="#ffffff"),
    "frosted_glass":  dict(rough=0.35, transmission=1.0, ior=1.45, color="#ffffff"),
    "tinted_glass":   dict(rough=0.03, transmission=1.0, ior=1.45),
    "water":          dict(rough=0.02, transmission=1.0, ior=1.33, color="#dff3ff"),
    "wood":           dict(rough=0.55, texture="wood", color="#a9744a"),
    "fabric":         dict(rough=0.9, sheen=0.6, specular=0.3, bump=0.25, bump_scale=220.0),
    "velvet":         dict(rough=1.0, sheen=1.0, specular=0.1),
    "leather":        dict(rough=0.5, bump=0.35, bump_scale=90.0),
    "skin":           dict(rough=0.5, subsurface=0.25, color="#e8b89a"),
    "wax":            dict(rough=0.4, subsurface=0.6),
    "stone":          dict(rough=0.8, bump=0.6, bump_scale=14.0, variation=0.12, color="#8d8a84"),
    "concrete":       dict(rough=0.9, bump=0.3, bump_scale=40.0, variation=0.08, color="#a7a6a2"),
    "clay":           dict(rough=0.7, specular=0.3, color="#c9c5bd"),
    "toon":           dict(rough=1.0, specular=0.0),
    "emissive":       dict(rough=0.5, emit_strength=4.0),
    "screen":         dict(rough=0.1, emit_strength=1.5),
}

_SOCKETS = {
    "color": ("Base Color",), "metal": ("Metallic",), "rough": ("Roughness",), "ior": ("IOR",), "alpha": ("Alpha",),
    "transmission": ("Transmission Weight", "Transmission"), "coat": ("Coat Weight", "Clearcoat"),
    "coat_rough": ("Coat Roughness", "Clearcoat Roughness"), "sheen": ("Sheen Weight", "Sheen"),
    "subsurface": ("Subsurface Weight", "Subsurface"), "specular": ("Specular IOR Level", "Specular"),
    "aniso": ("Anisotropic",), "emit": ("Emission Color", "Emission"), "emit_strength": ("Emission Strength",),
}


def _sock(node, key):
    for n in _SOCKETS[key]:
        if n in node.inputs:
            return node.inputs[n]
    return None


def _principled(m):
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    out = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    return nt, bsdf


def _coords(nt, scale=1.0):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    s = core.vec3(scale)
    mp.inputs["Scale"].default_value = s
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    return mp.outputs["Vector"]


def _ramp(nt, fac, c1, c2, p1=0.3, p2=0.7):
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.color_ramp.elements[0].position, r.color_ramp.elements[0].color = p1, c1
    r.color_ramp.elements[1].position, r.color_ramp.elements[1].color = p2, c2
    nt.links.new(fac, r.inputs["Fac"])
    return r.outputs["Color"]


def _darker(c, f=0.45):
    return (c[0] * f, c[1] * f, c[2] * f, 1.0)


def mat(color=None, preset="plastic", name=None, color2=None, texture=None, tex_scale=None,
        bump=None, bump_scale=None, variation=None, **props):
    """Create (or fetch by name) a material.

    color:   '#rrggbb' / colour name (sRGB) or a linear (r, g, b) tuple.
    preset:  one of k.MATERIALS, e.g. plastic, glossy_plastic, matte, rubber, paint, car_paint, ceramic, metal,
             steel, brushed_metal, chrome, aluminum, iron, gold, copper, brass, glass, frosted_glass, tinted_glass,
             water, wood, fabric, velvet, leather, skin, wax, stone, concrete, clay, toon, emissive, screen.
    overrides: rough, metal, specular, coat, coat_rough, sheen, subsurface, transmission, ior, alpha, aniso,
             emit (colour), emit_strength.
    texture: procedural pattern mixing `color` with `color2`: 'wood', 'marble', 'brushed', 'checker', 'stripes',
             'speckle', 'gradient' (bottom = color, top = color2), 'noise'. tex_scale sets its density.
    bump / bump_scale: fine surface relief.  variation: 0-0.3 subtle brightness mottling (kills the CG-flat look).
    Metals: metal is 1 or 0, never in between; their colour is the reflection tint.
    """
    if name and name in bpy.data.materials:
        return bpy.data.materials[name]
    if preset not in PRESETS:
        raise ValueError("unknown material preset %r (have: %s)" % (preset, ", ".join(sorted(PRESETS))))
    cfg = dict(PRESETS[preset])
    if color is None:
        color = cfg.get("color", "#b0b0b0")
    cfg.pop("color", None)
    texture = texture if texture is not None else cfg.pop("texture", None)
    cfg.pop("texture", None)
    bump = bump if bump is not None else cfg.pop("bump", None)
    bump_scale = bump_scale if bump_scale is not None else cfg.pop("bump_scale", 60.0)
    variation = variation if variation is not None else cfg.pop("variation", None)
    for k in ("bump", "bump_scale", "variation"):
        cfg.pop(k, None)
    for k, v in props.items():
        if k not in _SOCKETS:
            raise TypeError("mat() got an unknown property %r (known: %s)" % (k, ", ".join(sorted(_SOCKETS))))
        cfg[k] = v
    col = core.rgba(color)
    m = bpy.data.materials.new(name or "%s_%s" % (preset, core.hex_of(color).lstrip("#")))
    nt, bsdf = _principled(m)
    bsdf.inputs["Base Color"].default_value = col
    m.diffuse_color = col
    for key, val in cfg.items():
        s = _sock(bsdf, key)
        if s is None:
            continue
        if key == "emit":
            s.default_value = core.rgba(val)
        else:
            s.default_value = val
    if "emit_strength" in cfg and "emit" not in cfg:
        s = _sock(bsdf, "emit")
        if s is not None:
            s.default_value = col
    m["f3d_color"] = core.hex_of(color)
    m["f3d_preset"] = preset

    color_out = None
    c2 = core.rgba(color2) if color2 is not None else _darker(col)
    if texture:
        t = texture.lower()
        if t == "wood":
            vec = _coords(nt, (tex_scale or 1.0))
            w = nt.nodes.new("ShaderNodeTexWave")
            w.wave_type = "BANDS"
            w.bands_direction = "X"
            w.inputs["Scale"].default_value = 6.0
            w.inputs["Distortion"].default_value = 7.0
            w.inputs["Detail"].default_value = 3.0
            w.inputs["Detail Scale"].default_value = 1.5
            nt.links.new(vec, w.inputs["Vector"])
            color_out = _ramp(nt, w.outputs["Fac"], c2, col, 0.15, 0.85)
            bump = bump if bump is not None else 0.15
        elif t == "marble":
            vec = _coords(nt, (tex_scale or 1.0))
            w = nt.nodes.new("ShaderNodeTexWave")
            w.inputs["Scale"].default_value = 1.5
            w.inputs["Distortion"].default_value = 12.0
            w.inputs["Detail"].default_value = 4.0
            nt.links.new(vec, w.inputs["Vector"])
            color_out = _ramp(nt, w.outputs["Fac"], col, c2, 0.55, 0.95)
        elif t == "brushed":
            vec = _coords(nt, (1.0, 1.0, 60.0) if tex_scale is None else tex_scale)
            n = nt.nodes.new("ShaderNodeTexNoise")
            n.inputs["Scale"].default_value = 40.0
            n.inputs["Detail"].default_value = 6.0
            nt.links.new(vec, n.inputs["Vector"])
            mr = nt.nodes.new("ShaderNodeMapRange")
            base = bsdf.inputs["Roughness"].default_value
            mr.inputs["To Min"].default_value = max(0.0, base - 0.12)
            mr.inputs["To Max"].default_value = min(1.0, base + 0.12)
            nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])
        elif t == "checker":
            vec = _coords(nt, 1.0)
            ch = nt.nodes.new("ShaderNodeTexChecker")
            ch.inputs["Scale"].default_value = tex_scale or 8.0
            ch.inputs["Color1"].default_value = col
            ch.inputs["Color2"].default_value = c2
            nt.links.new(vec, ch.inputs["Vector"])
            color_out = ch.outputs["Color"]
        elif t == "stripes":
            vec = _coords(nt, 1.0)
            w = nt.nodes.new("ShaderNodeTexWave")
            w.wave_type = "BANDS"
            w.bands_direction = "Z"
            w.inputs["Scale"].default_value = tex_scale or 4.0
            nt.links.new(vec, w.inputs["Vector"])
            color_out = _ramp(nt, w.outputs["Fac"], col, c2, 0.49, 0.51)
        elif t == "speckle":
            vec = _coords(nt, 1.0)
            v = nt.nodes.new("ShaderNodeTexVoronoi")
            v.inputs["Scale"].default_value = tex_scale or 40.0
            nt.links.new(vec, v.inputs["Vector"])
            color_out = _ramp(nt, v.outputs["Distance"], c2, col, 0.05, 0.25)
        elif t == "noise":
            vec = _coords(nt, 1.0)
            n = nt.nodes.new("ShaderNodeTexNoise")
            n.inputs["Scale"].default_value = tex_scale or 5.0
            n.inputs["Detail"].default_value = 5.0
            nt.links.new(vec, n.inputs["Vector"])
            color_out = _ramp(nt, n.outputs["Fac"], col, c2, 0.35, 0.65)
        elif t == "gradient":
            tc = nt.nodes.new("ShaderNodeTexCoord")
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            nt.links.new(tc.outputs["Generated"], sep.inputs["Vector"])
            color_out = _ramp(nt, sep.outputs["Z"], col, c2, 0.0, 1.0)
        else:
            raise ValueError("unknown texture %r (wood, marble, brushed, checker, stripes, speckle, noise, gradient)"
                             % (texture,))
    if variation:
        vec = _coords(nt, 1.0)
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 3.0
        n.inputs["Detail"].default_value = 6.0
        nt.links.new(vec, n.inputs["Vector"])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = 1.0 - variation
        mr.inputs["To Max"].default_value = 1.0 + variation
        nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
        hsv = nt.nodes.new("ShaderNodeHueSaturation")
        nt.links.new(mr.outputs["Result"], hsv.inputs["Value"])
        if color_out is not None:
            nt.links.new(color_out, hsv.inputs["Color"])
        else:
            hsv.inputs["Color"].default_value = col
        color_out = hsv.outputs["Color"]
    if color_out is not None:
        nt.links.new(color_out, bsdf.inputs["Base Color"])
    if bump:
        vec = _coords(nt, 1.0)
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = bump_scale
        n.inputs["Detail"].default_value = 4.0
        nt.links.new(vec, n.inputs["Vector"])
        b = nt.nodes.new("ShaderNodeBump")
        b.inputs["Strength"].default_value = bump
        b.inputs["Distance"].default_value = 0.01
        nt.links.new(n.outputs["Fac"], b.inputs["Height"])
        nt.links.new(b.outputs["Normal"], bsdf.inputs["Normal"])

    transmission = cfg.get("transmission", 0.0)
    alpha = cfg.get("alpha", 1.0)
    if transmission > 0 or alpha < 1:
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "DITHERED" if transmission > 0 else "BLENDED"
        if hasattr(m, "use_raytrace_refraction"):
            m.use_raytrace_refraction = transmission > 0
        if hasattr(m, "blend_method") and alpha < 1:
            try:
                m.blend_method = "BLEND"
            except Exception:
                pass
    return m


def as_material(m):
    """Material, material name, or a colour -> Material."""
    if isinstance(m, bpy.types.Material):
        return m
    if isinstance(m, str) and m in bpy.data.materials:
        return bpy.data.materials[m]
    return mat(m)


def assign(objs, material):
    """Give object(s) (and their children) a single material. `material` may be a colour."""
    m = as_material(material)
    for o in core.expand(objs):
        if hasattr(o.data, "materials"):
            o.data.materials.clear()
            o.data.materials.append(m)
    return objs


_FACING = {"top": (0, 0, 1), "bottom": (0, 0, -1), "front": (0, -1, 0), "back": (0, 1, 0),
           "left": (-1, 0, 0), "right": (1, 0, 0)}


def assign_faces(obj, material, where):
    """Second (third, ...) material on part of one mesh.

    where: 'top' / 'bottom' / 'front' / 'back' / 'left' / 'right' / 'sides' (faces whose normal points that way), or
           a function (centre: Vector, normal: Vector) -> bool, both in world space, e.g.
           lambda c, n: c.z > 0.4 and n.y < 0
    Applies to the base mesh, so call it before k.apply() if modifiers reshape the object a lot.
    """
    obj = core.get(obj)
    m = as_material(material)
    me = obj.data
    if len(me.materials) == 0:
        me.materials.append(mat("#b0b0b0", name="f3d_default"))
    if m.name not in [x.name for x in me.materials if x]:
        me.materials.append(m)
    idx = [x.name if x else None for x in me.materials].index(m.name)
    if isinstance(where, str):
        key = where.lower()
        if key == "sides":
            fn = lambda c, n: abs(n.z) < 0.5
        elif key in _FACING:
            d = Vector(_FACING[key])
            fn = lambda c, n: n.dot(d) > 0.5
        else:
            raise ValueError("where must be a function or one of top/bottom/front/back/left/right/sides")
    else:
        fn = where
    mw = obj.matrix_world
    nm = mw.to_3x3().inverted_safe().transposed()
    count = 0
    for p in me.polygons:
        if fn(mw @ p.center, (nm @ p.normal).normalized()):
            p.material_index = idx
            count += 1
    me.update()
    if count == 0:
        print("[f3d] warning: assign_faces(%s, %s) matched no faces" % (obj.name, m.name))
    return obj


def image_mat(path, name=None, pixelated=False, rough=0.6, emit=0.0):
    """Material showing an image through the object's UV map. pixelated=True keeps pixel art crisp."""
    path = core.in_path(path)
    img = bpy.data.images.load(path, check_existing=True)
    m = bpy.data.materials.new(name or "image_" + img.name)
    nt, bsdf = _principled(m)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest" if pixelated else "Linear"
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = rough
    if emit:
        s = _sock(bsdf, "emit")
        nt.links.new(tex.outputs["Color"], s)
        bsdf.inputs["Emission Strength"].default_value = emit
    if hasattr(m, "surface_render_method"):
        m.surface_render_method = "DITHERED"
    m["f3d_image"] = path
    return m


def project_image(objs, image, view="front", frame=None, back_color=None, name=None, rough=0.6):
    """Paint object(s) with an image projected straight along a view, like a slide projector.

    Use it for a single picture you want on a part: a logo, label, screen content, a face drawn elsewhere.
    (To colour the model from the reference views use k.paint_from_ref, which handles several views, occlusion
    and unseen sides.)
    view: 'front', 'back', 'left', 'right' or 'top'.
    frame: (umin, vmin, umax, vmax) world extents the image should cover; default = bounding box of `objs`.
    back_color: colour for surfaces facing away from the projector (default: the image shows through mirrored).
    """
    items = [o for o in core.expand(objs) if o.type == "MESH"]
    image = core.in_path(image)
    img = bpy.data.images.load(image, check_existing=True)
    axes = {"front": (0, 2, (0, -1, 0), 1), "back": (0, 2, (0, 1, 0), -1), "right": (1, 2, (1, 0, 0), 1),
            "left": (1, 2, (-1, 0, 0), -1), "top": (0, 1, (0, 0, 1), 1)}
    if view not in axes:
        raise ValueError("view must be one of %s" % ", ".join(axes))
    ua, va, normal, usign = axes[view]
    if frame is None:
        bb = core.bbox(items)
        frame = (bb.min[ua], bb.min[va], bb.max[ua], bb.max[va])
    u0, v0, u1, v1 = frame
    du, dv = max(u1 - u0, 1e-9), max(v1 - v0, 1e-9)
    m = bpy.data.materials.new(name or "projected_" + img.name)
    nt, bsdf = _principled(m)
    uvn = nt.nodes.new("ShaderNodeUVMap")
    uvn.uv_map = "f3d_proj"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = "EXTEND"
    nt.links.new(uvn.outputs["UV"], tex.inputs["Vector"])
    bsdf.inputs["Roughness"].default_value = rough
    color_out = tex.outputs["Color"]
    if back_color is not None:
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        dot.inputs[1].default_value = normal
        nt.links.new(geo.outputs["True Normal"], dot.inputs[0])
        gt = nt.nodes.new("ShaderNodeMath")
        gt.operation = "GREATER_THAN"
        gt.inputs[1].default_value = -0.05
        nt.links.new(dot.outputs["Value"], gt.inputs[0])
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(gt.outputs["Value"], mix.inputs["Factor"])
        mix.inputs["A"].default_value = core.rgba(back_color)
        nt.links.new(tex.outputs["Color"], mix.inputs["B"])
        color_out = mix.outputs["Result"]
    nt.links.new(color_out, bsdf.inputs["Base Color"])
    m["f3d_image"] = image
    for o in items:
        me = o.data
        uv = me.uv_layers.get("f3d_proj") or me.uv_layers.new(name="f3d_proj")
        mw = o.matrix_world
        for loop in me.loops:
            w = mw @ me.vertices[loop.vertex_index].co
            u = (w[ua] - u0) / du
            if usign < 0:
                u = 1.0 - u
            uv.data[loop.index].uv = (u, (w[va] - v0) / dv)
        me.uv_layers.active = uv
        me.materials.clear()
        me.materials.append(m)
    return m


def flat(color, shading="soft", name=None):
    """A plain colour with the look of k.paint_part: half self-lit, so the colour reads as drawn (cel / figure
    style) and a plain piece matches the painted pieces next to it. Use it for the plain pieces of a model whose
    other pieces carry drawings; use k.mat for real materials (metal, glass, plastic).

    color: '#rrggbb' (read it off the picture: view.color(u, v)).
    shading: 'soft' (default, as k.paint_part), 'lit', 'unlit', or a number 0..1 = self-lit share.
    """
    from .paint import SHADING
    e = SHADING.get(shading, shading)
    if isinstance(e, bool) or not isinstance(e, (int, float)):
        raise ValueError("shading must be 'soft', 'lit', 'unlit' or a number 0..1")
    key = name or "flat_%s_%d" % (core.hex_of(color).lstrip("#"), int(round(float(e) * 100)))
    if key in bpy.data.materials:
        return bpy.data.materials[key]
    col = core.rgba(color)
    m = bpy.data.materials.new(key)
    nt, bsdf = _principled(m)
    bsdf.inputs["Base Color"].default_value = col
    bsdf.inputs["Roughness"].default_value = 0.85
    for nm in ("Specular IOR Level", "Specular"):
        if nm in bsdf.inputs:
            bsdf.inputs[nm].default_value = 0.1
            break
    m.diffuse_color = col
    if e > 0:
        out = next(n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = col
        ms = nt.nodes.new("ShaderNodeMixShader")
        ms.inputs["Fac"].default_value = float(e)
        nt.links.new(bsdf.outputs[0], ms.inputs[1])
        nt.links.new(em.outputs[0], ms.inputs[2])
        nt.links.new(ms.outputs[0], out.inputs["Surface"])
        m["f3d_flat"] = True
    return m
