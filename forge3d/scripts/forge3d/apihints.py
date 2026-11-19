"""Known Blender-API traps -> the fix. Matched against error text when a build script fails."""
import re

HINTS = [
    (r"BLENDER_EEVEE_NEXT|enum \"BLENDER_EEVEE\" not found",
     "The EEVEE engine id differs per version (BLENDER_EEVEE in 5.x and <=4.1, BLENDER_EEVEE_NEXT in 4.2-4.5). "
     "Use k.render(path, engine='EEVEE') or forge3d.kit.staging.set_engine('EEVEE')."),
    (r"use_auto_smooth|auto_smooth_angle",
     "Mesh.use_auto_smooth was removed in 4.1. Use k.shade(obj, smooth=35) (mesh.shade_smooth() + "
     "mesh.set_sharp_from_angle(angle=radians(35)))."),
    (r"NISHITA|sky_type",
     "Sky texture types are SINGLE_SCATTERING, MULTIPLE_SCATTERING, PREETHAM, HOSEK_WILKIE in 5.x (no NISHITA). "
     "For lighting prefer k.stage('outdoor') / k.world('courtyard')."),
    (r"enum \"FAST\" not found|solver.*FAST",
     "Boolean solvers are FLOAT, EXACT, MANIFOLD (no FAST). Use k.cut / k.union / k.boolean."),
    (r"key \"(Specular|Subsurface|Transmission|Clearcoat|Clearcoat Roughness|Sheen|Emission)\" not found",
     "Principled BSDF sockets were renamed in 4.0: Specular IOR Level, Subsurface Weight, Transmission Weight, "
     "Coat Weight, Coat Roughness, Sheen Weight, Emission Color. Simplest: k.mat(color, preset, rough=..., "
     "transmission=..., coat=..., emit=...)."),
    (r"use_ssr|use_bloom|use_gtao|use_soft_shadows|use_ssr_refraction|shadow_cascade_size|shadow_cube_size|gtao_distance|bloom_",
     "Legacy EEVEE settings (bloom, SSR, GTAO, soft shadows, shadow map sizes) no longer exist. Current ones: "
     "scene.eevee.use_raytracing, use_shadows, taa_render_samples. k.stage() sets sensible values."),
    (r"shadow_method|'Material' object has no attribute 'blend_method'|shadow_mode",
     "Material.shadow_method is gone and blend_method is replaced by surface_render_method ('DITHERED' or "
     "'BLENDED'). k.mat(..., alpha=0.5) / preset='glass' handle transparency."),
    (r"context is incorrect|poll\(\) failed",
     "bpy.ops operators need UI context that does not exist headless. Use kit functions or data-level APIs "
     "(bmesh, obj.modifiers.new, k.apply) instead; if an operator is unavoidable wrap it in "
     "`with bpy.context.temp_override(active_object=obj, selected_objects=[obj], object=obj):`."),
    (r"object has no attribute 'select'\b|objects\.link|scene\.objects\.active|scene\.update\(\)",
     "That is Blender 2.7 API. Use obj.select_set(True), collection.objects.link(obj), "
     "bpy.context.view_layer.objects.active = obj, bpy.context.view_layer.update()."),
    (r"calc_normals_split|calc_normals\b|free_normals_split|create_normals_split",
     "Mesh.calc_normals / calc_normals_split were removed (normals are always available). Just delete the call."),
    (r"action\.fcurves|has no attribute 'fcurves'|action\.groups|id_root",
     "Action.fcurves / groups were removed in 5.0 (layered actions). Use obj.keyframe_insert(data_path, frame=n) "
     "and, to edit curves, action.layers[0].strips[0].channelbag(slot).fcurves."),
    (r"No module named 'bgl'",
     "The bgl module was removed; use the gpu module (not available headless) or render with k.render."),
    (r"use_nodes",
     "Material.use_nodes / World.use_nodes are deprecated: materials always have a node tree in 5.x. "
     "k.mat() and k.world() build node trees for you."),
    (r"NoneType' object has no attribute '(location|data|modifiers|name|select_set|scale|rotation_euler)'",
     "A lookup returned None: bpy.context.active_object / bpy.context.object are unreliable headless, and "
     "bpy.data.objects.get(name) returns None when Blender renamed a duplicate (Cube.001). Keep the object "
     "returned by the function that created it instead of looking it up afterwards."),
    (r"primitive_\w+_add\(\) got an unexpected keyword|keyword \"(size|radius|depth|diameter\d?|view_align|enter_editmode)\" unrecognized",
     "bpy.ops.mesh.primitive_* arguments changed across versions. Use k.box / k.cyl / k.sphere / k.cone / "
     "k.torus instead."),
    (r"Object '.*' not in View Layer|can't be selected because it is not in View Layer",
     "The object is not linked to the scene. Objects made by kit functions are linked automatically; link "
     "your own with bpy.context.scene.collection.objects.link(obj)."),
    (r"ImportError: cannot import name|No module named 'forge3d'",
     "Run build scripts through `f3d build`, which puts forge3d on the path. In a live Blender session add "
     "sys.path.insert(0, '<skill>/scripts') before `from forge3d import kit as k`."),
    (r"has no attribute 'cut'|unexpected keyword argument|has no attribute '\w+'.*forge3d|module 'forge3d.kit' has no attribute",
     "Check the kit signature: `python3 f3d.py api <name>` prints the exact parameters and docstring; "
     "`python3 f3d.py api` lists every function."),
    (r"no reference pack",
     "Pass the pack on the command line (f3d build model.py --ref refs) or open it in the script: R = k.ref('refs')."),
    (r"has no region|the mask is empty",
     "Region ids are per view and change when the pack is rebuilt: look at refs/<view>_marks.png again, or select "
     "by box / colour instead: F.mask(box=(u0, v0, u1, v1), color='#rrggbb', tol=30)."),
    (r"is not painted",
     "k.dab works on projection paint: call it after k.paint_from_ref and before k.export / k.bake."),
    (r"view_transform|enum \"Filmic\" not found",
     "View transforms: 'Standard', 'AgX', 'Filmic' (availability varies), 'Khronos PBR Neutral'. "
     "k.stage(view='Standard') sets it safely."),
    (r"ShaderNodeMixRGB|MixRGB",
     "ShaderNodeMixRGB is legacy; use ShaderNodeMix with data_type='RGBA' (inputs 'Factor', 'A', 'B'; "
     "output 'Result')."),
    (r"face_maps|vertex_colors",
     "mesh.vertex_colors / face_maps are legacy. Use mesh.color_attributes.new(name, 'BYTE_COLOR', 'CORNER')."),
]


def hints_for(text):
    out = []
    for pat, hint in HINTS:
        if re.search(pat, text):
            out.append(hint)
    return out
