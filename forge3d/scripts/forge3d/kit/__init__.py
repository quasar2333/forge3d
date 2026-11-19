"""forge3d.kit - modelling vocabulary for Blender scripts.

    from forge3d import kit as k

Conventions: metres; Z up; the model FACES -Y (toward the front-view camera); X is its width, +X on the viewer's
right. Angles in degrees. Colours are '#rrggbb' strings.
"""
from .core import (anchor, attach, bbox, collection, delete, describe, dup, fit_height, get, ground, group,
                   hex_of, in_path, out_path, parent, place, raycast_all, reset, resize, rgba, rotate,
                   scene_objects, stick, surface_point, translate)
from .shapes import (box, capsule, catmull_rom, cone, cyl, disc, extrude, from_pydata, lathe, loft, plane, prism,
                     shade, skin, sphere, sweep, text, torus, tube)
from .modify import (apply, apply_scale, array, bend, bevel, boolean, cut, decimate, displace, intersect, join,
                     mirror, radial, remesh, smooth_mesh, solidify, subdiv, taper, twist, union)
from .materials import PRESETS as MATERIALS
from .materials import assign, assign_faces, flat, image_mat, mat, project_image
from .staging import STAGES, VIEWS, camera, light, render, stage, world
from .qa import allow_floating, lint
from .io import export, import_mesh, save
from .refkit import ref
from .adopt import adopt, drop_floaters, fit_to_ref, stretch_to_ref
from .paint import bake, dab, paint_from_ref, paint_part, undercoat
from .conform import conform
from .ironing import iron
from .sculpt import inflate, volume
from .layerkit import drape, layer, layer_ref, seat
from .partkit import assemble, slot
from .ink import outline
