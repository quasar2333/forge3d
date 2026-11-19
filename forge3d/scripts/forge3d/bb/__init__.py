"""Blockbench support: write .bbmodel files, paint their pixel textures, preview them in Blender, check them in the
real Blockbench app.

    from forge3d import bb
    m = bb.Model("name", height=32)
    m.cube(...); m.paint_from_ref("refs"); m.save("name.bbmodel")
"""
from .model import FACES, FORMATS, Bone, Cube, Face, Model, assemble, part, plan
from .voxel import voxelize
from .mesh import save_scene
