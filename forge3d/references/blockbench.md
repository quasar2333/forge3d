# Blockbench models

Three products, pick by what the user needs:

| | Mesh model | Designed cuboid model | Voxel model |
| --- | --- | --- | --- |
| Made by | `k.export("x.bbmodel")` / `f3d bb mesh` on a finished forge3d model | bones + cuboids you place: one script with `bb.Model`, or part by part with `bb.plan` | `f3d bb voxelize` of any forge3d model |
| Looks like | the Blender model itself: smooth, textured | a Minecraft-style character: few, deliberate boxes | a voxel statue |
| Elements | one mesh per object | 15-120 cuboids (a boss built from parts: 300-600) | hundreds to thousands of cuboids |
| Formats | Generic Model only | any (Bedrock, Java, modded ...) | any |
| Rig / animation | groups from parents, no pivots | bones with pivots, ready to animate | none (one bone) |
| Use for | likeness first; display, renders, glTF export | mods, mobs, items, animation, hand editing | blocky display pieces |

All write genuine `.bbmodel` files (texture embedded, also saved as `.png` next to it) and all go through the
same loop: `python3 "$F3D" build script.py --ref refs` (or `python3 "$F3D" compare file.bbmodel --ref refs`), look, fix.

## Mesh model

```python
# at the end of any build script (after k.assemble() for a model built from parts)
k.export("out/name.bbmodel")                       # or: bb.save_scene("out/name.bbmodel", height=64, faces=60000)
```

or from the shell: `python3 "$F3D" bb mesh model.py --ref refs -o out/name.bbmodel --height 64 --faces 60000`.
Each Blender object becomes one Blockbench mesh element with its name; objects parented to an empty
(`k.group`) land in a Blockbench group of that name. Projection paint is baked to one texture; flat materials
share a small palette texture. `height` = Blockbench pixels tall (16 = one block), `faces` = total face budget
(denser models are reduced; Blockbench gets sluggish far beyond 100k). Build the model from separate parts
(the parts route, or kit parts in one script) when the user wants to edit pieces in Blockbench; a single
generated mesh is one piece.

## Designed model in one script

For a simple subject (a mob, an item, a prop of up to about 60 boxes). Anything bigger is built part by part,
see below.

```python
from forge3d import bb, kit as k
R = k.ref("refs")
F = R["front"]
m = bb.Model("fox", height=32, snap=1)                 # the subject is 32 px tall; corners snap to whole pixels

root = m.bone("root")
body = m.bone("body", pivot=(0, 0, 10), parent=root)
m.cube("torso", lo=(-4, -3, 8), hi=(4, 3, 18), bone=body)
head = m.bone("head", pivot=(0, 0, 18), parent=root)   # pivot = where the part hinges
m.cube("head", size=(10, 8, 9), base=(0, 0, 18), bone=head)
m.cube("hair", size=(10, 8, 9), base=(0, 0, 18), bone=head, inflate=0.5)        # overlay layer
arm = m.bone("arm_left", pivot=(-5, 0, 17), parent=body, rot=(0, -10, 0))
m.cube("arm_left", lo=(-7, -1, 9), hi=(-4, 2, 17), bone=arm)
m.mirror(arm)                                           # bone + its cubes, across x = 0; names swap left/right
m.cube("ear_left", lo=(-9, 1, 20), hi=(-4, 2, 26), bone=head, cutout=F.mask(box=(-0.3, 0.6, -0.1, 0.85)))

m.paint_from_ref(R)                                     # pixel-art texture from the reference pictures
m.save("fox.bbmodel")
```

**Units and axes.** Blockbench pixels (16 px = one Minecraft block), same axes as the rest of forge3d: X = the
viewer's right in front view, Y = depth (front of the model at -Y), Z = up, feet on z = 0. A reference position
`u, v` (subject 1.0 tall) is `u * m.height, v * m.height` pixels. Faces are named
`front back left right top bottom` (left = the -X side). forge3d converts to Blockbench's own axes when saving.

**`bb.Model(name, height=32, format="free", density=1, snap=0)`**
- `height`: pixels tall. 16-24 chunky mob, 32-48 detailed character. More pixels = more detail in the texture.
- `format`: `free` (Generic Model: any rotation, exports glTF/OBJ/FBX; the safe default), `bedrock`
  (Bedrock Entity), `java_block` (Java Block/Item: rotations only about one axis in 22.5 degree steps, sizes
  limited), `modded_entity`, `optifine_entity`, `bedrock_block`. Use what the user asks for; otherwise `free`.
- `density`: texels per pixel (2 or 4 gives a finer texture on the same boxes).

**`m.bone(name, pivot=(x, y, z), parent=None, rot=None)`**: a Blockbench group. Children rotate about `pivot`.
Give every limb its own bone with the pivot at the joint (shoulder, hip, neck) so the model can be animated.

**`m.cube(name, lo=, hi=)`** or `size=(w, d, h)` with `at=` (centre) or `base=` (bottom centre). Options:
`bone=`, `rot=(x, y, z)` degrees about `pivot=` (default the cube centre), `inflate=` (grows the box without
changing its texture: hair over a head, sleeves over arms), `color="#rrggbb"`, `cutout=<RefMask>`.

**`m.beam(name, a, b, width, depth)`**: a cuboid running from point `a` to point `b` (limbs, spars, blades,
tail segments). **`m.mirror(cube_or_bone)`**: mirrored copy across x = 0. **`cube.panel(colour, style)`**:
hand-shaded pixel-art paint for one cube (see "Building a cuboid part").

**Textures.** `m.paint_from_ref(ref, colors=48, cutout="thin")` gives every texel the colour the best-placed
reference view shows there (front view on front faces, side view on side faces, ...; a face must be visible
from that view, i.e. not behind another cube). Texels no view sees copy their nearest painted neighbour on the
same cube. `colors` limits the palette for a pixel-art look (`None` keeps every shade).

Cut-outs make flat plates take the outline of the artwork:
- any cube at most 1.5 px thick becomes transparent wherever it sticks out of the reference silhouette
  (`cutout="thin"`, the default): hair curtains, capes, skirts' hems, wings;
- `m.cube(..., cutout=F.mask(...))` cuts a plate to one piece of the picture and shows that piece on it, even
  when other cubes are in front: fins, ears, bows, tails, hair locks, face details that stand off the head.

Hand painting, for details the projection misses or models without a reference:

```python
c = m.cube("eye_block", lo=(-3, -4.2, 22), hi=(3, -4, 25))
c.paint("#f4d9c6", top="#3b2a20")                 # all faces, then single faces
f = c.face("front")                               # .size -> (w, h) texels; origin top-left, y down
f.fill("#f4d9c6").rect_fill(1, 0, 3, 2, "#1d2a55").pixel(1, 0, "#ffffff")
f.rows(["KKK.KKK", "KWB.BWK"], {"K": "#101020", "W": "#ffffff", "B": "#3a6fd8"})   # pixel art from text
m.paint_from_ref(R, keep=True)                    # paint the rest, keep hand-painted faces
m.shade()                                         # soft top-to-bottom shading + grain on flat-coloured faces
```

**Design guidance**
- Block the silhouette first with 6-10 boxes (head, torso, limbs, main hair mass) and run the loop; the report
  says where the model is too wide or narrow and by how much. Then add secondary boxes and plates.
- Chibi and Minecraft proportions read best with whole-pixel sizes and at most one or two rotated parts.
- Slanted or curved outlines: stack a few boxes of decreasing size, or use a cut-out plate. Do not rotate boxes
  by odd angles to fake curves.
- Depth layering sells the model: face < bangs < headgear; torso < apron < arms. Offset layers by 1 px or use
  `inflate`.
- Keep boxes inside the reference outline where they are thick; blue areas in the compare picture are boxes
  that are too big.

## A cuboid model built part by part

A subject of several parts (a character, a mech, a boss with a tail and wings) goes the parts route of SKILL.md
with one extra line in the blockout, which makes every part a cuboid part of one shared Blockbench model:

```python
# blockout.py
from forge3d import bb, kit as k
bb.plan(height=80, format="bedrock", density=2)     # 80 px tall (5 blocks); 2 texels per pixel for the texture
k.slot("chassis", k.box(size=(0.34, 0.30, 0.34), at=(0, 0.02, 0.50)), note="...")
k.slot("tail", k.sweep([...], r=0.05), on="chassis", note="...")       # rough shapes, as for any blockout
```

- `height`: 32-48 for a mob, 64-96 for a detailed boss or statue. `format`: `bedrock` or `modded_entity` for
  Minecraft, `free` otherwise. The slot notes say what moves (jaw, wings, each tail segment, fingers), because
  a part's bones are its builder's job.
- `f3d blockout` then writes cuboid work orders; hand the parts out exactly as in SKILL.md.
- The final script joins them into one bone tree and one texture:

```python
# model.py
from forge3d import bb
m = bb.assemble(name="mech_boss")        # each part's bone hangs from the part it sits on, the rest from "root"
m.save("out/mech_boss.bbmodel")
```

`python3 "$F3D" build model.py --ref refs`, then `python3 "$F3D" bb check out/mech_boss.bbmodel --export bedrock:out/mech_boss.geo.json`.

## Building a cuboid part

For the builder of one part of a cuboid model (the brief says so). Pixels, same axes as everywhere: X = the
viewer's right in the front picture, Y = depth (front at -Y), Z = up. A point (u, v) read off the grid of
`ref.png` is `x = u * m.height`, `z = v * m.height`; the side picture gives `y`.

```python
from forge3d import bb, kit as k
m = bb.part(pivot=(0, -8, 48))            # this part's share of the model; pivot = where the whole part hinges
F = k.ref()["front"]
gold, iron = F.color(0.12, 0.55), F.color(0.20, 0.42)          # colours come from the picture

m.cube("skull", lo=(-5, -30, 44), hi=(5, -8, 52)).panel(gold)              # a box, painted as an armour plate
m.beam("crest", (0, -10, 53), (0, -2, 60), width=3, depth=2).panel(gold, style="edge")    # a slanted box, end to end
jaw = m.bone("jaw", pivot=(0, -12, 46))                                    # a bone for everything that moves
m.cube("jaw_plate", lo=(-4, -26, 42), hi=(4, -12, 44), bone=jaw).panel(iron)
eye = m.cube("eye_left", lo=(-5.2, -22, 48), hi=(-4.8, -19, 50)).panel("#14745f", style="flat")
eye.face("left").rect_fill(1, 1, 3, 2, "#46ae8e")                          # detail painted on a face
upper = m.bone("upper_arm_left", pivot=(17, 0, 46))
m.beam("upper_arm_left", (17, 0, 46), (30, -4, 30), width=9, bone=upper).panel(gold)
m.mirror(upper)                                                            # the bone, its cubes and child bones
```

- **Do not save**: `f3d part <name>` saves the part and shows it in `check.png`. `m.root` is the part's own bone;
  bones and cubes you add hang from it. `m.lo`, `m.hi` are the slot's box in pixels.
- **Bones** are what an animator moves: one per thing that hinges, pivot at the joint, child bones for the next
  joint down (upper arm > forearm > hand > fingers; a tail as a chain, one bone per segment). Rest pose: bones
  unrotated, the pose is in the cubes. Lowercase names with `left` / `right`. A bone may hang from a bone of
  another part: `m.bone("pauldron_left", parent="upper_arm_left")` is joined up at assembly.
- **`m.beam(name, a, b, width, depth)`** is the tool for everything slanted: limbs, wing spars, blades, claws,
  horn and tail segments. A curve is a chain of short beams. `m.cube(..., rot=(x, y, z))` turns a box about its
  centre.
- **Texture**: `.panel(colour, style)` paints a cube as hand-shaded pixel art: `"panel"` a bevelled plate,
  `"ribbed"` vents and joints, `"edge"` small parts, `"flat"` lights and skin. Then paint what the drawing shows
  on the faces: seams, vents, lenses, stripes (`face.rect_fill`, `.pixel`, `.rows`). Unpainted cubes are black.
- **Design**: block the part's outline with a few big boxes and check; then layer plates over them (shoulder
  shells, knee caps, vents, spikes, lights as small inset cubes). Layers overlap the box below by half a pixel
  and never share a face plane with it (coplanar faces flicker). Thin blades and fins are plates 1 px thick. A
  boss part takes 30 to 100 cubes; whole-pixel sizes read best.
- `check.png` reads as for any part: the magenta outline against the drawing in each view, then the
  three-quarter tiles.

## Voxel model

```bash
python3 "$F3D" bb voxelize model.py --ref refs --height 48 -o out/name.bbmodel
```

Voxelises the model a build script, `.blend` or mesh file produces (best: the finished model),
merges the voxels into cuboids and paints them from the reference pictures (or from the model's own colours
without `--ref`). `--height` 32 = chunky, 48 = detailed, 64 = fine (cube count grows quickly), `--colors`
palette size, `--format` as above. In a script: `m = bb.voxelize(height=48, ref=R); m.save("name.bbmodel")`
after building the Blender-side model.

## Checking and exporting with the real Blockbench

```bash
python3 "$F3D" bb check out/name.bbmodel                                        # needs Blockbench installed
python3 "$F3D" bb check out/name.bbmodel --export bedrock:out/name.geo.json --export gltf:out/name.gltf
```

Starts a private, hidden Blockbench (own profile; the user's windows and settings are untouched), loads the
file, prints what Blockbench found (format, cubes, groups, textures, validator messages) and saves a picture of
Blockbench's own viewport: look at it. `--export codec:path` runs Blockbench's exporters: `bedrock`
(.geo.json), `java_block` (.json), `modded_entity` (.java), `optifine_entity` (.jem), `gltf`, `obj`, `fbx`,
`collada`, `stl`. Without Blockbench the `.bbmodel` is still complete; say that the in-app check was skipped.
