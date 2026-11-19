# Building a model from parts with the kit

Read this for a model written as one script (hard-surface, props, vehicles, low-poly) and for the shape and
material vocabulary in general. A character or anything of many parts is laid out as a blockout and built part
by part instead (SKILL.md, "The parts route"; the builder of a part reads [parts.md](parts.md)). `from forge3d import kit as k`; `python3 "$F3D" api <name>` prints any signature in full.

Contents: reading the reference · planning · shape vocabulary · shapes from pictures · placing and joining ·
surface finish · colour · presentation and export · recipes.

## Reading the reference

`f3d ref` writes, per view, `<view>.png` (cutout), `<view>_grid.png` (coordinate grid in world units: read
positions and sizes off it), `<view>_marks.png` (numbered colour regions) and `overview.png` (all grids side by
side). In a script the same data is live:

```python
R = k.ref()                       # or k.ref("refs")
R.height, R.size                  # 1.0, {'width': .., 'depth': .., 'height': ..}
R.palette                         # dominant colours, '#rrggbb'
F, S = R["front"], R.side         # views: front back right left top bottom; R.side = right or left
F.bbox                            # (u0, v0, u1, v1) of the silhouette; u = picture right, v = up (world units)
F.span(0.55)                      # (left, right) edge of the silhouette at height 0.55
F.color(0.0, 0.52)                # colour of the picture at that point
F.region(10)                      # {'color', 'share', 'box': (u0, v0, u1, v1), 'pin': (u, v)}
```

`F.px([(x, y), ...])` converts pixel positions of the ORIGINAL image you were given (x right, y down) into
(u, v), for outlines you trace on that image: `F.mask(polygon=F.px([(717, 244), (720, 201), ...]))`.

In the front view u is world x and v is world z. In the `right` view (subject faces left) u is world y, so the
subject's front is at negative u. `F.world(u, v, depth)` converts picture coordinates to a world point.

Masks select a piece of a picture and are the input of the picture-driven shapes:

```python
m = F.mask()                              # whole silhouette
m = F.mask(regions=[4, 8])                # numbered regions from front_marks.png
m = F.mask(box=(-0.2, 0.45, 0.2, 0.95))   # everything of the subject inside a box (u0, v0, u1, v1)
m = F.mask(color="#f9e9e3", tol=20)       # pixels near a colour (skin, a white apron ...)
m = F.mask(polygon=[(u, v), ...])         # hand-drawn outline, clipped to the subject
m | m2, m & m2, m - m2                    # union, intersection, difference
m.fill() .close(0.01) .open(0.01) .grow(0.01) .shrink(0.01) .largest() .symmetric() .clip(v0=0.4)
m.bbox, m.center, m.area, m.outline(), m.profile(24), m.span(0.5)
```

Region maps are approximate (one region can leak across a thin outline). A box or polygon you read off the grid,
intersected with a colour, is often the most reliable way to isolate a part:
`F.mask(box=(...), color="#32386e", tol=30).close(0.01).largest()`.

## Planning

1. Name the parts before writing code: the big masses first (body, head, hull, chassis), then the secondary
   forms (limbs, wheels, handle, hair), then details. Ten well-proportioned parts beat forty misplaced ones.
2. Give every part real measurements from the grid. Typical mistake: correct shapes, wrong proportions.
3. Build symmetric things once and `k.mirror()` them. Name them `ear.L`, `wheel_left`, ... (names swap).
4. Let parts interpenetrate a little where they join (`embed=0.1..0.3` in `k.attach` / `k.stick`). Parts that
   merely touch look glued on and lint reports gaps as floating parts.
5. Build, look at the sheet, fix proportions of the big masses, and only then add detail.

## Shape vocabulary

All generators take `name=`, `at=` (centre) or `base=` (bottom centre: no z + h/2 arithmetic), `rot=` (XYZ
degrees), `mat=` (material or '#rrggbb'), `smooth=`, `parent=`.

| Want | Use |
| --- | --- |
| boxy, with soft edges | `k.box((x, y, z), bevel=0.02)` |
| round, turned on a lathe: bottle, wheel, lamp, vase, dome, cup | `k.lathe([(r, z), ...], smooth_profile=6)` |
| body that changes cross-section along an axis: torso, hull, fuselage, shoe, car body | `k.loft([dict(t=, w=, h=, n=, off=), ...], axis="Z", interp=6)` (n: 2 ellipse, 4-6 rounded box) |
| long and thin along a path: tail, horn, cable, handle, strand, strap | `k.sweep(path, radii=[...], smooth_path=8, squash=0.4)` |
| flat outline with thickness: plate, blade, fin, gear, sign | `k.extrude(outline, depth, plane="XZ", bevel=)` |
| branching organic: creature, tree, coral | `k.skin({name: (x, y, z, r), ...}, edges)` |
| primitives | `k.sphere(radii=(rx, ry, rz))`, `k.cyl`, `k.cone`, `k.capsule`, `k.torus`, `k.tube`, `k.prism`, `k.text` |
| holes, cut-outs, fused solids | `k.cut(target, cutter)`, `k.union`, `k.intersect` |
| repeats | `k.mirror`, `k.array(obj, n, offset)`, `k.radial(obj, n, axis, center)` |
| catch the light on hard edges | `k.bevel(objs, width)` (almost always worth it on hard-surface parts) |
| rounder, softer | `k.subdiv(objs, 2)`; fuse overlapping organic parts: `k.join` then `k.remesh(obj, voxel)` + `k.smooth_mesh` |
| deform | `k.bend`, `k.twist`, `k.taper`, `k.displace`, `k.resize(objs, (sx, sy, sz))` |

## Shapes from pictures

- `k.volume(front_mask, side_mask)`: rounded solid whose front and side outlines are those masks; at each
  height the cross-section is as wide as the front outline and as deep as the side outline. Use it for the main
  masses (whole body, head, hair mass, dress, creature) from two views of the same part. `roundness=2` elliptical,
  `3-4` boxier. Lobes that are separate in the front view (two legs) stay separate.
- `k.inflate(mask_or_polygon, view=, thickness=, at=)`: puffs one outline up like an inflated sticker: fins,
  ears, bows, tails, leaves, wings, hair locks, hands. `at` is where its middle sits along the view axis
  (front view: its y), `back=0` gives a flat-backed relief. Thin parts of the outline stay thin.
- `k.extrude(mask.outline(), depth, plane="XZ")`: the same outline with flat faces and sharp rims.
- `k.lathe([(right - centre, v) for v, left, right in mask.profile(24)])`: a turned shape with the reference's
  own profile. `k.loft` sections can be filled from `F.mask(...).profile()` and `S.mask(...).profile()` the same way.

Masks come from different drawings, so a front mask and a side mask of "the same" part rarely agree to the
pixel; `volume` uses the union of their height ranges.

## Placing and joining

- `k.attach(child, to, at="top", embed=0.1)` puts the child's opposite face on the parent's anchor
  (`top bottom left right front back` and combinations like `"top-front"`, or (x, y, z) factors in -1..1).
- `k.stick(child, to, at=(0.3, -1, 0.2), embed=0.2)` places a part on the parent's surface where a ray from that
  bounding-box spot hits it, oriented along the surface normal: eyes, buttons, spikes, knobs. (0.3, -1, 0.2) is
  "on the front, 30 % right of centre, 20 % above the middle", exactly as seen in the front view.
- `k.translate / k.rotate / k.resize / k.place`, `k.ground()`, `k.fit_height(objs, h)`, `k.group(name, objs)`.
- `k.bbox(obj)` gives `.min .max .size .center` and `.point("top-front")`; `k.describe()` prints every object's
  size and place when something lands in the wrong spot.
- Lint (run by `f3d build`, or `k.lint()`) reports floating parts with the gap and direction, parts hidden
  inside others, inverted meshes, duplicates, missing materials. Mark parts that are meant to hover with
  `k.allow_floating(obj)`.

## Surface finish

`k.conform(objs=None)` warps all given objects together so their combined silhouettes meet the reference
outlines in every view. It fixes proportions and contours, keeps joints joined, and never moves anything
further than `limit` (8 % of the height), so it cannot replace a missing part. Run it on the whole model, not
part by part. Skip it for strictly geometric hard-surface work where straight lines must stay straight (or pass
`rounds=2, limit=0.03`).

`k.iron(objs, view.mask(box=(u0, v0, u1, v1)), smooth=0.04)` then flattens the small relief under one part of
a reference picture, so that what is drawn there lies on a smooth surface like a print, and gives that area to
that view alone when painting. It is what keeps a projected face whole: on a generated or lumpy surface an eye
drawn across a lock of hair is printed half on the lock and half on the cheek, and tears apart in every view
but the picture's own. Relief narrower than about twice `smooth` (fraction of the height) goes, the broad form
stays; whatever stands in front of the ironed sheet is pressed into it. Use it under faces, prints, emblems
and lettering; never on plain-coloured masses. Parts you built yourself as smooth shapes do not need it.

## Colour

One piece, one material: when every colour area is its own shape nothing can smear. Plain materials carry most
pieces (`k.flat("#hex")` for the drawn, cel look that matches painted pieces; `k.mat` for real materials); `k.paint_part` adds a drawing (a face, an emblem, trim, lace) to the piece that carries it, from the
one picture that faces it.

Two ways, which can be mixed per object:

- `k.paint_from_ref(objs=None)`: projects every reference view onto the surfaces facing it (with occlusion)
  and uses the single side view mirrored for the other side (`mirror=False` for asymmetric subjects). It
  carries everything drawn: faces, prints, gradients, trim, shading. Call it after the geometry is final.
  How it decides, so you can predict it:
  - views are not averaged where they overlap: the front picture takes what it sees reasonably head-on, then
    the back, then the sides. Areas passed to `k.iron` belong to their view alone;
  - a surface seen at a slant gets the picture's flat-colour twin (same areas, no ink lines, no highlights),
    so nothing is dragged out into streaks;
  - a surface no view sees takes the flat colour of the nearest broadly painted surface, measured along the
    surface with folds counting as far away: hair behind a ribbon stays hair-coloured. It gets colour, never
    drawing; a pattern that should continue there has to be modelled or dabbed;
  - thin strips of a neighbouring part's colour along part edges (the picture and the geometry are never in
    perfect register) are replaced by the part's own colour (`tidy=False` turns that off).
  The defaults are tuned on real turnaround sheets; reach for an option only against a defect you can see:
  `erase=[S.mask(box=(u0, v0, u1, v1))]` (leave that part of a picture out: for accessories the sheet draws
  facing the viewer in every view, which would otherwise be printed two or three times),
  `zones=[(F.mask(box=(u0, v0, u1, v1)), "front")]` (only the front view paints what is seen through that part
  of its picture), `fill="#rrggbb"` (one flat colour for what no view sees), `shading="soft"|"lit"|"unlit"`,
  `mirror=False`, `views=[...]`. Retouch afterwards with
  `k.dab(obj, "#4a5ea3", at=(x, y, z), radius=0.2, facing="up", unseen_only=True)`.
- Materials: `k.mat("#c0392b", "car_paint")`. Presets: plastic, glossy_plastic, matte, rubber, paint, car_paint,
  ceramic, metal, steel, brushed_metal, chrome, aluminum, iron, gold, copper, brass, glass, frosted_glass,
  tinted_glass, water, wood, fabric, velvet, leather, skin, wax, stone, concrete, clay, toon, emissive, screen.
  Options: `rough=`, `metal=`, `emit=`, `texture="wood"|"marble"|"stripes"|"checker"|"speckle"|"noise"|"gradient"`
  with `color2=`, `bump=`, `variation=`. Take the colours from `R.palette` / `F.color(u, v)`, not from memory.
  `k.assign_faces(obj, mat, "top")` gives part of a mesh another material; `k.project_image` puts one picture
  (logo, label, screen) on a part.

- `k.paint_part(part, views, keep=, base=)`: one part, its own share of the drawing. Where the part faces a
  view (cosine above `facing=0.35`) it shows that picture, limited to the `keep` mask; everything else is the
  base colour (`"#rrggbb"`, `"auto"` = the main colour of the kept pixels, `"gradient"` = that colour as it
  changes with height). No occlusion from other parts, no stretching around sides, nothing invented. `views`
  may be a list in order of priority; `mirror=True` copies a side view to the far side; `keep=False` paints
  no drawing, only the base colour or gradient. The paint is fixed to the piece: cut a bow or a fin out of the
  picture that shows it broadside, paint it there, then `k.rotate` / `k.translate` it to where it sits in 3D.
- `k.drape(part, onto, view="front")` lays a flat-built piece onto the surface behind it (an apron on a skirt,
  a fringe on a forehead, an emblem on a chest) without changing its outline in that view or its paint.

Parts of a layered model want `paint_part`; a single continuous surface wants `paint_from_ref`; machined,
glossy or transparent things want real materials.

## Presentation and export

```python
k.stage("soft")                   # studio, soft (stylised / cute), dramatic, product_dark, outdoor, sunset, flat
k.camera("hero", lens=85)         # front back left right top iso iso_left iso_back three_quarter hero high, or az=, el=
k.render("out/beauty.png", res=(1600, 1200))        # engine="CYCLES" for glass, liquids, final hero shots
k.export("out/model.glb")         # .glb .gltf .fbx .obj .stl .usdz .ply; bakes projection paint to a texture
```

`python3 "$F3D" view <script | .blend | .glb | .bbmodel> --views front,iso,az120el20,clay:iso,parts:iso` renders any source
from any angles (`clay:` shows form without colour, `parts:` gives each object its own colour, `silhouette:`).
Relative paths inside a build script are relative to the script's folder.

## Recipes

**Drawn character.** Use the parts route (SKILL.md): a blockout of slots, every part built on its own from
its brief. The recipes for heads, hair, flat parts and parts lying on others are in [parts.md](parts.md).

Masks: `F.mask(color="#4e62a4", tol=40)` selects by colour (raise `tol` until shading and highlights are in,
lower it when a neighbouring colour leaks in), `F.mask(regions=[...])` by the numbered regions, `box=` /
`polygon=F.px([(x, y), ...])` by hand, where `F.px` converts pixel positions of the image you were given.
Combine with `|`, `&`, `-`; tidy with `.close(d)`, `.open(d)`, `.largest()`, `.clip(v0=...)`. Look at a mask
before trusting it: `mask.save("out/m.png")`.

**Quick solid without a generator.** `body = k.volume(F.mask().fill(), S.mask().fill())` gives the whole figure
as one rounded solid that already matches two views: a starting block for simple creatures and props, a pillow
for anything with layers.

**Hair.** A thin cap over the scalp plus every lock as its own tapered piece: `k.sweep(path, radii=[thick, ...,
0], squash=0.5, smooth_path=8)` for strands and bangs, `k.inflate` for flat curls drawn in the reference. Locks
overlap but stay separate, so each gets its line from `k.outline()`.

**Line art.** For a reference drawn with ink outlines (`f3d ref` says so), end the script with `k.outline()`:
a dark line around every separate object and shell, from any direction, in the width and colour of the
drawing. It is one extra object ("ink outline"); `.glb`, `.fbx` and `.usd` exports carry it, the others leave
it out.

**Cloth, skirts, capes.** `k.lathe` with a wavy hem profile, or `k.loft` with `n=2`; thin sheets get
`k.solidify`. Ruffles: `k.torus` rings or a `k.sweep` along the hem with `squash`.

**Vehicles, appliances, furniture.** One `k.loft` or bevelled `k.box` per main body, `k.cut` for wheel arches
and windows, `k.lathe` for wheels (`k.mirror` + `k.array`), `k.bevel` on everything, real materials
(`car_paint`, `glass`, `rubber`, `chrome`). Use `conform` gently or not at all.

**Props and tools.** Profile with `k.lathe` or `k.extrude` from `mask.outline()`; grips with `k.sweep`.

**Low-poly look.** `segs=8..12` on round shapes, `smooth=False`, flat materials, `k.stage("flat")`.
