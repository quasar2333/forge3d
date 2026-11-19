# References and generated base meshes

## What `f3d ref` accepts

```bash
python3 "$F3D" ref sheet.png -o refs                         # a turnaround sheet: views are found and split automatically
python3 "$F3D" ref front.png side.png back.png --views front,right,back -o refs      # or one file per view
python3 "$F3D" ref photo.jpg --bg rembg -o refs              # photo / busy background (first: python3 "$F3D" setup --extra rembg)
```

- View names, in reading order: `front back right left top bottom skip`. `right` = the camera stands on the
  subject's +X side, so the subject **faces left** in that picture; `left` = it faces right. The guess is
  printed and drawn into `overview.png`; always check it. Three-quarter or detail views cannot be measured:
  name them `skip`.
- `--height 1.7` sets the real height in metres (default 1.0, i.e. all numbers are fractions of the height).
- Background: flat colours and transparency are handled automatically (`--bg '#rrggbb'`, `--tol N` if the
  cut-out eats into the subject or leaves a halo). Enclosed areas of pure background colour become see-through
  gaps; `--holes keep` if that removed a white part of the subject.
- Views on one sheet are assumed to share one drawing scale and a common ground line; separate files are each
  scaled to the same height (`--scale same|height` overrides).
- x = 0 is the subject's symmetry axis in front / back views; in side views y = 0 is above the middle of the
  feet. z = 0 is the lowest point of each view.

The pack (`refs/`) holds, per view: the cutout, a mask, a texture source, `<view>_grid.png`,
`<view>_marks.png`, plus `overview.png`, `palette.png` and `ref.json`. It is the single source of truth for
`k.ref()`, `k.adopt`, `k.conform`, `k.paint_from_ref`, `bb` painting and `f3d compare`.

## One picture only

A single image is treated as the front view. A generated mesh still works (`gen3d` handles one image, and the
mesh is fitted to that picture), but nothing tells the pipeline what the back looks like: unseen sides are filled
with the surrounding colours. If you can generate images, draw the missing back and side views first (same
subject, same pose, orthographic, same scale, plain background), add them as extra files and rebuild the pack:
the result improves a lot. A picture taken at a three-quarter angle is fine for a generated mesh as long as it
is the only view.

## No picture at all

Generate the reference before modelling. A prompt that produces sheets `f3d ref` splits cleanly:

> Character / object turnaround sheet of <subject, with the colours, materials and style the user wants>.
> Exactly three views in one row: front view, side view facing left, back view. Orthographic, no perspective,
> identical scale and pose in every view, all standing on one common ground line, full subject visible with
> margin around each view, clear gaps between the views. Plain pure white background, flat even lighting, no
> cast shadows, no text, no labels, no extra props.

Add "arms slightly away from the body, legs apart" for characters, since limbs glued to the body cannot be
separated later. Show the sheet to the user if the design is theirs to approve, then run the loop.

## `f3d gen3d`

```bash
python3 "$F3D" gen3d refs                       # writes gen/base.glb beside refs/
python3 "$F3D" gen3d refs --quality high        # draft | standard | high (finer surface, slower, larger file)
python3 "$F3D" gen3d refs --seed 7 --force      # another attempt
python3 "$F3D" gen3d --list                     # backends
```

It sends the pack's cut-out views (front / back / left / right, all at one scale) to a public hosted model and
downloads the mesh. Default with several views: Hunyuan3D-2mv, which returns shape only; colour comes from
`k.paint_from_ref`. Results are cached: the same pictures and settings are not sent twice.

It needs network access and the free quota of the hosting service. On failure it says why; the usual causes are
no network in the sandbox, the service being busy, or the free GPU quota being used up (`export
HF_TOKEN=<Hugging Face token>` raises it). When it cannot be made to work, build the shape with the kit
instead (`k.volume`, `k.inflate`, primitives).

Any mesh works as a base: if the user has one from another generator or a scan, put it next to the script and
`k.adopt("their_file.glb")`.

## One generated mesh: `k.adopt`, `k.conform`, `k.paint_from_ref`

For a subject that is one continuous surface (an animal, a fruit, a rock, a statue) or a quick draft:

```python
from forge3d import kit as k
base = k.adopt("gen/base.glb", name="hero")   # clean up, reduce, turn/scale/place to match the reference views
k.conform(base, limit=0.04)                   # nudge the outlines onto the reference outlines
k.paint_from_ref(base)                        # the reference pictures projected on
k.export("out/hero.glb")
```

- `k.adopt(path, faces=150000, yaw=None, keep_materials=True)` joins the file into one object, deletes floating
  debris, reduces it to about `faces` triangles and fits it to the reference: it tries the four quarter turns
  about Z, then scale and position, and prints the silhouette IoU per view. If it picks the wrong turn (front
  and back look alike), pass `yaw=180`. Low IoU for every turn means the mesh and the pack disagree: check the
  view names first. The processed mesh is cached beside the source file.
- `k.stretch_to_ref(obj, ref)` scales it separately along X, Y and Z to the size the pictures show. It never
  bends the mesh, so a clean mesh stays clean. `k.conform(base)` warps the outlines onto the reference
  outlines; it closes more of the gap but creases a detailed mesh when the gap is large: keep `limit` small.
- `k.iron(base, view.mask(box=...))` flattens the generator's invented relief under the drawn features that
  matter (a face; prints, emblems), so the drawing is not torn over it.
- `k.paint_from_ref(base)` colours it from all pictures. A generated mesh is one fused surface, so the pictures
  decide which colour goes where. A face is painted by the front picture alone and eyes drawn in other pictures
  are left out; even so this is the smear-prone way to colour, which is why anything with a face, hair or
  clothes goes the parts route instead. Fix what the three-quarter tiles show: an accessory twice ->
  `erase=[S.mask(box=...)]`; a wrong colour where no picture looks -> `k.dab(base, "#4a5ea3", at=..., radius=...)`.

## Layer pictures: when a part hides what matters of another

By default a part is built from what the pictures show of it, and its hidden side is built plain. That is
enough for most subjects. Draw the hidden things first only when they matter and cannot be guessed: a face
mostly behind hair, a garment under a coat whose design the user cares about. It costs several minutes.

With the image tool, EDIT the reference sheet itself (view the file first so it is in context), once per layer,
every time starting from the original sheet:

> This is a layer separation for 3D modelling. Keep the canvas and every view exactly where they are: same
> positions, same scale, same pose, same drawing style, line weight and colours. Show ONLY <the layer>; remove
> everything else. Whatever stays visible is copied exactly: same outline, same length, every lock, fold and
> detail; do not simplify, shorten or tidy. Draw what was hidden behind the removed things, so the layer is
> complete on its own. Transparent background, no shadow, no text.

Two to four layers, inner to outer: `body` (the bare character: bald head with the complete face, the body in
its outfit, limbs, shoes, tail), `hair` (only the hair, like a wig hanging in place), `headwear` /
`accessories`. Then, before the blockout:

```bash
python3 "$F3D" layers refs body=layers/body.png hair=layers/hair.png headwear=layers/headwear.png    # several minutes
```

- `f3d layers` finds where and at what scale each layer picture sits on the sheet `refs` was made from and
  writes a pack `refs_<name>` per layer in the world frame of `refs`, plus `refs/layers_check.png`. Look at it:
  a layer that was moved, resized, simplified or is missing something is drawn again, saying what was wrong.
  Where the original shows the layer, the pack keeps the original's own pixels; the layer picture only supplies
  what was hidden.
- The work orders written by `f3d blockout` then show each part in the layer that has it alone, and a part
  script reads it with `B = k.layer_ref("body")` (masks, colours, `k.paint_part(..., ref=B)`).
- `f3d gen3d refs_hair` generates a mesh for one layer into `gen/<layer>.glb`; `k.layer(name, over=S.below)`
  in a part script brings it in, placed and stretched to the layer's pictures. A generated mesh has the right
  overall form and a smooth, fused surface: sculpted locks read better as hair.
- `k.seat(outer, inner, gap=0.004)` pushes every vertex of `outer` that is inside `inner` out to just above
  its surface.

## What a generated base is good for, and what not

The generator is a diffusion model guessing a 3D shape. Its guess is reliable for the overall form (how deep
the head is, where the hair mass goes behind the body) and unreliable for small relief: locks, folds, frills
and facial features come out as bumps that are near, but not on, what the pictures draw. Three consequences:

- Use it for form, let the pictures carry detail. Do not raise `gen3d --quality` hoping for a cleaner
  texture: more invented relief means more places where drawing and surface disagree.
- Surfaces no picture shows (between the views, the crown, undersides) get flat colour, not drawing. With
  three views roughly 40 % of the surface is painted from a picture; the rest is inferred. More views (a
  second side, a top view) help more than any setting.
- It cannot stay editable as separate parts, keep hard edges crisp, or respect exact dimensions. For those,
  use it as a visual guide (`python3 "$F3D" view gen/base.glb --views clay:front,clay:right,clay:iso`) and build with the
  kit.

The picture limits the result too: a view that is 300 px tall cannot give a sharp close-up. Ask for, or
generate, the largest reference available.
