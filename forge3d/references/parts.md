# Building one part

You were given one part of a larger model. Its work order is `parts/<name>/brief.md`: what the part is, where its
slot is, which pictures show it. Other agents build the other parts at the same time; the blockout keeps you all in
one frame. Build only your part, and make it as good as the pictures: its shape, its colours, every detail the
brief names.

```bash
F3D="<folder of this skill>/scripts/f3d.py"      # the brief has the full path
python3 "$F3D" part <name>            # builds parts/<name>/part.py in its slot, checks it, writes parts/<name>/check.png
python3 "$F3D" api                    # every kit function;  python3 "$F3D" api sweep  shows one in full (check before first use)
```

Open `ref.png` and `slot.png`, write `part.py`, build it, and inspect `check.png` and the fitted probe. Rebuild
when you can name a visible mismatch or a failed fit. Once those checks pass, hand the part back. `f3d part` takes 4 to 40 s:
wait for it in one call (Codex: `yield_time_ms` 120000) instead of polling. Metres, Z up, the model faces -Y, +X
is the viewer's right in the front view.

## The pictures are a drawing, not photographs

A turnaround sheet is three separate drawings. They agree roughly, never exactly, and drawings cheat: a bow, an ear
or an eye is drawn turned toward the viewer in every view.

- **Build the real thing the drawing means**, one consistent 3D shape. The front picture says where things are and
  how wide and tall; the side picture adds how deep and the profile; where the two disagree the front wins.
- **Every drawn feature is painted from exactly one picture**: the one that faces it. An eye in the side picture is
  the same eye the front picture shows. Painting it again from the side gives the head a second pair.
- **Colours are read off the picture** (`V.color(u, v)`), never guessed.

## The script

```python
from forge3d import kit as k
S = k.slot()                    # your slot: S.lo, S.hi, S.center, S.size, S.box("front"), S.below, S.note
R = k.ref()                     # the pictures
F, SD = R["front"], R.side      # views: "front", "back", "right" / "left" (R.side = whichever exists)
navy = F.color(0.05, 0.22)      # the colour at u = 0.05, v = 0.22 (coordinates as on the grid of ref.png)
left, right = F.span(0.30)      # where the subject's outline is at height 0.30
```

Everything the script leaves in the scene is your part. The neighbours are in the scene too (finished parts where
they exist, grey blockout shapes otherwise) so you can lean on them: `S.below` is what you sit on,
`k.raycast_all(S.below, origin, direction)` finds its surface.

## Sculpt it

Write the part as real geometry, the way a modeller would build it, in code:

- **Surfaces from functions.** Rings or rows of points you compute, joined with `k.from_pydata(verts, faces)`: a
  skirt as a revolved profile with pleats (`r(theta, t)`), a sleeve as sections along an arm, a shoe as lofted
  rings. `k.loft(sections)`, `k.lathe(profile)`, `k.sweep(path, radii=[...], squash=)` for tubes, straps and
  locks, `k.sphere(radii=)`, `k.cyl`, `k.box(bevel=)`; `k.solidify` gives cloth its thickness, `k.subdiv`
  rounds, `k.mirror` makes the other one of a pair (build the one on +X).
- **Detail is geometry.** Pleats, ruffles, scalloped hems, lace with its holes, buttons, bows, stitching,
  embroidery as thin tubes lying on the cloth, fingers, shoe straps: build them. That is what makes the part
  read as the drawing from every side. Only what is printed or drawn on a surface (a face, an emblem) is painted.
- **One piece per thing the drawing outlines.** The assembled model gets the drawing's ink line around every
  separate piece, and `check.png` already shows your part with it. A lock of hair, a ruffle, a cuff, a bow loop
  that is its own object (or its own closed shell inside an object) gets its own line, as in the drawing; pieces
  fused into one skin lose it. So let pieces overlap and keep them separate: no `k.remesh` or boolean union
  across pieces. Thin pieces get a finer line automatically.
- **Proportions come from the pictures.** Read positions and sizes off the grid of `ref.png`; after each build the
  magenta outline in `check.png` shows where your part differs from the drawing.
- **Solid, and joined.** A body part is a closed solid that runs into what it attaches to: a leg up into the
  hips, a neck into the torso, a wrist into the cuff, well past the visible edge. Trace both thighs into
  the same pelvis in a front cut. When stockings own the lower leg, seat their tops on the corresponding
  left and right thighs. Cloth has thickness, and what
  it covers belongs to the body or limb slot. In `probe.png` follow each leg up into the hips and each wrist
  into its cuff. The surface row isolates your part. The fitted row shows it with finished attachments in
  colour. A waistband and a cuff have openings. If the fitted view reveals a missing limb, report its owner
  in your handoff. Keep the garment wearable and the anatomy in its own slot.
- **As fine as the drawing.** Build what the drawing shows at the size it shows it: every pleat, every button,
  knees and calves on a leg, sole, heel and straps on a boot, separate fingers. Your part will be judged next
  to the drawing by a reviewer who has not seen your script.
- Keep it light: a part over 200,000 triangles is too heavy; 20,000 to 80,000 is typical.

When the outline itself is the shape, take it from the picture: `k.inflate(mask, thickness=0.03)` for something
flat drawn broadside (a fin, a wing, a leaf), `k.volume(front_mask, side_mask)` for a plain rounded body. Masks:
`V.mask(box=(u0, v0, u1, v1))`, `V.mask(color="#4e62a4", tol=40)`, combined with `|`, `&`, `-`; then `.largest()`,
`.fill()`, `.hull()`, `.grow(d)`, `.shrink(d)`, `.symmetric("union")`. A part cut out where the picture shows it
can be turned and moved to where it really sits (`k.rotate`, `k.translate`); `k.drape(obj, S.below, view="front")`
lays a flat part onto the surface behind it.

## A character's face

- Establish the face proportions before the eyes are painted. On a front view mark the two visible skin edges
  at the widest exposed cheek, near the mouth and partway from mouth to chin. Keep their heights, each edge's
  offset from the midline, and the chin height. Use the inner and outer eye corners to keep eye spacing
  independent of jaw width. Hair, ears and the full head slot can all extend past those skin edges.
  If the drawing hides a landmark, use a visible one or leave that width as an estimate.
- Shape the skull, cheek and mandible as different cross-sections. A visible taper in front gives the widths;
  the profile supplies the chin projection, and the turn toward the ear controls fullness in three-quarter
  view. A broad chibi jaw, a square jaw and a long tapered jaw have different width profiles. Derive this
  profile for the subject at hand. Inspect `portrait_form.png` with the hair hidden and a single clay material.
  Compare its outline with exposed jaw segments and check its cheek planes in both turns.
- Enlarge the original front and profile to the same head height. Mark the inner and outer eye corners, pupil centres,
  brow tilt, nose tip, mouth corners and chin in reference coordinates. The eye opening, spacing and slope of the
  upper lid carry the expression. Preserve asymmetry that is actually in the drawing. Use the cheek outline and
  the profile to establish the jaw taper and the depth of the forehead, bridge, lips and chin.
- Give the cheek, eye sockets, upper lids and lip line shallow, continuous forms. Turn the far eye around the head
  so the nose and cheek progressively conceal it at an angle. Follow the drawing's visible sclera, iris rim, lid
  overlap, pupil and catchlight. For illustrated eyes, clean front-view eye pixels can be masked and mapped onto
  the curved surface; reconstruct areas hidden by bangs in the same style. Keep one iris per eye. Mask out hair
  and skin shadows cast by the fringe. A coarse enlarged bitmap may need a traced lid and iris at texture resolution.
- Choose skin colour from a lit cheek and compare it with the finished neck in the same light. Build a restrained
  lip contour and mouth line to match the expression. Inspect both three-quarter views for a flat far eye, heavy
  lash shelves, a missing mouth or a dark groove across the cheeks. Correct the underlying form or material.
- Inspect `portrait.png` with the finished hair when available. At assembly the same framing shows what the fringe
  covers. For a revision, keep the approved scalp, hairline and neck interface and fit the face within them.

## Hair, fur, manes

Hair is many locks, and the drawing separates them with lines. Built as one smooth mass it reads as a helmet.

- First a cap: a thin shell just outside the scalp in the hair's root colour, so no skin shows between locks.
- Then every lock the drawing shows, each its own tapered piece: the fringe locks over the forehead, ending
  where the drawing ends them; the side locks; the long locks of the back with their waves and curled tips; an
  ahoge. A lock is `k.sweep(path, radii=[...], squash=0.35..0.6)` along a path you read off the pictures, or a
  ribbon of rings; wide and overlapping at the roots, separate and pointed at the tips.
- **Trace the visible layers before adding fine strands.** In the back view, follow each prominent lock from
  its root through each crossing to its tip. Record which lock passes in front, the height of the tip and
  the gaps it leaves beside its neighbours. Use the profile to set the depth of each layer. Build these
  curves first, then add the fine strands visible in the drawing. Count visible tips again on the fitted
  back and three-quarter views. Buried duplicates add mesh without restoring any of those lines.
- The outer locks trace the outline of the front and back pictures; the side picture gives how far the mass
  stands off the head and the back. Fill behind them with larger, plainer locks, never with one blob.
- Both eyes stay clear: check the front view and both three-quarter tiles; a fringe lock that crosses an eye is
  too long or too far forward.
- Colour all locks in one call: `k.paint_part(locks, "front", keep=False, base="gradient")` gives the hair its
  colour as it changes with height in the picture (or name the colours bottom to top: `base=["#7295d2",
  "#607aba", "#4e62a4"]`). Drawn highlights: small flat pieces in the highlight colour lying on the locks.

## Colour

- **Plain**: `mat=k.flat("#30407a")` when creating, or `k.assign(obj, k.flat(...))`. Most pieces are plain.
  `k.flat` is half self-lit, so the colour reads as drawn; `k.mat(colour, "metal" | "glass" | ...)` is for real
  materials. A colour change is a new piece (a white cuff on a navy sleeve is two pieces).
- **A drawing on it** (a face, an emblem, a print): `k.paint_part(obj, "front", keep=mask, base=colour)`. `keep` =
  the pixels of the picture that belong to this part; the rest of the surface takes `base`. A face under a
  fringe: keep the skin colour plus boxes around the eyes, brows and mouth. `["front", "back"]` paints opposite
  faces from their own pictures.
- A colour that changes with height: `k.paint_part(objs, "front", keep=False, base="gradient")`.
- Colour only with `k.flat`, `k.mat` and `k.paint_part`: node materials you wire up yourself do not survive export.

## Reading `check.png`

Each reference row shows the drawing, the isolated part over grey neighbours, its full outline in magenta, and
the assembled view from the same camera when neighbours exist. Compare the visible portion in that last tile
where hair overlaps the hair cap or clothing hides a limb. The bottom row shows three-quarter angles.

- The magenta outline should run along the drawn outline of your part in the front view. In the other views it
  should agree as far as the drawing is one consistent thing; where a view cheats, keep the clean 3D shape.
- The three-quarter tiles are the real test: does it look like that part of the drawing, with its details, from
  an angle no picture shows? Pieces separated by their lines, drawn features whole and only once, nothing
  floating, no holes, no neighbour's colour on your part.
- The printed list: a part that leaves its slot or does not touch what it sits on will not fit the others.

## Done, and the report

Done: the outline follows the front picture to about a line's width; the three-quarter tiles look like the
drawing; every feature the brief names is there, once; it touches what it sits on; the list is empty or you can
say why not.

Reply with three lines and nothing else: what you built; what the last check says; what is still wrong,
simplified or guessed. If the slot itself is wrong (your part cannot fit it), do not move the part out: say by
how much and in which direction, the lead moves slots.

Not yours: other parts' folders, the blockout, exporting, lighting, the assembled model.
