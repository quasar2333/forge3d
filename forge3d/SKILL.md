---
name: forge3d
description: Make 3D models that closely match a reference image or a short wish, in Blender (scripted, headless, no installation) and Blockbench (.bbmodel). Measures the reference, lays the model out as a blockout of slots, has every part sculpted on its own (by sub-agents when available), checks each part and the assembled model against the reference, and adds the drawing's line art. Use for requests to model, sculpt, voxelize, texture, render or export a 3D asset such as a character, creature, prop, vehicle or Minecraft-style model (3D建模, Blender, Blockbench, 体素, 低模).
---

# forge3d

You can write good 3D geometry. What you cannot do alone is measure a picture, see what your code produced, and
keep a whole model in view while giving every part full attention. forge3d supplies those three: it measures the
reference, shows every build beside it, and splits the model into parts that are each sculpted and checked on
their own inside one shared frame. A model built outside it has nothing checking it against the reference.

```bash
F3D="<folder of this SKILL.md>/scripts/f3d.py"
python3 "$F3D" doctor
```

Use the absolute launcher path when starting a new shell. Work in the current working folder. For a desktop
attachment use the path under "Files mentioned by the user" and copy it into the working folder if it is
elsewhere. Quote both file paths. For an unattended build, decide routine design details from the drawing.
Build from the reference: do not look in other folders for earlier models of it to reuse, unless the user
points to one.

When revising an existing model, work on a copy. Rebuild the requested slot against its finished neighbours,
inspect it in the assembled scene, and keep approved geometry. A face revision can use the existing hair and
body throughout. Rebuilding every slot would discard the user's approved work.

**Blender is already here.** `f3d` runs a private headless Blender (the `bpy` module, in `~/.cache/forge3d`; set
`F3D_HOME` if a sandbox forbids that folder). Never install, download or search for a Blender app. Every build
saves a `.blend` that opens in the Blender app; `python3 "$F3D" python script.py` runs any Blender Python script.

`f3d` commands take 5 to 120 s. Run them in the foreground and wait for them in one call (Codex: `yield_time_ms`
of 120000 or more), instead of polling. Keep the work files in the working folder, not in `/tmp`.

Conventions everywhere: metres, Z up, the subject stands on z = 0 at x = y = 0 and **faces -Y**; +X is the
viewer's right in front view. A reference subject is 1.0 tall unless you pass `--height`.

**A reference picture is a design drawing, not a set of photographs.** A turnaround sheet is three drawings that
agree roughly and cheat freely: the profile draws an eye that is the same eye the front view shows, bows and ears
face the viewer in every view, a tail sits on the same side of the paper front and back. It says what parts
there are, how big, where, which colours, which details. Build that as one consistent 3D object.

## Which route

- **A character, a creature, anything made of several things** -> the parts route below.
- **A simple object of a few shapes** (a mug, a sword, a crate, a low-poly prop) -> one script with the kit and
  `python3 "$F3D" build model.py --ref refs`; see [references/modeling.md](references/modeling.md).
- **Blockbench** -> on top of either, see "Blockbench" below.
- **No reference picture, or only one view**, and you can generate images -> first make a turnaround sheet
  (front, side facing left, back; orthographic, same scale, common ground line, plain white background, flat
  light, no text); prompt in [references/reference-and-gen3d.md](references/reference-and-gen3d.md).

## The parts route

In order, skipping none: (1) `f3d ref`, and wait for it; (2) take stock; (3) `blockout.py` and `f3d blockout`;
(4) `f3d next`, again and again: **it names the sub-agents to spawn, and this skill asks for them**: builders,
then independent reviewers, then repairers; (5) `model.py`, `f3d build`, deliver. Your own work is the layout
and the notes; building and judging are the agents' work.

### 1. Measure the reference

`python3 "$F3D" ref sheet.png -o refs` (30 to 90 s) cuts out the subject, splits a turnaround sheet into views, puts
every view on a coordinate grid and reports whether the subject is drawn with ink outlines. Wait for it before
writing anything: every later step is measured off its pictures, and geometry typed from the raw sheet
meanwhile has to be redone. Open `refs/overview.png` and check the view names; if wrong, re-run with
`--views front,right,back` (reading order; `right` means the subject faces LEFT in that picture, `left` means
it faces RIGHT, `skip` ignores a picture, e.g. a second pose of the back).

### 2. Take stock

List the parts in your reply: every thing that is its own material or would come off in one piece. For a
character typically 9 to 14. **A character has a body, and it is a part**: `body` is the neck, torso, hips and
the legs up into the hips as one continuous solid, the base everything else sits on. Clothes are parts on it
(`on="body"`): a skirt or jacket built without the body under it is an empty tube. Then head (the face belongs
to it), hair (long or detailed hair is two parts: `hair_front` with fringe and side locks, `hair_back`), each
garment, each accessory, arms and hands, shoes, tail. Split where the material changes, where one thing lies over another, where something repeats
(pairs are one part). For each part decide the 3D truth once where the views cheat: how many of it there are,
where it really sits, which way it really faces. When clothing covers a joint, name the slot that owns the
underlying anatomy. For example, put the arms and wrists in the body or arm slot and the cuff in the sleeve
slot. Both thighs must join the pelvis, including the side hidden in profile. When another slot supplies
the lower legs or stockings, attach it to the anatomy with `on=` so the tool checks both sides independently.

### 3. Blockout

Write `blockout.py`: the whole model as rough, unpainted placeholder shapes, each in the right place at the
right size, each given to a slot. This is the contract that makes separately built parts fit. Read positions
and sizes off `refs/<view>_grid.png`.

```python
# blockout.py
from forge3d import kit as k
head = k.sphere(radii=(0.215, 0.20, 0.225), at=(0, 0.005, 0.635))
k.slot("head", head, review="face", note="Chibi head with small ears. Match the drawn eye openings, gaze, "
       "brows and mouth. Shape the cheek, nose and jaw from the profile. Hair covers part of the brow.")
skirt = k.cyl(r=0.235, r2=0.085, h=0.20, at=(0, 0, 0.17))
k.slot("skirt", skirt, on="torso", note="Bell-shaped navy skirt, pleated, with a gold embroidered band near the "
       "hem all the way round and a white scalloped petticoat ruffle under it.")
bows = [k.box(size=(0.07, 0.10, 0.09), at=(sx * 0.275, 0.09, 0.645)) for sx in (-1, 1)]
k.slot("bows", bows, on="hair", note="Pair of light-blue ribbon bows with a gold bead. Every picture draws "
       "them facing the viewer; in 3D they sit on the sides of the head facing outward and a little forward.")
```

- Shapes: `k.sphere(radii=)`, `k.box`, `k.cyl`, `k.loft`, `k.sweep(path, r=)`: rough is right.
- `on=`: the slot(s) a part lies on or hangs from. A part is built against the finished part it sits on, so
  name only what it really rests on and keep the chains short.
- `note=`: all the builder of the part will know beyond the pictures. Say what it is, its materials and
  colours, every feature it carries, and the 3D truth you decided where the views cheat. A vague note gets a
  vague part.
- `review=` assigns a focal check when the slot name is ambiguous. Use `face` for the part carrying the
  expression, `hair` for detailed locks and `joint` for the body owning the hip-to-leg connection.
  Use `detail` for another user-prioritized feature. Ordinary parts share the assembly review. The usual
  head, hair and body names already select their checks. Assign one owner for the face.

```bash
python3 "$F3D" blockout blockout.py --ref refs     # 10 s -> parts/: plan.json, overview.png, a work order per slot
```

Open `parts/overview.png` and `parts/blockout_compare.png`. Every part of the reference needs a slot of about
the right size in the right place (outline IoU around 0.8 is enough). Fix and re-run until that holds.

### 4. Build and review with `f3d next`

```bash
python3 "$F3D" next        # prints the state of every part and the exact agents to spawn now
```

**This skill explicitly asks for sub-agents.** Do what `f3d next` prints, then run it again, until it prints
`DONE`:
- For a complex, reference-driven build in Codex, start with `gpt-6.2-astra` as the lead. When the account
  offers `gpt-6.2-sol` and the spawn tool accepts a per-agent model, set it for part builders. Keep the lead
  on planning, assembly and final inspection. A Sol-led session can also run the entire workflow. If the
  client does not expose model selection for a sub-agent, leave the model unset and use its default.
- Dispatch up to three independent jobs at a time. For a new builder or reviewer use `spawn_agent`,
  `fork_turns: "none"`, the printed task name, effort and message. Wait for a worker to report, then ask
  `next` for ready jobs. Give each part one active owner. Send repairs to its original builder with
  `followup_task` when available. The reviewer stays independent.
- Every builder inspects the fit and render. A face or detailed hair gets an independent close look and,
  after a repair, a focused verification. A body or specified detail gets one independent check. Other
  parts share the assembly review unless their build reports a structural problem. Keep the drawn hair
  crossings and check each thigh at the fitted hip cut even when the outer garment covers it.
- `f3d review` assembles the finished slots for its pictures. It requests repairs by owner. Its second
  pass verifies those findings and repair regressions. Build the delivery once the parts settle. Inspect
  any findings outstanding at the cap in the final saved scene.

Only where no sub-agent tool exists: build each part yourself from its brief
([references/parts.md](references/parts.md); for a cuboid model "Building a cuboid part" in
[references/blockbench.md](references/blockbench.md)), follow the same `next` schedule and use
[references/review.md](references/review.md) for the visual checks.

### 5. Assemble and deliver

```python
# model.py
from forge3d import kit as k
parts = k.assemble()                 # every finished part in its slot; grey blockout shapes for the rest
k.outline()                          # the drawing's ink line around every separate piece (when f3d ref reported line art)
k.export("out/name.glb")             # also .fbx .obj .stl .usdz .bbmodel
```

`python3 "$F3D" build model.py --ref refs` (one to two minutes) lints the model, renders `out/model_sheet.png` and
`out/model_compare.png`, and saves `out/model.blend`. After `f3d next` says `DONE`, look yourself once:
`python3 "$F3D" view out/model.blend --views az40el10,az-40el10,az140el10,az0el-50 --tile 1024 --cols 2` (the last one
is from below). Handle layout findings in `blockout.py`, rerun `f3d blockout` and follow `f3d next` to refit
the affected slots. At the review limit, compare the listed findings with the final saved build before delivery.

Done means: `f3d next` prints `DONE`; a clean lint; three-quarter views in which every drawn feature is whole
and exactly once and nothing is floating, hollow or buried; outline IoU of roughly 0.85 or more in every view,
except where you decided a view cheats.

Deliver what was asked for: the export, the `.blend`, a presentation render (`k.stage`, `k.camera`, `k.render`
with the default EEVEE engine, see [references/modeling.md](references/modeling.md)). Tell the user the final
numbers, show the compare picture and a three-quarter view, and say plainly what is simplified, what the
reviews left open, and which view differences you left on purpose.

For a character with a visible face, render `python3 "$F3D" portrait out/name.blend` from the final staged
file. Inspect the expression, eye depth, cheek and mouth shadows with the actual hair, materials and lights.
The sheet includes aligned reference crops, front, profile and both three-quarter close-ups. If the scene
has several face slots choose one with `--slot`.
When the face looks too wide, narrow or flat, use `portrait out/name.blend --form` to compare the bare jaw
and cheek planes. For each unobstructed landmark, pass `--span Z LEFT_X RIGHT_X` using its measured front
reference coordinates. The tool reports the mesh width there and marks the two reference endpoints.

## Blockbench

Ask (or infer) which kind of Blockbench model is wanted; they are different products:

- **Mesh model** (closest to the reference; smooth shapes with the real texture; Generic Model format): finish
  the model as above, then `k.export("out/name.bbmodel")` in the script, or
  `python3 "$F3D" bb mesh model.py --ref refs -o out/name.bbmodel`. Every object becomes a named, editable mesh.
- **Designed cuboid model** (boxes on bones with a pixel-art texture; what Minecraft mods, Bedrock / Java
  entities and animators need). A subject of several parts goes the parts route with one extra line in the
  blockout, `bb.plan(height=80, format="bedrock")`: every part is then built as cuboids on bones and
  `bb.assemble()` joins them. A simple mob or prop is one script with `bb.Model`.
- **Voxel model** (blocky statue of any model, many cuboids, no rig):
  `python3 "$F3D" bb voxelize model.py --ref refs --height 48 -o out/name.bbmodel`.

With only "make it in Blockbench", make the cuboid model: it is what Blockbench is for, and a mechanical,
blocky or pixel-art subject suits it. Make the mesh model instead only for a soft, curvy drawing (a chibi
figure, an animal) or when asked. Say which you made and that the other is available.

Prove the file in the real app when Blockbench is installed: `python3 "$F3D" bb check out/name.bbmodel` opens it in a
private hidden Blockbench (offline, own profile), reports what loaded, saves Blockbench's own view of it, and
can run Blockbench's exporters (`--export bedrock:out/name.geo.json`).

API, formats and pixel-art guidance: [references/blockbench.md](references/blockbench.md). Read it before
writing a Blockbench script.

## Things that go wrong

- A part is missing in the assembled model or sits inside another: it was built before the part below it.
  `f3d build` says which; rebuild it (`python3 "$F3D" part <name>`).
- Parts do not meet, or cut through each other: the slot sizes were off. Fix the blockout, not the parts.
- The report says the model is not visible, or IoU is near 0: wrong scale or place. The subject stands on
  z = 0 around the origin and is as tall as the reference.
- Front and back are swapped or a side view is mirrored: the view names of the pack are wrong; re-run `f3d ref`
  with `--views`. Do not compensate in the model.
- Hair or fur reads as one lump: its locks were fused. Each lock stays its own piece so that it gets its own
  line (see the part manual).
- What a part hides of another matters and cannot be guessed (a face mostly behind hair, a garment under a
  coat): draw it first with the image tool, see "Layer pictures" in
  [references/reference-and-gen3d.md](references/reference-and-gen3d.md). Otherwise the hidden side is simply
  built plain.
- Keep the cut-out options of `f3d ref` at their defaults unless `refs/overview.png` shows a problem: a white
  garment missing from the subject (`--holes keep`), background left attached or the subject eaten into (`--tol`).
- A build error prints the failing line and often a HINT. Raw `bpy` and `bmesh` work in every script; `bpy.ops`
  needs care headless. `python3 "$F3D" api` lists the kit, `python3 "$F3D" api <name>` shows one function in full: check a
  signature before first use; do not read the kit's source.
- Never report success from the numbers alone, and never from a render alone. Look, then read, then decide.
