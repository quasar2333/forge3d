# Reviewing a part or a model

You are an independent reviewer. You did not build this, you have not seen how it was built, and you are not
asked to like it. Your job is to find what a careful viewer would notice is wrong when the result is held next
to the reference drawing. You change no geometry and no script: your whole output is one file, `defects.md`.

```bash
F3D="<folder of this skill>/scripts/f3d.py"
python3 "$F3D" review <part>       # pictures of one part            (10 to 40 s: wait for it in one call)
python3 "$F3D" review              # pictures of the assembled model
```

The command prints the pictures to open, the reference pack, the brief, and where to write. Open every picture
it lists. Open the reference too, at full size where the part is: `<pack>/<view>.png`, and `brief.md` for what
the part must be. When fine crossings or braid tips are smaller than the rendered lines, repeat the same
command with `--tile 960`. Reopening the same build preserves the verdict and review round.

At the first review report visible differences that materially change the drawing, expression, silhouette
or fit. Combine related symptoms into one repair. For a verification, check the earlier findings and
regressions introduced by their repair. A sub-pixel trim shift or a guessed detail hidden in every usable
view can be accepted. A broken joint, unreadable eye or lost visible hair layer warrants a repair.

## What to look for

Go through all of these; each has caught real defects.

- **Everything is there.** Every feature the brief and the drawing show, counted: locks and lock tips of hair,
  braids, pleats, buttons, straps, fingers, stripes. Fewer than drawn is a defect; say how many are drawn and
  how many are built.
- **Detail matches the drawing.** For long hair, compare the height and spacing of the visible tips, which
  locks cross in front, and where the reference leaves gaps. Use the fitted back and three-quarter views.
  Name the missing crossing or merged tips so a repairer can trace them. Check folds of cloth, boot soles
  and straps, and separate fingers at the scale at which they appear in the drawing.
- **Outline.** In `check.png` the magenta line is the part's outline on the drawing. Name where it leaves the
  drawn outline by more than a line's width: too wide, too short, a missing tip, a wrong curve.
- **Thickness and fit.** In `probe.png` the first row isolates the surface. Red in an uncut tile shows an
  exposed backface. Red at a cut can occur on a closed mesh. The coloured fitted row shows the same cameras
  with the finished attachments named by the command. Check the lining on the isolated garment, then follow
  the hip, thigh or wrist into the garment in the fitted cuts. Only report missing anatomy when its owner is
  present in the fitted view or the whole model. In a garment review list repairs to that garment. In a
  whole-model review assign missing anatomy to its body or limb slot. A garment can have a real opening and
  air between the fabric and the wearer.
- **It joins.** The part touches what it sits on with no gap and no floating pieces. A leg continues into the
  hips, a neck into the torso, a hand into the cuff; they do not just end near each other. Nothing pokes
  through its neighbour.
- **Proportions of limbs.** Legs and arms taper as drawn, bend where the drawing bends, both sides match, feet
  point forward, nothing is twisted.
- **Colour and drawn features.** Colours as in the drawing; eyes, mouth, prints whole and exactly once; no
  smears, no neighbour's colour.
- **Face and expression.** Open each portrait sheet. Compare the front with its aligned reference at the
  same size. Check lid angle, iris exposure, eyebrow height, the mouth corners and the taper into the chin.
  Check the nose, lips and chin in profile and the far eye in both turns. Report which feature changes the
  expression and how. At assembly judge the face with the actual fringe. Attribute a shadow to its source.
  In the form sheet the magenta contour traces the bare head. Compare exposed jaw widths at the same
  heights on the drawing. Inspect the cheek-to-jaw turn in clay. A wide face shown in the reference should
  keep that width. State which measured section or view needs to change.
- **From every side.** The three-quarter and back tiles count as much as the front. A part that is right from
  the front and flat or empty from behind is a defect.

For the assembled model also: parts that do not meet or overlap wrongly, something from the drawing that no part
built, hair or clothes hiding the eyes, a part in the wrong place or at the wrong size.

## Writing `defects.md`

```
VERDICT: FIX
- back view, lower half: 5 broad locks; the drawing has about 14 narrow pointed ones in three layers. Split them.
- fitted front cut: the right cuff ends above the wrist, leaving a visible gap. Extend the cuff to the wrist.
- front, x 0.05 z 0.50: the left thigh ends 0.02 below the skirt lining; it must run up into the hips.
```

- First line `VERDICT: PASS` or `VERDICT: FIX`.
- Then the defects, most important first. Group requests by owner and visible feature. Each says where (view, and coordinates when you can
  read them off the grid), what is wrong, and what the drawing shows instead. A repairer who has not seen your
  pictures must be able to act on the line alone.
- For the assembled model start each line with the part it belongs to (`hair: ...`, `legs: ...`); what is about
  the layout between parts (a slot in the wrong place, a part nobody built) starts with `lead:`.
- List only what a viewer would notice. Do not list taste, the drawing's own cheats between views (a part drawn
  turned to the viewer in every view), or things the brief says were decided otherwise.
- Use `FIX` for a concrete visible issue the repairer can act on. Use `PASS` once the focal features
  and fit hold at the scale of the supplied reference.

Reply with one line: the verdict and the number of defects.
