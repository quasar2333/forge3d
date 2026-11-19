"""f3d command line. Run through the launcher:  python3 <skill>/scripts/f3d.py <command> ..."""
import argparse
import json
import os
import sys
import time


def _out_dir(path):
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def _rel(path):
    try:
        r = os.path.relpath(path)
        return r if not r.startswith("..") else path
    except ValueError:
        return path


def _fail(msg, code=1):
    print(msg)
    raise SystemExit(code)


# ---------------------------------------------------------------- ref
def cmd_ref(a):
    from . import refs
    views = [v for v in a.views.split(",")] if a.views else None
    out = _out_dir(a.out)
    doc = refs.build_pack(a.images, out, views=views, height=a.height, bg=a.bg, tol=a.tol, scale=a.scale,
                          holes=a.holes, regions=a.regions, sharpen=a.sharpen)
    print(refs.describe(doc, _rel(out)))
    if a.regions and a.table:
        for n in doc["order"]:
            print(refs.region_table(doc, n))
    print("LOOK AT: %s  (all views with their coordinate grids)" % _rel(os.path.join(out, "overview.png")))
    print("NEXT: model against it with  f3d build model.py --ref %s  (the pack feeds k.ref(), k.paint_from_ref() and "
          "the comparison)" % _rel(out))


# ---------------------------------------------------------------- layers
def _layer_dir(base, name):
    return base.rstrip(os.sep) + "_" + name


def cmd_layers(a):
    import json
    from . import layers
    base = os.path.abspath(a.ref)
    if not os.path.isfile(os.path.join(base, "ref.json")):
        _fail("LAYERS: %s is not a reference pack (run `f3d ref` on the whole picture first)" % _rel(base), 2)
    items = []
    for item in a.layers:
        name, _, pic = item.partition("=")
        name = name.strip()
        if not name.replace("_", "").replace("-", "").isalnum():
            _fail("LAYERS: layer names are plain words such as body, hair, headwear (got %r)" % name, 2)
        d = _layer_dir(base, name)
        if not pic and os.path.isfile(os.path.join(d, "ref.json")):
            with open(os.path.join(d, "ref.json")) as f:
                pic = (json.load(f).get("sources") or [""])[0]
        if not pic or not os.path.isfile(pic):
            _fail("LAYERS: give the picture of every layer:  %s=<picture.png>%s" % (
                name, "  (no such file: %s)" % pic if pic else ""), 2)
        items.append((name, pic, d))
    if len(items) < 2:
        _fail("LAYERS: name at least two layers, inner first, all in one call:  f3d layers refs body=body.png "
              "hair=hair.png headwear=headwear.png", 2)
    rep = layers.split(base, items, bg=a.bg, tol=a.tol, holes=a.holes, align=not a.no_align)
    with open(os.path.join(base, "layers.json"), "w") as f:
        json.dump(dict(layers=[os.path.relpath(d, base) for _n, _p, d in items], report=rep), f, indent=1)
    print("LAYERS of %s, inner to outer (every pixel of the original now belongs to exactly one of them):" % _rel(base))
    for name, _pic, d in items:
        st = rep["layers"][name]
        al = st["align"]
        print("  %-9s -> %s  (%s)  %2.0f %% of it is the original's own pixels, %2.0f %% was hidden and comes from "
              "your layer picture" % (name, _rel(d), ", ".join(st["views"]), st["original"] * 100, st["drawn"] * 100))
        if st["agreed"] < 0.7:
            print("      NOTE: the layer picture redrew this layer differently from the original (only %.0f %% of the "
                  "visible part agrees; found at scale x%.3f). What the original shows is kept anyway; if the newly "
                  "drawn hidden parts do not continue it well in layers_check.png, draw the layer again."
                  % (st["agreed"] * 100, al["scale"]))
        if st["outside"] > 0.12:
            print("      NOTE: %.0f %% of the layer picture lies outside the original's outline and was cut off: it "
                  "was moved or resized. Look at this layer in layers_check.png." % (st["outside"] * 100))
    worst = max(rep["views"].items(), key=lambda kv: kv[1]["adopted"])
    if worst[1]["adopted"] > 0.02:
        print("  %.0f %% of the %s view was in no layer picture (white outlines in the check picture) and was given to "
              "the layer around it. If that is a whole part (a tail, a prop), it needs its own layer." % (
                  worst[1]["adopted"] * 100, worst[0]))
    print("LOOK AT: %s  (original | who owns what | every layer complete)" % _rel(rep["picture"]))
    print("NEXT: the layer packs show every part alone and complete: cut shapes and drawings from them with "
          "k.layer_ref(\"%s\") etc. A generated mesh for an organic layer (hair, fur):  f3d gen3d %s_<layer>" % (
              items[0][0], _rel(base)))


# ---------------------------------------------------------------- blockout / part
def _spawn_text(parts_dir, name):
    return ("You build one part of a 3D model: `%s`. Read %s, then do exactly what it says. Work only inside %s. "
            "Reply with three lines when the part checks out." % (
                name, _rel(os.path.join(parts_dir, name, "brief.md")), _rel(os.path.join(parts_dir, name))))


def cmd_blockout(a):
    from . import parts, runner, compare
    script = os.path.abspath(a.script)
    parts_dir = os.path.abspath(a.out or os.path.join(os.path.dirname(script), "parts"))
    ref = os.path.abspath(a.ref) if a.ref else None
    try:
        rep = parts.blockout(script, ref, parts_dir)
    except runner.BuildError as e:
        _fail(str(e), 2)
    except parts.PartsError as e:
        _fail(str(e), 2)
    plan = rep["plan"]
    print("BLOCKOUT ok  %s -> %s  (%d slots, %.1fs)" % (_rel(script), _rel(parts_dir), rep["slots"], rep["time"]))
    if rep["loose"]:
        print("  WARN  %d shape(s) are in no slot and will not be built by anyone: %s" % (
            len(rep["loose"]), ", ".join(rep["loose"][:8])))
    if rep["compare"]:
        for r in rep["compare"]["views"]:
            print("  [%s] outline IoU %.2f  missing %.0f%%  extra %.0f%%" % (r["view"], r["iou"], r["missing"] * 100,
                                                                         r["extra"] * 100))
        worst = min(r["iou"] for r in rep["compare"]["views"])
        print("  a blockout is rough: IoU around 0.8 is enough%s. What matters is that every part of the reference "
              "has a slot of about the right size in the right place." % (
                  "" if worst >= 0.75 else "; this one is further off, fix the sizes and places first"))
    for w in sorted(rep["waves"]):
        print("  wave %d: %s" % (w, ", ".join(
            n + (" (on %s)" % ", ".join(plan["slots"][n]["on"]) if plan["slots"][n]["on"] else "") for n in rep["waves"][w])))
    if rep["stale"]:
        print("  NOTE: these parts were built for a slot that has now moved; build them again: " + ", ".join(rep["stale"]))
    print("LOOK AT: %s  (the blockout, one colour per slot)%s" % (
        _rel(rep["overview"]), "  and  %s  (against the reference)" % _rel(rep["compare"]["image"]) if rep["compare"] else ""))
    print("NEXT: hand out the parts, each to its own fresh agent (work orders: %s/<slot>/brief.md), three agents at "
          "a time, starting with wave 1; a part can start once what it sits on is finished (`f3d part` lists who "
          "can start now). Message for a part's agent:" % _rel(parts_dir))
    print("    " + _spawn_text(parts_dir, plan["order"][0]))
    print("  Without agents to hand out to: build the parts yourself in that order, one at a time, each from its brief.")
    if len(plan["order"]) >= 4 and not any(plan["slots"][n]["on"] for n in plan["order"]):
        print("  WARN  no slot sits on another (no on=). Parts built without the part they rest on do not meet: give "
              "clothes on=\"body\", hair on=\"head\", and so on, and run this again.")
    print("  From here `f3d next` says who to spawn: builders, then independent reviewers, then repairers.")
    if plan.get("bb"):
        print("  Then a final script with  m = bb.assemble(); m.save(\"out/name.bbmodel\")  for  f3d build.")
    else:
        print("  Then a final script with k.assemble() (and k.outline() for a reference drawn with ink lines) for  f3d build.")


def _find_parts(a):
    from .kit import partkit
    for cand in ([a.parts] if a.parts else []) + ["parts", "."]:
        d = os.path.abspath(cand)
        if os.path.isfile(os.path.join(d, partkit.PLAN)):
            return d
    _fail("PART: no blockout plan here (looked for parts/plan.json). Run `f3d blockout blockout.py --ref refs` "
          "first, or pass --parts <folder>.", 2)


def cmd_part(a):
    from . import parts, runner
    name = a.name
    if name and (name.endswith(".py") or os.sep in name):
        p = os.path.abspath(name)
        d = p if os.path.isdir(p) else os.path.dirname(p)
        a.parts, name = os.path.dirname(d), os.path.basename(d)
    parts_dir = _find_parts(a)
    if not name:
        rows = parts.status(parts_dir)
        print("PARTS in %s:" % _rel(parts_dir))
        for n, w, state, note in rows:
            print("  wave %d  %-14s %-18s %s" % (w, n, state, note))
        go, wait = parts.ready(parts_dir)
        stale = [n for n, _w, s, _n in rows if s == "stale"]
        if go or wait:
            print("NEXT: %s%s" % (
                "can start now: " + ", ".join(go) if go else "nothing can start yet",
                "".join("; %s waits for %s" % (n, " and ".join(m)) for n, m in wait)))
        elif stale:
            print("NEXT: build again: " + ", ".join(stale))
        else:
            from .kit import partkit
            join = "m = bb.assemble(); m.save(\"out/name.bbmodel\")" if partkit.load_plan(parts_dir).get("bb") \
                else "k.assemble()"
            print("NEXT: every part is built: put them together with %s in a build script "
                  "(f3d build model.py --ref <pack>)" % join)
        return
    try:
        rep = parts.part(parts_dir, name, render=not a.no_render)
    except runner.BuildError as e:
        _fail(str(e), 2)
    except parts.PartsError as e:
        _fail(str(e), 2)
    if rep.get("refreshed"):
        print("NOTE: layer packs were added since the work orders were written. %s and ref.png now show them "
              "(the part alone, complete where the whole picture hides it): read both again." % _rel(
                  os.path.join(parts_dir, name, "brief.md")))
    warn = [i for i in rep["issues"] if i[0] in ("ERROR", "WARN")]
    size = "%d cube(s) on %d bone(s)" % (rep["cubes"], rep["bones"]) if rep.get("cubes") else \
        "%d object(s), %s triangles" % (rep["objects"], format(rep["tris"], ","))
    print("PART %s built  (%s, %.1fs)  %s" % (
        name, size, rep["time"], "checks clean" if not warn else "%d thing(s) to fix" % len(warn)))
    b = rep["bbox"]
    print("  it spans x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f" % (b["lo"][0], b["hi"][0], b["lo"][1], b["hi"][1],
                                                                 b["lo"][2], b["hi"][2]))
    for m, g in rep["touch"].items():
        if g is not None and g <= 0.006:
            print("  touches %s" % m)
    for level, _code, msg in rep["issues"]:
        print("  %-5s %s" % (level, msg))
    if rep["check"]:
        print("LOOK AT: %s  (per view: reference | isolated part | isolated outline | in place; last row: "
              "three-quarter views alone and in place)" % _rel(rep["check"]))
        print("LOOK AT: %s  (uncut surfaces and sections; the fitted row includes finished attachments)" %
              _rel(os.path.join(parts_dir, name, "probe.png")))
        if rep.get("portrait"):
            print("LOOK AT: %s  (front, profile and both turns at face scale)" % _rel(rep["portrait"]))
            print("LOOK AT: %s  (bare outline on the drawing and the clay face from both turns)" % _rel(rep["form"]))
    print("NEXT: fix what the picture and the list above show, then run this again. The part is done when its "
          "outline follows the reference in every view, its colours are clean from the three-quarter views, and "
          "nothing above is left to fix.")
    if any(i[0] == "ERROR" for i in rep["issues"]):
        raise SystemExit(1)


def _review_text(parts_dir, name, previous=None):
    from . import parts
    what = "one part of a 3D model: `%s`" % name if name else "an assembled 3D model"
    message = ("You review %s independently. Read `%s`. Run `%s` in the project working folder and inspect the "
            "pictures it lists. Reply with the verdict and number of defects." % (
                what, parts._skill_file("references", "review.md"), parts._parts_command("review", parts_dir, name)))
    if previous:
        message += " Verify these earlier findings and any visible regression caused by their repair:\n" + "\n".join(
            "- " + item for item in previous)
    return message


def _fix_text(parts_dir, name, defects, report):
    d = _rel(os.path.join(parts_dir, name))
    return ("You repair `%s`. Read %s/brief.md. Review report: %s. Your assigned defects:\n%s\n"
            "Edit %s/part.py, build it using the command in the brief, and inspect the result. Work only inside %s. "
            "Report what changed, what the check shows, and any defects that remain." %
            (name, d, _rel(report), "\n".join("- " + x for x in defects), d, d))


def cmd_review(a):
    from . import parts
    parts_dir = _find_parts(a)
    try:
        rep = parts.review(parts_dir, a.name, tile=a.tile)
    except parts.PartsError as e:
        _fail(str(e), 2)
    d = _rel(rep["dir"])
    print("REVIEW of %s, round %d" % ("part `%s`" % a.name if a.name else "the assembled model", rep["round"]))
    if rep["reused"]:
        print("Reusing the builder's pictures: geometry, fitted neighbours and reference match.")
    for p in rep["pictures"]:
        print("LOOK AT: %s" % _rel(p))
    if rep["ref"]:
        print("REFERENCE: %s/overview.png  (and <view>.png in that folder at full size)" % _rel(rep["ref"]))
    if a.name:
        print("BRIEF: %s/brief.md  (what the part must be)" % d)
        print("FITTED: %s" % (", ".join(rep["fitted"]) or "no finished attachments yet; check the joint at assembly"))
    else:
        for owner in rep["parts"]:
            report = os.path.join(parts_dir, owner, "report.json")
            if os.path.isfile(report):
                with open(report) as f:
                    for level, code, message in json.load(f).get("issues", []):
                        if level == "ERROR" or (level == "WARN" and code in parts.STRUCTURAL_ISSUES):
                            print("VERIFY IN PLACE: %s: %s" % (owner, message))
    if rep["inside"] > 0.04:
        print("NOTE: %.0f %% backfaces in the uncut surface views (red in probe.png); check normals and openings" % (
            rep["inside"] * 100))
    if rep["current"]:
        print("CURRENT: %s/defects.md already covers this build. Its verdict and round are preserved." % d)
        return
    if rep["at_limit"]:
        print("LIMIT: pictures refreshed for final verification. The review history is preserved.")
        return
    print("WRITE: %s/defects.md  (first line `VERDICT: PASS` or `VERDICT: FIX`, then the defects as a list%s)" % (
        d, "" if a.name else ", each starting with the name of its part: " + ", ".join(rep["parts"])))


def cmd_portrait(a):
    """Inspect the actual delivery materials, lights and ink at the scale of the face."""
    import bpy
    from . import parts, runner
    from .kit import partkit, refkit
    parts_dir = _find_parts(a)
    plan = partkit.load_plan(parts_dir)
    names = [a.slot] if a.slot else [n for n in plan["order"] if parts.review_focus(plan, n) == "face"]
    if len(names) != 1 or names[0] not in plan["slots"]:
        _fail("PORTRAIT: choose the face slot with --slot NAME. Slots: " + ", ".join(plan["order"]), 2)
    source = os.path.abspath(a.source)
    if not source.lower().endswith(".blend"):
        _fail("PORTRAIT: pass a saved Blender scene with its presentation lighting.", 2)
    ref = refkit.Ref(os.path.join(parts_dir, plan["ref"])) if plan.get("ref") else None
    try:
        runner.load_source(source)
        scene = bpy.context.scene
        scene.render.resolution_percentage = 100
        scene.render.use_border = False
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = min(scene.cycles.samples, 16)
        form = a.form or bool(a.span)
        suffix = "_portrait_form.png" if form else "_portrait.png"
        path = os.path.abspath(a.out or os.path.splitext(source)[0] + suffix)
        if os.path.splitext(path)[1].lower() != ".png":
            _fail("PORTRAIT: the output is a .png file.", 2)
        parts.portrait_picture(partkit.Slot(parts_dir, plan, names[0]), ref, path,
                               tile=a.tile, studio=True, form=form, spans=a.span or ())
    except (runner.BuildError, parts.PartsError) as e:
        _fail(str(e), 2)
    print("LOOK AT: %s  (%s)" % (_rel(path), "bare face in clay with reference outlines" if form else
                                 "saved materials and lighting, reference crops, both turns"))


def cmd_next(a):
    from . import parts
    parts_dir = _find_parts(a)
    st = parts.stages(parts_dir)
    md = os.path.join(parts_dir, parts.MODEL)
    mrounds = parts._rounds(md)
    print("STATE of %s  (focused part checks, then an assembled visual review)" % _rel(parts_dir))
    for r in st:
        print("  %-14s %-7s %s" % (r["name"], r["stage"], "focus %s, local reviews %d, budget %d%s%s" % (
            r["focus"], r["round"], r["limit"], "; waits for " + ", ".join(r["waits"]) if r["stage"] == "build" and r["waits"] else "",
            "; open: %d" % len(r["defects"]) if r["defects"] else "")))
    todo = []
    for r in st:
        effort = "high" if r["focus"] in ("face", "hair", "joint") else "medium"
        if r["stage"] == "build" and not r["waits"]:
            msg = _spawn_text(parts_dir, r["name"])
            if r["stale"]:
                msg = "The blockout moved this slot. Refit the existing part to its updated brief. " + msg
            todo.append((("refit_" if r["stale"] else "part_") + r["name"], effort, msg))
        elif r["stage"] == "review":
            todo.append(("review_%s_%d" % (r["name"], r["round"] + 1), "medium",
                         _review_text(parts_dir, r["name"], r.get("previous"))))
        elif r["stage"] == "fix":
            tag = "m%d" % mrounds if r.get("why") == "model" else "%d" % r["round"]
            msg = _fix_text(parts_dir, r["name"], r["defects"], r["report"])
            todo.append(("fix_%s_%s" % (r["name"], tag), effort, msg))
    if todo:
        # a task that stays listed for a long time has lost its agent: the lead must not wait for it for ever
        seen_file = os.path.join(parts_dir, ".next.json")
        try:
            with open(seen_file) as f:
                seen = json.load(f)
        except (OSError, ValueError):
            seen = {}
        now = time.time()
        seen = {k: seen.get(k, now) for k, _e, _m in todo}
        with open(seen_file, "w") as f:
            json.dump(seen, f)
        print("NEXT: dispatch up to three available jobs. Use fresh builders and independent reviewers with the "
              "effort shown and fork_turns \"none\". Send a fix to that part's builder as a follow-up if available; "
              "otherwise spawn a repairer. Skip jobs with an active owner, then run `f3d next` again:")
        for task, effort, msg in todo:
            late = now - seen[task] > 2400
            print("  task_name %s   reasoning_effort %s%s\n    %s" % (
                task, effort, "   (listed for %d minutes: if its agent is gone, do this task yourself now)" % (
                    (now - seen[task]) // 60) if late else "", msg))
        return
    if any(r["stage"] != "done" for r in st):
        print("NEXT: wait for the agents that are working, then run `f3d next` again.")
        return
    _report, verdict, items, reviewed = parts._review_record(md, parts.MODEL_ROUNDS)
    newest = max(os.path.getmtime(parts.partkit.part_file(parts_dir, r["name"])) for r in st)
    fresh = verdict is not None and reviewed >= newest
    if verdict is None or (not fresh and mrounds < parts.MODEL_ROUNDS):
        print("NEXT: review the assembled parts directly. Spawn one independent agent:\n  task_name "
              "review_model_%d   reasoning_effort high\n    %s\n  then run "
              "`f3d next` again." % (
                  mrounds + 1, _review_text(parts_dir, None, items if verdict == "fix" else None)))
        return
    owners = {r["name"] for r in st}
    yours = [x for x in items if x.partition(":")[0].lower() not in owners] if verdict == "fix" else []
    late = [r["name"] for r in st if r.get("unverified") and
            reviewed < os.path.getmtime(parts.partkit.part_file(parts_dir, r["name"]))]
    if fresh and yours:
        print("NEXT: lead, resolve the following assembly findings and rebuild the affected slots. "
              "Then run `f3d next` again.")
        for x in yours:
            print("    - " + x)
        return
    print("DONE: review rounds complete." if fresh else
          "DONE: review limit reached. The last assembly verdict predates the latest repairs. "
          "Verify these findings on the final build:")
    if not fresh:
        for x in items:
            print("    - " + x)
    if late:
        print("  repaired after their last review and not reviewed again: %s. Look at them yourself." % ", ".join(late))
    print("  Build the delivery, inspect its three-quarter views, joints and any staged face portrait.")


# ---------------------------------------------------------------- build / view / compare
def _load(source, a):
    from . import runner
    try:
        return runner.load_source(source, ref=getattr(a, "ref", None))
    except runner.BuildError as e:
        _fail(str(e), 2)


def _sheet(a, out_png, title=None):
    from . import sheet
    path, infos = sheet.make_sheet(out_png, views=a.views or sheet.DEFAULT_VIEWS, tile=a.tile, cols=a.cols, title=title)
    return path


def cmd_build(a):
    from .kit import io, qa
    t0 = time.time()
    script = os.path.abspath(a.script)
    stem = os.path.splitext(os.path.basename(script))[0]
    out = _out_dir(a.out or os.path.join(os.path.dirname(script), "out"))
    _load(script, a)
    print("BUILD ok  %s  (%.1fs)" % (_rel(script), time.time() - t0))
    res = qa.lint()
    blend = io.save(os.path.join(out, stem + ".blend"))
    looks = []
    bad = False
    if not a.no_render:
        looks.append((_sheet(a, os.path.join(out, stem + "_sheet.png"), stem), "the model from all sides"))
    if a.ref and not a.no_render:
        from . import compare
        rep = compare.run(a.ref, os.path.join(out, stem + "_compare.png"))
        print(compare.format_report(rep))
        looks.insert(0, (rep["image"], "reference | model | difference for every reference view"))
        from . import parts
        blame = parts.format_blame(parts.blame(a.ref))
        if blame:
            print(blame)
        from . import features
        eyes = features.eye_check(a.ref, os.path.join(out, stem + "_eyes.png"))
        if eyes is not None:
            print(features.format_eye_check(eyes))
            if eyes["errors"]:
                bad = True
                looks.insert(0, (eyes["image"], "the eyes found on the model from eight directions"))
    print("SAVED %s" % _rel(blend))
    for p, why in looks:
        print("LOOK AT: %s  (%s)" % (_rel(p), why))
    if not res["ok"] or bad:
        raise SystemExit(3)


def cmd_view(a):
    kind = _load(os.path.abspath(a.source), a)
    stem = os.path.splitext(os.path.basename(a.source))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), stem + "_sheet.png")
    if os.path.isdir(out):
        out = os.path.join(out, stem + "_sheet.png")
    elif os.path.splitext(out)[1].lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        out += ".png"
    from .kit import qa
    if a.lint:
        qa.lint()
    path = _sheet(a, os.path.abspath(out), stem)
    print("LOOK AT: %s" % _rel(path))


def cmd_compare(a):
    from . import compare
    _load(os.path.abspath(a.source), a)
    stem = os.path.splitext(os.path.basename(a.source))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.source)), stem + "_compare.png")
    rep = compare.run(a.ref, os.path.abspath(out), views=a.views.split(",") if a.views else None)
    print(compare.format_report(rep))
    print("LOOK AT: %s  (reference | model | difference)" % _rel(rep["image"]))


# ---------------------------------------------------------------- gen3d
def cmd_gen3d(a):
    from . import gen3d
    if a.list:
        for k, v in gen3d.BACKENDS.items():
            print("%-11s %s" % (k, v["note"]))
        return
    import json
    if a.out and len(a.source) > 1:
        _fail("GEN3D: -o names one file; with several packs leave it out (each goes to gen/<layer>.glb)", 2)
    failed = 0
    for source in a.source:
        src = os.path.abspath(source)
        stem = "base"
        if os.path.isfile(os.path.join(src, "ref.json")):
            with open(os.path.join(src, "ref.json")) as f:
                stem = json.load(f).get("layer") or "base"
        out = a.out or os.path.join(os.path.dirname(src.rstrip(os.sep)), "gen", stem + ".glb")
        try:
            path, meta = gen3d.generate(src, out, backend=a.backend, quality=a.quality, seed=a.seed, force=a.force)
        except gen3d.GenError as e:
            if len(a.source) == 1:
                _fail("GEN3D FAILED: %s" % e, 4)
            print("GEN3D FAILED for %s: %s" % (_rel(src), e))
            failed += 1
            continue
        print("MESH %s  (backend %s, %s)" % (_rel(path), meta["backend"], "textured" if meta["textured"] else
                                             "untextured: the build colours it from its pictures"))
        if stem == "base":
            print("NEXT: in the build script  base = k.adopt(%r)  then  k.paint_from_ref(base)  and run  f3d build" % _rel(path))
        else:
            print("NEXT: in the build script  %s = k.layer(%r)" % (stem, stem))
    if failed:
        print("GEN3D: %d of %d meshes made. For the others: run this command again later (finished meshes are kept), "
              "or go on without: k.layer(name) shapes a layer that has no mesh from its own outlines."
              % (len(a.source) - failed, len(a.source)))
        raise SystemExit(4)


# ---------------------------------------------------------------- blockbench
def cmd_bb(a):
    if a.action == "voxelize":
        from .bb import voxel
        src = os.path.abspath(a.source)
        _load(src, a)
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.abspath(a.out or os.path.join(os.path.dirname(src), stem + ".bbmodel"))
        m = voxel.voxelize(height=a.height or 40, name=stem, ref=a.ref, colors=a.colors or None, format=a.format)
        path = m.save(out)
        print("BBMODEL %s  (%d cuboids)" % (_rel(path), len(m.cubes)))
        print("NEXT: f3d compare %s --ref <pack>   to see it against the reference;  f3d bb check %s   to open it in "
              "the real Blockbench" % (_rel(path), _rel(path)))
        return
    if a.action == "mesh":
        from .bb import mesh
        src = os.path.abspath(a.source)
        _load(src, a)
        stem = os.path.splitext(os.path.basename(src))[0]
        out = os.path.abspath(a.out or os.path.join(os.path.dirname(src), stem + ".bbmodel"))
        path = mesh.save_scene(out, height=a.height or 64, faces=a.faces)
        print("BBMODEL %s" % _rel(path))
        print("NEXT: f3d compare %s --ref <pack>;  f3d bb check %s   to open it in the real Blockbench" % (
            _rel(path), _rel(path)))
        return
    if a.action == "check":
        from .bb import app
        src = os.path.abspath(a.source)
        shots = os.path.abspath(a.shots or os.path.splitext(src)[0] + "_blockbench.png")
        try:
            with app.Blockbench() as bbx:
                info = bbx.open(src)
                what = "%d cubes" % info["cubes"]
                if info.get("meshes"):
                    what += ", %d meshes (%d faces)" % (info["meshes"], info["mesh_faces"])
                print("BLOCKBENCH %s opened %s: format %s, %s, %d groups, textures %s" % (
                    info["version"], os.path.basename(src), info["format_name"], what, info["groups"],
                    ", ".join("%s %dx%d" % (t["name"], t["width"], t["height"]) for t in info["textures"]) or "none"))
                if info.get("mesh_faces_untextured"):
                    print("  note: %d mesh faces have no texture" % info["mesh_faces_untextured"])
                if not info["cubes"] and not info.get("meshes"):
                    print("  PROBLEM: Blockbench found no cubes or meshes in this file")
                for pr in info["problems"][:12]:
                    print("  " + pr)
                if info["untextured_faces"]:
                    print("  note: %d faces have no texture assigned" % info["untextured_faces"])
                import cv2
                import numpy as np
                tiles = []
                tmp = os.path.splitext(shots)[0]
                for ang in ("north", "west", "south", "isometric_right"):
                    f = bbx.screenshot("%s_%s.png" % (tmp, ang), ang, 520)
                    tiles.append(cv2.imread(f))
                    os.remove(f)
                cv2.imwrite(shots, np.hstack(tiles))
                print("LOOK AT: %s  (Blockbench's own viewport: front, side, back, isometric)" % _rel(shots))
                for spec in a.export or []:
                    codec, _, dst = spec.partition(":")
                    if not dst:
                        _fail("--export wants codec:path, e.g. bedrock:out/model.geo.json")
                    f = bbx.export(codec, os.path.abspath(dst))
                    print("EXPORTED %s  (Blockbench %s exporter)" % (_rel(f), codec))
        except app.BlockbenchError as e:
            _fail("BB CHECK unavailable: %s" % e, 5)
        return


# ---------------------------------------------------------------- api
def cmd_api(a):
    import inspect
    from . import kit
    names = [n for n in dir(kit) if not n.startswith("_")]
    if a.name:
        hits = [n for n in names if a.name.lower() in n.lower()]
        exact = [n for n in hits if n.lower() == a.name.lower()]
        for n in (exact or hits):
            obj = getattr(kit, n)
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sig = "(...)"
                print("k.%s%s" % (n, sig))
                doc = inspect.getdoc(obj)
                if doc:
                    print("    " + doc.replace("\n", "\n    "))
                print()
            elif isinstance(obj, dict):
                print("k.%s = {%s}" % (n, ", ".join(sorted(map(str, obj)))))
        if not hits:
            print("no kit function matches %r. `f3d api` lists everything." % a.name)
        return
    groups = {}
    for n in names:
        obj = getattr(kit, n)
        if inspect.ismodule(obj):
            continue
        mod = getattr(obj, "__module__", "") or ""
        groups.setdefault(mod.split(".")[-1] if callable(obj) else "constants", []).append(n)
    for g in sorted(groups):
        print("[%s]" % g)
        for n in groups[g]:
            obj = getattr(kit, n)
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    sig = "(...)"
                first = (inspect.getdoc(obj) or "").split("\n")[0]
                if len(sig) > 90:
                    sig = sig[:87] + "...)"
                print("  k.%s%s\n      %s" % (n, sig, first))
            elif isinstance(obj, dict):
                print("  k.%s: %s" % (n, ", ".join(sorted(map(str, obj)))))
    print("\nDetails for one function:  f3d api <name>")


# ---------------------------------------------------------------- doctor
def cmd_doctor(a):
    import platform
    import importlib.util as iu
    import bpy
    import numpy
    import cv2
    print("forge3d runtime")
    print("  python   %s  (%s)" % (platform.python_version(), sys.executable))
    print("  blender  %s (bpy module, headless)   numpy %s   opencv %s" % (bpy.app.version_string, numpy.__version__,
                                                                         cv2.__version__))
    engines = []
    for e in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES", "BLENDER_WORKBENCH"):
        try:
            bpy.context.scene.render.engine = e
            engines.append(e)
        except TypeError:
            pass
    print("  render engines: %s" % ", ".join(engines))
    for mod, extra, what in (("gradio_client", "gen3d", "hosted image-to-3D (f3d gen3d)"),
                             ("rembg", "rembg", "neural background removal for photos (f3d ref --bg rembg)")):
        ok = iu.find_spec(mod) is not None
        print("  optional %-6s %s  %s%s" % (extra, "installed" if ok else "missing  ", what,
                                           "" if ok else "  -> python3 f3d.py setup --extra " + extra))
    from . import vision
    what = {"upscale4x": "super-resolution of small reference pictures", "eye": "finds drawn eyes (one pair per face)",
            "face": "finds drawn faces"}
    for name, state in vision.status().items():
        print("  vision   %-9s %s  (%s)" % (name, state, what[name]))
    from .bb import app
    exe = app.find_blockbench()
    print("  blockbench app: %s" % (exe or "not found (only needed for `f3d bb check` / native exports; .bbmodel "
                                           "files are written without it)"))


def main(argv=None):
    p = argparse.ArgumentParser(prog="f3d", description="forge3d: reference-driven 3D modelling toolkit")
    sub = p.add_subparsers(dest="cmd", metavar="command")

    s = sub.add_parser("ref", help="analyse reference image(s) into a measured reference pack")
    s.add_argument("images", nargs="+", help="a turnaround sheet, or one image per view")
    s.add_argument("-o", "--out", default="refs", help="output folder (default: refs)")
    s.add_argument("--views", help="names of the views in reading order, e.g. front,right,back (right = subject "
                                   "faces LEFT in the picture, left = faces RIGHT; skip = ignore that one)")
    s.add_argument("--height", type=float, default=1.0, help="real height of the subject in metres (default 1.0)")
    s.add_argument("--bg", default="auto", help="auto | alpha | '#rrggbb' | rembg")
    s.add_argument("--tol", type=float, default=10.0, help="background colour tolerance, 0-255 (default 10)")
    s.add_argument("--holes", default="auto", help="auto | keep | min area fraction: enclosed background-coloured "
                                                   "areas become see-through gaps")
    s.add_argument("--scale", default="auto", choices=("auto", "same", "height"),
                   help="same = views share one drawing scale, height = each view is scaled to the same height")
    s.add_argument("--regions", type=int, default=36, help="numbered colour regions per view (0 = off)")
    s.add_argument("--sharpen", default="auto", choices=("auto", "off"),
                   help="auto = small drawings are redrawn larger by a super-resolution model before measuring")
    s.add_argument("--table", action="store_true", help="print the region tables")
    s.set_defaults(fn=cmd_ref)

    s = sub.add_parser("layers", help="take a picture apart: place layer pictures (bare body, hair, headwear ...) in "
                                      "the frame of the pack they were taken from, and check they add up to it")
    s.add_argument("ref", help="the reference pack of the whole picture (made by `f3d ref`)")
    s.add_argument("layers", nargs="+", help="name=picture.png for every layer (name alone re-checks an existing "
                                             "layer pack), e.g. body=layers/body.png hair=layers/hair.png")
    s.add_argument("--bg", default="auto", help="auto | alpha | '#rrggbb' | rembg (background of the layer pictures)")
    s.add_argument("--tol", type=float, default=10.0, help="background colour tolerance, 0-255 (default 10)")
    s.add_argument("--holes", default="auto", help="auto | keep | min area fraction (as for `f3d ref`)")
    s.add_argument("--no-align", action="store_true", help="trust the layer pictures to be pixel-aligned with the "
                                                           "sheet; skip the search for their place")
    s.set_defaults(fn=cmd_layers)

    def common(s, ref=True):
        if ref:
            s.add_argument("--ref", help="reference pack folder made by `f3d ref`")
        s.add_argument("--views", help="comma list: front,right,back,left,top,iso,... prefix clay: / parts: / "
                                       "silhouette: for other render modes, azNNelNN for any angle")
        s.add_argument("--tile", type=int, default=448, help="tile size in px")
        s.add_argument("--cols", type=int, default=4)

    s = sub.add_parser("build", help="run a build script, lint it, render it, compare it with the reference")
    s.add_argument("script")
    s.add_argument("-o", "--out", help="output folder (default: out/ next to the script)")
    s.add_argument("--no-render", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("blockout", help="parts route, step 1: run a blockout script (rough placeholder shapes given "
                                        "to slots with k.slot) and write a work order per part")
    s.add_argument("script")
    s.add_argument("--ref", help="reference pack folder made by `f3d ref`")
    s.add_argument("-o", "--out", help="parts folder (default: parts/ next to the script)")
    s.set_defaults(fn=cmd_blockout)

    s = sub.add_parser("part", help="parts route, step 2: build one part in its slot and check it there (no name: "
                                    "status of all parts)")
    s.add_argument("name", nargs="?", help="slot name (or the path of its part.py)")
    s.add_argument("--parts", help="parts folder (default: ./parts)")
    s.add_argument("--no-render", action="store_true")
    s.set_defaults(fn=cmd_part)

    s = sub.add_parser("review", help="parts route: pictures for an independent reviewer of one part (no name: of "
                                      "the assembled model)")
    s.add_argument("name", nargs="?")
    s.add_argument("--parts")
    s.add_argument("--tile", type=int, help="pixels per tile; use 960 to inspect fine details such as overlapping locks")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("portrait", help="compare a staged .blend face with the reference in matched close-ups")
    s.add_argument("source")
    s.add_argument("--parts")
    s.add_argument("--slot", help="face slot in the blockout plan")
    s.add_argument("--form", action="store_true", help="bare face in clay and its outline over the reference")
    s.add_argument("--span", nargs=3, action="append", type=float, metavar=("Z", "LEFT_X", "RIGHT_X"),
                   help="measure at two visible jaw edges from the front grid; repeat for additional heights")
    s.add_argument("--tile", type=int, default=448)
    s.add_argument("-o", "--out", help="PNG path (default: beside the .blend)")
    s.set_defaults(fn=cmd_portrait)

    s = sub.add_parser("next", help="parts route: what to do next (who builds, who reviews, who repairs), until done")
    s.add_argument("--parts")
    s.set_defaults(fn=cmd_next)

    s = sub.add_parser("view", help="contact sheet of a build script / .blend / mesh file / .bbmodel")
    s.add_argument("source")
    s.add_argument("-o", "--out")
    s.add_argument("--lint", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_view)

    s = sub.add_parser("compare", help="compare a model with a reference pack, view by view")
    s.add_argument("source")
    s.add_argument("-o", "--out")
    common(s)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("gen3d", help="image-to-3D base mesh from a reference pack (hosted model)")
    s.add_argument("source", nargs="*", default=["refs"], help="reference pack folder(s) (or a single image); layer "
                                                             "packs can be given together")
    s.add_argument("-o", "--out", help="output .glb (default: gen/base.glb beside the pack, gen/<layer>.glb for a "
                                       "layer pack)")
    s.add_argument("--backend", default="auto", help="auto | hunyuan-mv | hunyuan")
    s.add_argument("--quality", default="standard", choices=("draft", "standard", "high"))
    s.add_argument("--seed", type=int, default=1234, help="change it to get a different attempt")
    s.add_argument("--force", action="store_true", help="generate again even if the result is cached")
    s.add_argument("--list", action="store_true", help="list backends")
    s.set_defaults(fn=cmd_gen3d)

    s = sub.add_parser("bb", help="Blockbench: turn a model into a .bbmodel (mesh or voxels), check one in the real app")
    s.add_argument("action", choices=("mesh", "voxelize", "check"))
    s.add_argument("source", help="mesh / voxelize: build script, .blend or mesh file;  check: a .bbmodel")
    s.add_argument("--faces", type=int, default=60000, help="mesh: total face budget (default 60000)")
    s.add_argument("-o", "--out")
    s.add_argument("--ref", help="reference pack: colours are taken from its pictures")
    s.add_argument("--height", type=int, default=None, help="Blockbench pixels along the subject's height "
                                                             "(default: 40 for voxelize, 64 for mesh)")
    s.add_argument("--colors", type=int, default=48, help="palette size, 0 = unlimited")
    s.add_argument("--format", default="free", help="free | bedrock | java_block | modded_entity | ...")
    s.add_argument("--shots", help="check: where to save the Blockbench viewport picture")
    s.add_argument("--export", action="append", help="check: also export with Blockbench's own exporter, "
                                                     "codec:path (bedrock, java_block, gltf, obj, fbx, ...)")
    s.set_defaults(fn=cmd_bb)

    s = sub.add_parser("api", help="list kit functions, or show one in full")
    s.add_argument("name", nargs="?")
    s.set_defaults(fn=cmd_api)

    s = sub.add_parser("doctor", help="check the runtime and optional components")
    s.set_defaults(fn=cmd_doctor)

    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
