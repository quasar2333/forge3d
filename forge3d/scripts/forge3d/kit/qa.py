"""Geometry lint: the checks an artist does by orbiting the model, done numerically."""
import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import core


def allow_floating(objs):
    """Mark object(s) as intentionally detached (hovering parts, particles) so lint does not flag them."""
    for o in core.expand(objs, geometry_only=False):
        o["f3d_float"] = True
    return objs


def _inside(bvh, p, far):
    hits = 0
    o = Vector(p)
    d = Vector((0.0123, 0.0071, 1.0)).normalized()
    for _ in range(64):
        loc, _n, _i, dist = bvh.ray_cast(o, d, far)
        if loc is None:
            break
        hits += 1
        o = loc + d * 1e-5
    return hits % 2 == 1


def lint(objs=None, verbose=True, tolerance=None):
    """Check the model for the defects that most often ruin generated 3D:

    floating parts, parts hidden inside others, duplicates stacked on the same spot, inverted or broken meshes,
    model sunk below / hovering above the ground. Prints a report and returns
    {'ok', 'errors', 'warnings', 'issues': [(level, code, message)], 'stats'}.
    tolerance: gap (m) still counted as touching; default 0.3 % of the model's size.
    """
    items = [o for o in (core.scene_objects() if objs is None else core.expand(objs)) if o.type == "MESH"]
    issues = []
    dg = core.depsgraph()
    info = []
    total_tris = 0
    for o in items:
        v, t = core.world_tris(o, dg)
        if len(v) == 0 or len(t) == 0:
            issues.append(("ERROR", "empty", "'%s' has no faces (nothing will render)." % o.name))
            continue
        if not np.isfinite(v).all():
            issues.append(("ERROR", "nan", "'%s' has invalid (NaN/inf) coordinates." % o.name))
            continue
        total_tris += len(t)
        info.append(dict(obj=o, v=v, t=t, lo=v.min(0), hi=v.max(0)))
    if not info:
        issues.append(("ERROR", "no_geometry", "The scene contains no mesh geometry."))
        return _report(issues, {"objects": 0, "tris": 0}, verbose)
    lo = np.min([d["lo"] for d in info], axis=0)
    hi = np.max([d["hi"] for d in info], axis=0)
    size = hi - lo
    diag = float(np.linalg.norm(size))
    eps = tolerance if tolerance is not None else max(1e-5, 0.003 * diag)

    # ---- per-mesh topology
    for d in info:
        o = d["obj"]
        bm = bmesh.new()
        bm.from_object(o, dg)
        wire = sum(1 for e in bm.edges if len(e.link_faces) == 0)
        boundary = sum(1 for e in bm.edges if len(e.link_faces) == 1)
        multi = sum(1 for e in bm.edges if len(e.link_faces) > 2)
        degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-14)
        closed = boundary == 0 and multi == 0 and wire == 0 and len(bm.faces) > 3
        d["closed"] = closed
        if closed:
            vol = bm.calc_volume(signed=True)
            if o.matrix_world.determinant() < 0:
                vol = -vol
            if vol < 0:
                issues.append(("ERROR", "inverted", "'%s' is inside-out (normals point inward); it will shade dark "
                               "and break booleans. Rebuild it or flip its normals." % o.name))
        if multi and not o.get("f3d_adopted"):
            issues.append(("WARN", "non_manifold", "'%s' has %d edges shared by more than two faces "
                           "(self-intersecting construction)." % (o.name, multi)))
        if degenerate > max(2, 0.01 * len(bm.faces)):
            issues.append(("WARN", "degenerate", "'%s' has %d zero-area faces." % (o.name, degenerate)))
        if wire:
            issues.append(("INFO", "wire", "'%s' has %d loose edges without faces." % (o.name, wire)))
        d["open"] = boundary
        bm.free()
        s = o.matrix_world.to_scale()
        if min(abs(s.x), abs(s.y), abs(s.z)) < 1e-9:
            issues.append(("ERROR", "zero_scale", "'%s' has a zero scale on some axis." % o.name))
        ext = d["hi"] - d["lo"]
        if float(np.linalg.norm(ext)) < 0.004 * diag and len(info) > 1:
            issues.append(("INFO", "tiny", "'%s' is tiny relative to the model (%.4f m); it may not be visible."
                           % (o.name, float(np.linalg.norm(ext)))))
        if not o.data.materials or all(m is None for m in o.data.materials):
            d["no_mat"] = True
    no_mat = [d["obj"].name for d in info if d.get("no_mat")]
    if no_mat:
        issues.append(("WARN", "no_material", "%d object(s) have no material and will render default grey: %s"
                       % (len(no_mat), ", ".join(no_mat[:6]) + (" ..." if len(no_mat) > 6 else ""))))

    odd = sorted({m.name for d in info for m in d["obj"].data.materials if m is not None and not m.name.isascii()})
    if odd:
        issues.append(("WARN", "material_name", "material name(s) with non-ASCII characters: %s. Blender's EEVEE on "
                       "macOS can crash on those when the .blend is opened and rendered; rename them to plain ASCII "
                       "(object and collection names are free)." % ", ".join(repr(n) for n in odd[:4])))

    # ---- duplicates on the same spot
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            a, b = info[i], info[j]
            if len(a["v"]) == len(b["v"]) and np.allclose(a["lo"], b["lo"], atol=eps * 0.2) \
                    and np.allclose(a["hi"], b["hi"], atol=eps * 0.2):
                issues.append(("WARN", "duplicate", "'%s' and '%s' occupy exactly the same space (duplicate object? "
                               "it will z-fight)." % (a["obj"].name, b["obj"].name)))

    # ---- contact graph
    for d in info:
        d["bvh"] = BVHTree.FromPolygons(d["v"].tolist(), d["t"].tolist(), all_triangles=True)
    n = len(info)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def sample(v, k=300):
        return v if len(v) <= k else v[np.linspace(0, len(v) - 1, k).astype(int)]

    hidden = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = info[i], info[j]
            if np.any(a["lo"] > b["hi"] + eps) or np.any(b["lo"] > a["hi"] + eps):
                continue
            touching = bool(a["bvh"].overlap(b["bvh"]))
            if not touching:
                for src, dst in ((a, b), (b, a)):
                    for p in sample(src["v"]):
                        hit = dst["bvh"].find_nearest(Vector(p.tolist()), eps)
                        if hit[0] is not None:
                            touching = True
                            break
                    if touching:
                        break
            if not touching:
                for inner, outer in ((a, b), (b, a)):
                    if outer["closed"] and np.all(inner["lo"] >= outer["lo"] - eps) and np.all(inner["hi"] <= outer["hi"] + eps):
                        pts = sample(inner["v"], 5)
                        if all(_inside(outer["bvh"], p.tolist(), diag * 4) for p in pts):
                            touching = True
                            hidden.append((inner["obj"].name, outer["obj"].name))
                            break
            if touching:
                parent[find(i)] = find(j)
    for inner, outer in hidden:
        issues.append(("WARN", "hidden", "'%s' is completely inside '%s' and cannot be seen. Move it out to the "
                       "surface (k.stick / k.attach) or delete it." % (inner, outer)))
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    if len(comps) > 1:
        def weight(idx):
            return sum(float(np.prod(np.maximum(info[i]["hi"] - info[i]["lo"], 1e-9))) for i in idx)
        order = sorted(comps.values(), key=weight, reverse=True)
        main = order[0]
        for comp in order[1:]:
            if all(info[i]["obj"].get("f3d_float") for i in comp):
                continue
            best = None
            for i in comp:
                for p in sample(info[i]["v"], 200):
                    pv = Vector(p.tolist())
                    for j in main:
                        loc, _nrm, _idx, dist = info[j]["bvh"].find_nearest(pv)
                        if loc is not None and (best is None or dist < best[0]):
                            best = (dist, info[i]["obj"].name, info[j]["obj"].name, loc - pv)
            names = ", ".join("'%s'" % info[i]["obj"].name for i in comp[:4]) + (" ..." if len(comp) > 4 else "")
            if best is not None:
                dvec = best[3]
                ax = int(np.argmax(np.abs(np.array(dvec))))
                dirn = ("+" if dvec[ax] > 0 else "-") + "XYZ"[ax]
                issues.append(("ERROR", "floating", "%s %s not touching the rest of the model: %.4f m gap between "
                               "'%s' and '%s' (move it %s by that much, or use k.attach / k.stick with embed)."
                               % (names, "is" if len(comp) == 1 else "are", best[0], best[1], best[2], dirn)))
            else:
                issues.append(("ERROR", "floating", "%s not touching the rest of the model." % names))

    # ---- ground
    if abs(lo[2]) > max(eps, 0.01 * size[2]):
        if lo[2] < 0:
            issues.append(("WARN", "below_ground", "The model's lowest point is z = %.4f, below the ground plane "
                           "z = 0. Call k.ground() at the end." % lo[2]))
        else:
            issues.append(("INFO", "above_ground", "The model's lowest point is z = %.4f (hovering above z = 0). "
                           "Call k.ground() unless that is intended." % lo[2]))
    open_objs = [d["obj"].name for d in info if d.get("open")]
    stats = {"objects": len(info), "tris": int(total_tris),
             "size": [round(float(x), 4) for x in size], "min": [round(float(x), 4) for x in lo],
             "max": [round(float(x), 4) for x in hi], "open_meshes": open_objs[:20],
             "materials": len({m.name for d in info for m in d["obj"].data.materials if m})}
    return _report(issues, stats, verbose)


def _report(issues, stats, verbose):
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    issues = sorted(issues, key=lambda x: order[x[0]])
    errors = sum(1 for i in issues if i[0] == "ERROR")
    warnings = sum(1 for i in issues if i[0] == "WARN")
    result = {"ok": errors == 0, "errors": errors, "warnings": warnings, "issues": issues, "stats": stats}
    if verbose:
        head = "LINT: clean" if not errors and not warnings else "LINT: %d error(s), %d warning(s)" % (errors, warnings)
        print(head)
        for level, _code, msg in issues:
            print("  %-5s %s" % (level, msg))
        if stats.get("objects"):
            sx, sy, sz = stats["size"]
            print("  model: %d objects, %s triangles, %d materials, size %.3f x %.3f x %.3f m (X width, Y depth, Z height)"
                  % (stats["objects"], format(stats["tris"], ","), stats["materials"], sx, sy, sz))
            if stats["open_meshes"]:
                print("  open (non-watertight) meshes: %s" % ", ".join(stats["open_meshes"][:8]))
    return result
