"""Checks that look for drawn features on the finished model, from directions no reference shows.

The compare score sees the model only from the reference cameras, where projected paint matches by
construction; a head with two pairs of eyes scores as well as a head with one. This looks at the model from
all round and counts.
"""
import os
import tempfile

import cv2
import numpy as np

AZIMUTHS = (0, 40, 90, 140, 180, 220, 270, 320)


def eye_check(ref_dir, out_png=None, tile=640):
    """Count the eyes that can be seen on the model from eight directions and hold them against the number the
    front reference shows. -> dict(want, seen={azimuth: count}, errors=[...], image) or None when it cannot be
    done (no eyes in the reference, detector not available)."""
    from . import compare, sheet, vision
    from .kit import core, staging
    doc = compare.load_pack(ref_dir)
    front = doc["views"].get("front") or {}
    want = len(front.get("eyes") or [])
    if want == 0 or not vision.available("eye"):
        return None
    side = max(len((doc["views"].get(n) or {}).get("eyes") or []) for n in ("right", "left")) if any(
        n in doc["views"] for n in ("right", "left")) else None
    back = len((doc["views"].get("back") or {}).get("eyes") or []) if "back" in doc["views"] else None
    eyes = front["eyes"]
    eh = max(e[3] - e[1] for e in eyes)
    band = (min(e[1] for e in eyes) - 0.8 * eh, max(e[3] for e in eyes) + 0.8 * eh)   # heights where a face can be
    items = core.scene_objects()
    if not items:
        return None
    pts = staging._sample_points(items, 20000)
    scale = float((pts.max(0) - pts.min(0)).max()) * 1.12
    tmp = tempfile.mkdtemp(prefix="f3d_eyes_")
    seen, tiles = {}, []
    with staging.evaluation_setup("color"):
        for az in AZIMUTHS:
            cam = staging.temp_camera(float(az), 0.0, pts, ortho=True, margin=1.12, ortho_scale=scale)
            path = os.path.join(tmp, "eyes_%03d.png" % az)
            staging.render_tile(cam, path, (tile, tile))
            cam_z = cam.matrix_world.translation.z
            staging.remove_camera(cam)
            img, _alpha = sheet.composite(path)
            ppm = tile / scale
            found = []
            for x0, y0, x1, y1, _s in vision.detect(img, "eye", conf=0.5) or []:
                z = (tile / 2.0 + cam_z * ppm - (y0 + y1) / 2.0) / ppm          # height of the box on the model
                size = (y1 - y0) / ppm
                if band[0] <= z <= band[1] and 0.35 * eh <= size <= 2.5 * eh:    # at face height, about eye-sized
                    found.append((x0, y0, x1, y1))
            seen[az] = len(found)
            for x0, y0, x1, y1 in found:
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (40, 40, 235), 2)
            sheet.put(img, "%d deg: %d eye%s" % (az, len(found), "" if len(found) == 1 else "s"), (8, 22), 0.6)
            tiles.append(img)
    errors = []
    many = [az for az in AZIMUTHS if seen[az] > want]
    if many:
        errors.append("%d eyes are visible from %s; the reference face has %d. A second pair was painted on: the "
                      "eye of a side picture is the same eye the front picture shows, and a face takes its eyes "
                      "from the front picture only. Look at each part's check.png to find the part that carries "
                      "the copy (hair or a head painted from a side view) and send it back" % (
                          max(seen[az] for az in many), ", ".join("%d deg" % az for az in many), want))
    if seen[0] < want:
        errors.append("only %d of the %d eyes can be seen from the front: something covers the face more than in "
                      "the reference, or the face was not painted" % (seen[0], want))
    if back == 0 and seen[180] > 0:
        errors.append("%d eye(s) can be seen from behind; the back reference shows none" % seen[180])
    image = None
    if out_png:
        row = np.hstack(tiles)
        if row.shape[1] > 2600:
            row = cv2.resize(row, None, fx=2600.0 / row.shape[1], fy=2600.0 / row.shape[1], interpolation=cv2.INTER_AREA)
        cv2.imwrite(out_png, row)
        image = os.path.abspath(out_png)
    return dict(want=want, side=side, seen=seen, errors=errors, image=image)


def format_eye_check(rep):
    head = "EYES: the front reference shows %d; on the model, going round: %s" % (
        rep["want"], ", ".join("%d deg: %d" % (az, c) for az, c in rep["seen"].items()))
    if not rep["errors"]:
        return head + "  -> ok"
    return "\n".join([head] + ["  ERROR: " + e for e in rep["errors"]])
