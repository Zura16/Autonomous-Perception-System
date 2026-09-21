"""Export a replay the browser can play, with its telemetry.

    python tools/export_replay.py --drive 2011_09_26_drive_0059 --frames 60 200

Writes `site/data/<drive>/` : numbered JPEG frames, `telemetry.json`, and
`meta.json`. `site/app.js` plays them.

**This exports measured output. It is not a live service, and there is no way to
make it one from a static page.** APS is offline and open-loop by charter, and
the previously published site advertised live play/pause/stop controls wired to
a backend that did not exist. A replay of frames this pipeline actually produced
is the honest version of "run it in the browser": every number the viewer sees
was computed by `app/replay.py`, not re-derived by JavaScript.

That last point is the design constraint. The composited frame is exported from
`build_frame` itself rather than redrawn in the browser from boxes, so the HUD
the viewer sees is the HUD the project ships -- including its abstention and
envelope rules. Re-implementing those rules in JavaScript would create a second
source of display truth, free to drift from the first, which is the defect class
this repository keeps finding. The JSON carries only what a side panel needs and
is explicitly *derived* from the same objects the frame was drawn from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.replay import build_frame  # noqa: E402
from aps.detect import Detector  # noqa: E402
from aps.fcw import FcwConfig, FcwDecider, corridor_range_m, lateral_offset_m  # noqa: E402
from aps.geometry import (  # noqa: E402
    CameraModel,
    CredibleEnvelope,
    range_contact_point,
    range_size_prior,
)
from aps.kitti import load_split  # noqa: E402
from aps.kitti.road import RoadGeometry  # noqa: E402
from aps.lanes import BevGrid, LaneConfig, estimate_lane  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402
from aps.viz import CLASS_COLOURS, DEFAULT_COLOUR  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _clean(value: float) -> float | None:
    """JSON has no NaN. An abstention must survive the trip as `null`.

    `json.dumps` emits bare `NaN`, which is invalid JSON and which
    `JSON.parse` rejects outright. Writing 0.0 instead would turn "the
    estimator declined" into "the estimator said zero" -- the exact
    fabrication hard rule 14 forbids, laundered through a serialiser.
    """
    return float(value) if np.isfinite(value) else None


def export(args: argparse.Namespace) -> int:
    drive = next((d for d in load_split(["dev", "val"]) if d.id == args.drive), None)
    if drive is None:
        raise SystemExit(f"unknown drive {args.drive}")

    camera = CameraModel.from_config()
    envelope = CredibleEnvelope.from_config()
    lane_cfg = LaneConfig.from_config()
    fcw_cfg = FcwConfig.from_config()
    detector = Detector(args.weights, device=args.device)
    tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
    motion = ScaleRateEstimator(MotionConfig.from_config(), deadband="fixed")
    decider = FcwDecider(fcw_cfg)
    lane_geom = RoadGeometry.nominal(drive.calib.P_rect[2], lane_cfg.camera_height_m)
    grid = BevGrid.from_config(lane_cfg)
    fx, _, cx, _ = drive.calib.intrinsics(2)

    out_dir = REPO / "site" / "data" / args.drive
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()

    lo, hi = args.frames
    # The tracker and the scale-rate filter carry history, so a clip that starts
    # cold reports `--` for every TTC until the filter matures. Replay the
    # preceding frames to warm them, and export only from `lo`.
    warm = max(0, lo - args.warmup)
    frames = list(range(warm, min(hi, len(drive))))
    detector.warmup(drive.frame(frames[0]).image())

    telemetry = []
    last_t: dict[int, float] = {}
    written = 0

    for fi in frames:
        image = drive.frame(fi).image()
        dets = detector.detect(image).above(args.conf)
        tracks = tracker.update(dets.boxes, dets.scores)
        now = float(drive.timestamps_s[fi])
        group_of = {tuple(np.round(b, 3)): g for b, g in zip(dets.boxes, dets.groups, strict=True)}

        objects, any_warn, any_aeb = [], False, False
        for tr in tracks:
            box = tr.box
            group = group_of.get(tuple(np.round(box, 3)), "vehicle")
            contact = range_contact_point(box, drive.calib, camera)
            prior = range_size_prior(box, drive.calib, camera, group)
            dt = now - last_t.get(tr.track_id, now)
            last_t[tr.track_id] = now
            state = motion.update(tr.track_id, float(box[3] - box[1]), dt, contact)
            lateral = lateral_offset_m(
                0.5 * (box[0] + box[2]), fx, cx, corridor_range_m(contact, prior, fcw_cfg)
            )
            out = decider.update(tr.track_id, state.is_closing, state.ttc_s, lateral)
            any_warn |= out.warning
            any_aeb |= out.aeb_request
            objects.append(
                {
                    "track_id": int(tr.track_id),
                    "box": [float(v) for v in box],
                    "range_m": contact,
                    "ttc_s": state.ttc_s if state.is_closing else float("nan"),
                    "lateral_m": lateral,
                    "warning": bool(out.warning),
                    "group": group,
                }
            )

        lane, _ = estimate_lane(image, lane_geom, lane_cfg, grid) if args.lanes else (None, None)

        if fi < lo:
            continue  # warm-up frame: state advanced, nothing written

        # Drawn by the shipped compositor, so the exported frame cannot disagree
        # with what `app/replay.py` would show.
        # Same colour lookup app/replay.py uses. Passing a flat grey here would
        # silently drop the vehicle/VRU distinction from the exported frame while
        # the shipped HUD still showed it.
        canvas = build_frame(
            image,
            [
                {
                    **o,
                    "colour": CLASS_COLOURS.get(
                        "Car" if o["group"] == "vehicle" else "Pedestrian", DEFAULT_COLOUR
                    ),
                }
                for o in objects
            ],
            lane,
            fcw_cfg,
            lane_geom,
            any_warn,
            any_aeb,
        )
        if args.width and canvas.shape[1] > args.width:
            scale = args.width / canvas.shape[1]
            canvas = cv2.resize(
                canvas, (args.width, int(round(canvas.shape[0] * scale))), cv2.INTER_AREA
            )
        cv2.imwrite(
            str(out_dir / f"{written:04d}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
        )

        telemetry.append(
            {
                "frame": int(fi),
                "t_s": round(now, 3),
                "warning": bool(any_warn),
                "aeb_request": bool(any_aeb),
                "objects": [
                    {
                        "id": o["track_id"],
                        "group": o["group"],
                        "range_m": _clean(o["range_m"]),
                        "ttc_s": _clean(o["ttc_s"]),
                        "lateral_m": _clean(o["lateral_m"]),
                        "in_envelope": envelope.contains(o["range_m"]),
                        "warning": o["warning"],
                    }
                    for o in objects
                ],
            }
        )
        written += 1
        if written % 20 == 0:
            print(f"  {written} frames", flush=True)

    size = canvas.shape
    meta = {
        "drive": args.drive,
        "split": drive.split,
        "scene": drive.scene,
        "frames": written,
        "source_range": [lo, lo + written],
        "fps": round(1.0 / float(np.mean(np.diff(drive.timestamps_s))), 3),
        "width": int(size[1]),
        "height": int(size[0]),
        "envelope_m": [envelope.min_range_m, envelope.max_range_m],
        "envelope_max_mape_pct": envelope.max_mape_pct,
        "ttc_gated_by_envelope": envelope.gates_ttc,
        "note": (
            "Offline replay of measured pipeline output. Frames are composited by "
            "app/replay.py; this is not a live service and nothing is actuated."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out_dir / "telemetry.json").write_text(json.dumps(telemetry, separators=(",", ":")) + "\n")

    total_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
    print(f"\nwrote {written} frames + telemetry to {out_dir}  ({total_mb:.1f} MB)")
    if total_mb > 40:
        print("  WARNING: large for a static site; reduce --frames or --width")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default="2011_09_26_drive_0059")
    ap.add_argument("--frames", type=int, nargs=2, default=(60, 200), metavar=("LO", "HI"))
    ap.add_argument("--warmup", type=int, default=20, help="frames replayed to warm the filters")
    ap.add_argument("--width", type=int, default=1100, help="output width; 0 keeps native")
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--no-lanes", dest="lanes", action="store_false")
    return export(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
