"""Phase 8: HUD and top-down replay. Presents results; produces none.

    python app/replay.py --drive 2011_09_26_drive_0059 --frames 0 200 --out artifacts/replay.mp4
    python app/replay.py --drive 2011_09_26_drive_0059 --still 96 --out docs/figures/hud.jpg

Deliberately the last stage built. Every number it draws was measured in an
earlier phase and is reproducible from `docs/benchmarks.md`; this file adds no
estimate of its own. It exists because a stack that cannot be watched is hard to
debug, not because a video is evidence.

Three rules govern what it is allowed to draw (CLAUDE.md hard rules 13, 14):

  **Abstentions stay blank.** Where an estimator declined -- no ground-plane
  contact, a box clipped at the image edge, a track too young for a TTC -- the
  HUD shows "--", never a filled-in guess. A display that invents a number is
  worse than one that admits a gap.

  **Every figure carries its envelope.** Range is annotated against the measured
  credible band, read from `configs/camera.yaml` (20-50 m at <=13% MAPE, Rows
  3.1/9.7); values outside it are drawn dimmed and their range withheld, because
  the project cannot vouch for them. The band moved OUTWARD after the held-out
  split refuted the earlier val-only 10-30 m figure -- so this HUD now withholds
  range in the near field, where a collision is most imminent. That is the
  measurement talking, and it is not styled away.

  TTC is exempt from that gate and says so in the config: it is measured
  separately and is unbiased at GT TTC < 3 s (D-022). Gating it on a range
  criterion would hide the most trustworthy number here.

  **The decision layer is labelled as what it is.** "AEB REQUEST (logged, not
  actuated)". Monocular, offline, open-loop -- stated on every frame, so a
  screenshot cannot be mistaken for something it is not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from aps.lanes import BevGrid, LaneConfig, LaneEstimate, estimate_lane  # noqa: E402
from aps.motion import MotionConfig, ScaleRateEstimator  # noqa: E402
from aps.tracking import make_tracker  # noqa: E402
from aps.viz import CLASS_COLOURS, DEFAULT_COLOUR, draw_box_2d  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# The band the project can vouch for, read from configs/camera.yaml because it
# is a measured result (Rows 3.1, 9.7) and moved outward when the held-out split
# refuted the val-only figure. A HUD that hardcodes it goes on vouching for a
# band the evaluation has withdrawn.
ENVELOPE = CredibleEnvelope.from_config()
TOPDOWN_W, TOPDOWN_H = 420, 560
TOPDOWN_MAX_RANGE_M = 50.0
TOPDOWN_HALF_WIDTH_M = 12.0

GREY = (150, 150, 150)
WHITE = (240, 240, 240)
AMBER = (40, 190, 250)
RED = (60, 60, 240)


def _fmt(value: float, unit: str) -> str:
    """A number, or '--' when the estimator abstained. Never a fabricated value."""
    return f"{value:.1f}{unit}" if np.isfinite(value) else "--"


def draw_topdown(objects: list[dict], cfg: FcwConfig) -> np.ndarray:
    """Bird's-eye panel: ego at the bottom, range rings, corridor, tracked objects."""
    panel = np.full((TOPDOWN_H, TOPDOWN_W, 3), 24, dtype=np.uint8)
    cx, cy = TOPDOWN_W // 2, TOPDOWN_H - 40
    px_per_m_z = (TOPDOWN_H - 70) / TOPDOWN_MAX_RANGE_M
    px_per_m_x = (TOPDOWN_W / 2 - 10) / TOPDOWN_HALF_WIDTH_M

    def to_px(x_m: float, z_m: float) -> tuple[int, int]:
        return int(cx + x_m * px_per_m_x), int(cy - z_m * px_per_m_z)

    # The credible band, drawn as the only region the project vouches for.
    top = to_px(0, ENVELOPE.max_range_m)[1]
    bottom = to_px(0, ENVELOPE.min_range_m)[1]
    band = panel[top:bottom].copy()
    cv2.addWeighted(band, 0.75, np.full_like(band, (40, 70, 40)), 0.25, 0, band)
    panel[top:bottom] = band

    for r in (10, 20, 30, 40, 50):
        y = to_px(0, r)[1]
        cv2.line(panel, (10, y), (TOPDOWN_W - 10, y), (60, 60, 60), 1)
        cv2.putText(
            panel, f"{r} m", (12, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREY, 1, cv2.LINE_AA
        )

    for sign in (-1, 1):
        x = to_px(sign * cfg.corridor_half_width_m, 0)[0]
        cv2.line(panel, (x, cy), (x, top), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(panel, f"ego corridor +/-{cfg.corridor_half_width_m:g} m (straight)",
                (10, TOPDOWN_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, GREY, 1, cv2.LINE_AA)  # fmt: skip
    cv2.drawMarker(panel, (cx, cy), WHITE, cv2.MARKER_TRIANGLE_UP, 14, 2)

    for o in objects:
        if not np.isfinite(o["range_m"]):
            continue
        inside = ENVELOPE.contains(o["range_m"])
        colour = RED if o["warning"] else (o["colour"] if inside else GREY)
        p = to_px(o["lateral_m"], min(o["range_m"], TOPDOWN_MAX_RANGE_M))
        cv2.circle(panel, p, 6 if inside else 4, colour, -1 if inside else 1, cv2.LINE_AA)
        cv2.putText(panel, f"{o['track_id']}", (p[0] + 8, p[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, colour, 1, cv2.LINE_AA)  # fmt: skip

    cv2.putText(
        panel,
        "TOP-DOWN (estimated)",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, "green band = credible range envelope", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 200, 120), 1, cv2.LINE_AA)  # fmt: skip
    return panel


def _scope_line() -> str:
    """The caveat text, generated from the envelope rather than retyped.

    The old hardcoded "range credible 10-30 m (<=15% MAPE)" stayed on screen
    after the held-out split withdrew that band. A caption is a claim, and a
    claim that cannot follow its measurement will eventually contradict it.
    """
    return (
        f"range credible {ENVELOPE.min_range_m:g}-{ENVELOPE.max_range_m:g} m "
        f"(<={ENVELOPE.max_mape_pct:g}% MAPE, val AND held-out test) | "
        "TTC MAE 0.41 s under 3 s | LiDAR = ground truth only, not an input"
    )


def footer(width: int, warning: bool, aeb: bool) -> np.ndarray:
    """The caveat strip. Present on every frame so a screenshot carries its scope."""
    strip = np.full((58, width, 3), 18, dtype=np.uint8)
    cv2.putText(strip, "MONOCULAR CAMERA ONLY - OFFLINE, OPEN-LOOP EVALUATION - NOTHING IS ACTUATED",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA)  # fmt: skip
    cv2.putText(strip, _scope_line(),
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)  # fmt: skip
    if warning:
        cv2.rectangle(strip, (width - 330, 8), (width - 12, 48), (0, 0, 120), -1)
        cv2.putText(strip, "AEB REQUEST (logged)" if aeb else "FCW WARNING",
                    (width - 320, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)  # fmt: skip
    return strip


def build_frame(
    image: np.ndarray,
    objects: list[dict],
    lane: LaneEstimate | None,
    cfg: FcwConfig,
    geom: RoadGeometry,
    warning: bool,
    aeb: bool,
) -> np.ndarray:
    canvas = image.copy()
    if lane is not None and lane.detected:
        for b in (lane.left, lane.right):
            z = np.linspace(max(b.z_min_m, 6.0), min(b.z_max_m, 40.0), 50)
            u, v = geom.road_to_pixel(b.x_at(z), z)
            pts = np.stack([u, v], 1)
            pts = pts[np.isfinite(pts).all(1)].astype(np.int32)
            cv2.polylines(canvas, [pts], False, (120, 220, 120), 2, cv2.LINE_AA)

    for o in objects:
        inside = ENVELOPE.contains(o["range_m"])
        colour = RED if o["warning"] else (o["colour"] if inside else GREY)
        draw_box_2d(canvas, o["box"], "", colour, thickness=2 if inside else 1)
        # Outside the credible envelope the RANGE is withheld -- the project
        # cannot vouch for it, and printing it anyway is how a HUD launders an
        # unsupported figure into a screenshot.
        #
        # TTC is not withheld with it. The two are measured separately, and TTC
        # is unbiased where it matters (median +0.01 s at GT TTC < 3 s, D-022)
        # precisely because the scale-rate ratio cancels the box bias that drives
        # range error. Gating TTC on a range criterion suppressed this stack's
        # most trustworthy output on the strength of a different quantity's
        # failure -- and did so hardest in the near field, where a warning
        # matters most.
        parts = [f"#{o['track_id']}"]
        parts.append(_fmt(o["range_m"], " m") if inside else "-- m")
        if not ENVELOPE.gates_ttc or inside:
            parts.append(f"TTC {_fmt(o['ttc_s'], ' s')}")
        draw_box_2d(canvas, o["box"], "  ".join(parts), colour, thickness=0)

    panel = draw_topdown(objects, cfg)
    # Scale the camera view to the panel's height rather than padding it: dead
    # black space in a HUD is space not spent on the thing being inspected.
    scale = panel.shape[0] / canvas.shape[0]
    canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), panel.shape[0]))
    body = np.hstack([canvas, panel])
    return np.vstack([body, footer(body.shape[1], warning, aeb)])


def run(args: argparse.Namespace) -> int:
    # dev+val only: the test drives are deliberately not fetched until Phase 9,
    # and a viewer must not be the thing that opens them.
    drive = next((d for d in load_split(["dev", "val"]) if d.id == args.drive), None)
    if drive is None:
        print(f"unknown drive {args.drive}")
        return 1

    camera = CameraModel.from_config()
    lane_cfg = LaneConfig.from_config()
    fcw_cfg = FcwConfig.from_config()
    detector = Detector(args.weights, device=args.device)
    tracker = make_tracker("sort_det", iou_threshold=0.3, max_age=3)
    motion = ScaleRateEstimator(MotionConfig.from_config(), deadband="fixed")
    decider = FcwDecider(fcw_cfg)
    lane_geom = RoadGeometry.nominal(drive.calib.P_rect[2], lane_cfg.camera_height_m)
    grid = BevGrid.from_config(lane_cfg)
    fx, _, cx, _ = drive.calib.intrinsics(2)

    lo, hi = args.frames if args.frames else (0, len(drive))
    if args.still is not None:
        # A single frame leaves the scale-rate filter with one observation, so it
        # would correctly refuse every TTC and the still would misrepresent the
        # system. Replay the preceding frames to warm the tracker and filter.
        frames = list(range(max(0, args.still - args.warmup), args.still + 1))
    else:
        frames = list(range(lo, min(hi, len(drive))))
    writer = None
    last_t: dict[int, float] = {}

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
            lateral = lateral_offset_m(0.5 * (box[0] + box[2]), fx, cx,
                                       corridor_range_m(contact, prior, fcw_cfg))  # fmt: skip
            out = decider.update(tr.track_id, state.is_closing, state.ttc_s, lateral)
            any_warn |= out.warning
            any_aeb |= out.aeb_request
            objects.append(
                dict(
                    track_id=tr.track_id,
                    box=box,
                    range_m=contact,
                    ttc_s=state.ttc_s if state.is_closing else float("nan"),
                    lateral_m=lateral,
                    warning=out.warning,
                    colour=CLASS_COLOURS.get(
                        "Car" if group == "vehicle" else "Pedestrian", DEFAULT_COLOUR
                    ),
                )
            )

        lane, _ = estimate_lane(image, lane_geom, lane_cfg, grid) if args.lanes else (None, None)
        canvas = build_frame(image, objects, lane, fcw_cfg, lane_geom, any_warn, any_aeb)

        if args.still is not None:
            if fi != args.still:
                continue  # warm-up frame: state advanced, nothing written
            args.out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"wrote {args.out}  ({canvas.shape[1]}x{canvas.shape[0]})")
            return 0
        if writer is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     1.0 / drive.mean_dt_s, (canvas.shape[1], canvas.shape[0]))  # fmt: skip
        writer.write(canvas)

    if writer is not None:
        writer.release()
        print(f"wrote {args.out}  ({len(frames)} frames at {1 / drive.mean_dt_s:.2f} Hz)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default="2011_09_26_drive_0059")
    ap.add_argument("--frames", nargs=2, type=int, default=None, metavar=("START", "END"))
    ap.add_argument("--still", type=int, default=None, help="render a single frame and exit")
    ap.add_argument("--warmup", type=int, default=20, help="frames replayed before a still")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "replay.mp4")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lanes", action="store_true", help="overlay the lane solver")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
