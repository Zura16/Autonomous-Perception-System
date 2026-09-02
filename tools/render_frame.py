"""Render one frame with projected LiDAR and labelled 3D boxes.

This is the Phase 0 deliverable and the standing calibration check: if the
velodyne -> rect0 -> image chain is wrong, this picture shows it. Regenerate the
committed figure with:

    python tools/render_frame.py --drive 2011_09_26_drive_0013 --frame 20 \
        --out docs/figures/phase0_projection_check.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from aps.kitti import RawDrive, load_tracklets
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame
from aps.viz import CLASS_COLOURS, DEFAULT_COLOUR, banner, draw_box_2d, draw_box_3d, draw_lidar

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drive", default="2011_09_26_drive_0013")
    ap.add_argument("--frame", type=int, default=20)
    ap.add_argument("--out", type=Path, default=REPO / "artifacts" / "frame.png")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-boxes", action="store_true")
    ap.add_argument("--box-2d", action="store_true", help="also draw the derived 2D boxes")
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "dataset.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    meta = next((d for d in cfg["drives"] if d["id"] == args.drive), {})
    drive = RawDrive.open(
        REPO / cfg["root"],
        cfg["calib_date"],
        args.drive,
        meta.get("split", ""),
        meta.get("scene", ""),
    )
    frame = drive.frame(args.frame)
    img = frame.image()
    ox = frame.oxts()

    if not args.no_lidar:
        img = draw_lidar(img, frame.points(min_forward_m=0.5), drive.calib)

    n_boxes = 0
    if not args.no_boxes:
        boxes = boxes_by_frame(
            load_tracklets(drive.dir / "tracklet_labels.xml"), classes=FCW_CLASSES
        ).get(args.frame, [])
        # Far to near, so nearer objects draw on top.
        for tbox in sorted(boxes, key=lambda b: -b.ranges_m(drive.calib)[0]):
            near_m, _ = tbox.ranges_m(drive.calib)
            draw_box_3d(img, tbox, drive.calib)
            box2d = tbox.box_2d(drive.calib)
            if box2d is None:
                continue
            n_boxes += 1
            colour = CLASS_COLOURS.get(tbox.object_type, DEFAULT_COLOUR)
            if args.box_2d:
                draw_box_2d(img, box2d, "", colour, thickness=1)
            draw_box_2d(
                img,
                box2d,
                f"#{tbox.track_id} {tbox.object_type} {near_m:.1f}m",
                colour,
                thickness=0,
            )

    banner(
        img,
        [
            f"{drive.id}  frame {args.frame:04d}  t={frame.t_s:.3f}s  [{drive.split}/{drive.scene}]",
            f"ego {ox.vf_mps * 3.6:5.1f} km/h   pitch {np.degrees(ox.pitch_rad):+.2f} deg"
            f"   labelled objects: {n_boxes}",
            "LiDAR depth: warm = near, cool = far   |   GROUND TRUTH ONLY, not an input",
        ],
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]}, {n_boxes} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
