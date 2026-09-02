"""Report label density and range distribution per drive.

Two modes:

    python tools/audit_labels.py                    # drives already on disk
    python tools/audit_labels.py --probe 0059 0084  # fetch tracklets ONLY, then audit

The probe mode exists because a KITTI tracklet zip is ~1.5 MB against ~1.5 GB
for the drive it labels. Choosing a split by downloading drives and looking at
them afterwards costs an hour per mistake; choosing it by auditing labels first
costs seconds. The current val split was selected this way -- see
docs/decisions.md D-005.

Probing a drive is not peeking at a test set: it reads label metadata (how many
objects, at what ranges), never model performance. Test drives are excluded from
probe mode anyway, on the principle that the cheapest way to avoid a leak is to
not build the path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_kitti import BASE, download  # noqa: E402

from aps.kitti import Calibration, load_tracklets  # noqa: E402
from aps.kitti.tracklets import FCW_CLASSES, boxes_by_frame  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BINS = [(0, 10), (10, 20), (20, 30), (30, 50), (50, np.inf)]
BIN_LABELS = ["0-10", "10-20", "20-30", "30-50", "50+"]


def audit(xml: Path, calib: Calibration, n_frames: int | None = None) -> dict:
    tracklets = load_tracklets(xml)
    by_frame = boxes_by_frame(tracklets, classes=FCW_CLASSES)
    boxes = [b for f in by_frame for b in by_frame[f]]
    ranges = np.array([b.ranges_m(calib)[0] for b in boxes if b.box_2d(calib) is not None])
    return {
        "tracklets": len(tracklets),
        "boxes": len(boxes),
        "imageable": len(ranges),
        "frames_covered": len(by_frame),
        "n_frames": n_frames,
        "types": dict(Counter(t.object_type for t in tracklets)),
        "bins": [int(((ranges >= lo) & (ranges < hi)).sum()) for lo, hi in BINS],
        "ranges": ranges,
    }


def print_row(name: str, split: str, a: dict) -> None:
    cov = f"{100 * a['frames_covered'] / a['n_frames']:.0f}%" if a["n_frames"] else "  -"
    bins = "/".join(f"{b:>4}" for b in a["bins"])
    print(f"{name:<26}{split:<6}{a['tracklets']:>5}{a['boxes']:>7}{cov:>6}  {bins}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=REPO / "configs" / "dataset.yaml")
    ap.add_argument("--split", nargs="*", help="restrict to these splits")
    ap.add_argument(
        "--probe",
        nargs="*",
        metavar="NNNN",
        help="drive numbers to probe by fetching tracklets only (e.g. 0059 0084)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    root = REPO / cfg["root"]
    date = cfg["calib_date"]
    calib = Calibration.from_dir(root / date)

    print(
        f"{'drive':<26}{'split':<6}{'trk':>5}{'boxes':>7}{'cov':>6}  "
        + "/".join(f"{b:>4}" for b in BIN_LABELS)
    )
    print(f"{'':<26}{'':<6}{'':>5}{'':>7}{'':>6}  near-face range bins (m)")
    print("-" * 84)

    if args.probe:
        held_out = {d["id"] for d in cfg["drives"] if d["split"] == "test"}
        tmp = Path(tempfile.mkdtemp())
        try:
            for num in args.probe:
                did = f"{date}_drive_{num}"
                if did in held_out:
                    print(f"{did:<26}SKIPPED -- held-out test drive")
                    continue
                z = download(f"{BASE}/{did}/{did}_tracklets.zip", tmp / f"{num}.zip")
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(tmp / num)
                print_row(
                    did, "probe", audit(next((tmp / num).rglob("tracklet_labels.xml")), calib)
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0

    totals: dict[str, list[int]] = {}
    for d in cfg["drives"]:
        if args.split and d["split"] not in args.split:
            continue
        xml = root / date / f"{d['id']}_sync" / "tracklet_labels.xml"
        if not xml.exists():
            print(f"{d['id']:<26}{d['split']:<6}  not on disk -- python tools/fetch_kitti.py")
            continue
        a = audit(xml, calib, d["frames"])
        print_row(d["id"], d["split"], a)
        acc = totals.setdefault(d["split"], [0] * len(BINS) + [0])
        for i, b in enumerate(a["bins"]):
            acc[i] += b
        acc[-1] += a["boxes"]

    print("-" * 84)
    for split, acc in sorted(totals.items()):
        bins = "/".join(f"{b:>4}" for b in acc[:-1])
        print(f"{'TOTAL ' + split:<26}{'':<6}{'':>5}{acc[-1]:>7}{'':>6}  {bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
