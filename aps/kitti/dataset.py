"""KITTI raw drive access: frames, images, point clouds, ego motion.

Frames are lazy. A drive is 150--450 frames of 1242x375 PNG plus ~120k LiDAR
points each; eagerly loading a drive is how an eval script becomes an OOM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml

from aps.kitti.calibration import Calibration

REPO = Path(__file__).resolve().parents[2]

# OXTS column indices (see dataformat.txt shipped with every raw drive).
_OXTS_ROLL, _OXTS_PITCH, _OXTS_YAW = 3, 4, 5
_OXTS_VF, _OXTS_VL, _OXTS_VU = 8, 9, 10
_OXTS_AF = 14
_OXTS_WY = 18


@dataclass(frozen=True)
class Oxts:
    """Ego state from the OXTS RT3003 IMU/GNSS unit, for one frame.

    `pitch_rad` is the single most consequential field in this project. Flat-
    ground monocular range assumes zero camera pitch, and a 2 deg error alone
    already exceeds 10% range error -- so the spread of this value over a drive
    is a direct, measurable bound on the geometry's honesty. It is the vehicle's
    pitch in the navigation frame, not the camera's pitch relative to the road,
    so it is an indicator and a sanity check, not a correction term.
    See docs/error-budget.md and CLAUDE.md hard rule 5.
    """

    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    vf_mps: float  # forward velocity, vehicle frame
    vl_mps: float  # leftward velocity
    vu_mps: float  # upward velocity
    af_mps2: float  # forward acceleration
    wy_radps: float  # angular rate about the vehicle's y (pitch rate)

    @classmethod
    def from_file(cls, path: Path) -> Oxts:
        v = np.array(path.read_text().split(), dtype=np.float64)
        return cls(
            roll_rad=float(v[_OXTS_ROLL]),
            pitch_rad=float(v[_OXTS_PITCH]),
            yaw_rad=float(v[_OXTS_YAW]),
            vf_mps=float(v[_OXTS_VF]),
            vl_mps=float(v[_OXTS_VL]),
            vu_mps=float(v[_OXTS_VU]),
            af_mps2=float(v[_OXTS_AF]),
            wy_radps=float(v[_OXTS_WY]),
        )


@dataclass(frozen=True)
class Frame:
    """One synchronised camera + LiDAR sample."""

    drive: RawDrive
    index: int
    t_s: float  # seconds since the first frame of this drive (camera clock)

    @property
    def image_path(self) -> Path:
        return self.drive.dir / "image_02" / "data" / f"{self.index:010d}.png"

    @property
    def velo_path(self) -> Path:
        return self.drive.dir / "velodyne_points" / "data" / f"{self.index:010d}.bin"

    def image(self) -> np.ndarray:
        """(H, W, 3) BGR uint8, as OpenCV wants it."""
        import cv2

        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"unreadable image: {self.image_path}")
        return img

    @property
    def has_velodyne(self) -> bool:
        """Whether this frame has a LiDAR scan.

        Not every KITTI raw drive ships one scan per image: `drive_0009` has 447
        images and 443 velodyne files. Ground-truth harnesses must skip those
        frames rather than assume a scan exists -- and must do so visibly, since
        a silently skipped frame is a silently shrunken denominator.
        """
        return self.velo_path.exists()

    def points(self, min_forward_m: float = 0.0) -> np.ndarray:
        """(N, 4) LiDAR returns as [x, y, z, reflectance] in the velodyne frame.

        `min_forward_m` drops returns at or behind the sensor origin. The
        Velodyne sweeps a full 360 deg, so slightly over half of every cloud is
        behind the car and can only project into the image as garbage.
        """
        pts = np.fromfile(self.velo_path, dtype=np.float32).reshape(-1, 4)
        if min_forward_m > 0:
            pts = pts[pts[:, 0] > min_forward_m]
        return pts

    def oxts(self) -> Oxts:
        return Oxts.from_file(self.drive.dir / "oxts" / "data" / f"{self.index:010d}.txt")

    def __repr__(self) -> str:
        return f"Frame({self.drive.id}#{self.index:04d} t={self.t_s:.3f}s)"


@dataclass
class RawDrive:
    """One KITTI raw drive, e.g. 2011_09_26_drive_0013_sync."""

    id: str
    dir: Path
    split: str
    scene: str
    calib: Calibration

    @classmethod
    def open(
        cls, root: Path, date: str, drive_id: str, split: str = "", scene: str = ""
    ) -> RawDrive:
        root = Path(root)
        d = root / date / f"{drive_id}_sync"
        if not d.is_dir():
            raise FileNotFoundError(
                f"drive not found: {d}\n"
                f"fetch it with:  python tools/fetch_kitti.py --drive {drive_id}"
            )
        return cls(drive_id, d, split, scene, Calibration.from_dir(root / date))

    @cached_property
    def timestamps_s(self) -> np.ndarray:
        """Camera frame times in seconds, relative to frame 0.

        Read from the sensor, never assumed to be a uniform 10 Hz. Velocity and
        TTC divide by dt; baking in a nominal rate silently biases every
        closing-speed number by the sync jitter.
        """
        raw = (self.dir / "image_02" / "timestamps.txt").read_text().splitlines()
        # KITTI writes nanoseconds; datetime carries microseconds. The discarded
        # digits are ~1e-9 s against frame intervals of ~1e-1 s.
        t = [datetime.fromisoformat(line.strip()[:26]) for line in raw if line.strip()]
        t0 = t[0]
        return np.array([(x - t0).total_seconds() for x in t])

    @cached_property
    def n_frames(self) -> int:
        return len(list((self.dir / "image_02" / "data").glob("*.png")))

    @cached_property
    def mean_dt_s(self) -> float:
        return float(np.mean(np.diff(self.timestamps_s)))

    def frame(self, index: int) -> Frame:
        if not 0 <= index < self.n_frames:
            raise IndexError(f"{self.id}: frame {index} outside [0, {self.n_frames})")
        return Frame(self, index, float(self.timestamps_s[index]))

    def frames(self) -> list[Frame]:
        return [self.frame(i) for i in range(self.n_frames)]

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self):
        for i in range(self.n_frames):
            yield self.frame(i)

    def __repr__(self) -> str:
        return f"RawDrive({self.id} split={self.split} scene={self.scene} n={self.n_frames})"


def load_split(
    split: str | list[str] | None = None,
    config: Path = REPO / "configs" / "dataset.yaml",
) -> list[RawDrive]:
    """Open every drive belonging to `split` ('dev' | 'val' | 'test' | None = all).

    Going through the config rather than globbing the data directory is the
    point: the split is a committed, reviewable artefact, so a benchmark row can
    always be traced back to the exact sequences that produced it.
    """
    cfg = yaml.safe_load(Path(config).read_text())
    root = REPO / cfg["root"]
    date = cfg["calib_date"]
    wanted = [split] if isinstance(split, str) else split

    drives = []
    for d in cfg["drives"]:
        if wanted and d["split"] not in wanted:
            continue
        drives.append(RawDrive.open(root, date, d["id"], d["split"], d["scene"]))
    if not drives:
        raise ValueError(f"no drives matched split={split!r} in {config}")
    return drives
