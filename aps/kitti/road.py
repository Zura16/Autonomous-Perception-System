"""KITTI road benchmark: images, ego-lane masks, and the per-image road plane.

Used for ONE purpose: a human-labelled lane set to score Phase 6 against. KITTI
raw has no lane labels, and hard rule 9 forbids scoring lane detection on a
hand-picked clip (docs/decisions.md D-023).

Only the `um` (urban marked) category carries ego-lane labels: 95 training
frames. The `umm` and `uu` categories have road-area labels only and are not
used. The benchmark's testing labels are withheld by its authors.

Mask encoding (BGR): magenta (255,0,255) = ego lane, red (0,0,255) = not ego
lane, black = outside the labelled region (don't care).

Each frame's calibration carries `Tr_cam_to_road`, a camera-to-road-plane
transform fitted per frame. In the road frame the road surface is `y = 0` with
`y` pointing down, as in the camera frame. It is used as GROUND-TRUTH geometry;
the lane estimator is not given it except in the explicitly-labelled oracle run.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROAD_ROOT = REPO / "data" / "kitti_road" / "data_road" / "training"


def _parse(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if rest.strip():
            out[key.strip()] = np.array([float(x) for x in rest.split()])
    return out


@dataclass(frozen=True)
class RoadGeometry:
    """Camera projection plus a camera-to-road rigid transform.

    `P2` projects rectified-reference (rect0) coordinates into image_2.
    `cam_to_road` maps rect0 coordinates into the road frame, where the road
    surface is the plane y = 0.
    """

    P2: np.ndarray  # 3x4
    cam_to_road: np.ndarray  # 4x4

    @classmethod
    def nominal(cls, P2: np.ndarray, camera_height_m: float) -> RoadGeometry:
        """A level camera `camera_height_m` above a flat road -- no per-frame fit.

        This is what a deployed camera-only system has: a fixed mounting
        calibration. The rect0 origin sits `h` above the road, and the road
        frame shares the camera's axes (y down), so the transform is a pure
        translation of -h along y.
        """
        T = np.eye(4)
        T[1, 3] = -camera_height_m
        return cls(P2, T)

    @property
    def camera_height_m(self) -> float:
        return float(-self.cam_to_road[1, 3])

    @cached_property
    def road_to_cam(self) -> np.ndarray:
        return np.linalg.inv(self.cam_to_road)

    def road_to_pixel(self, x_m: np.ndarray, z_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Road-surface points (lateral x, forward z, y = 0) -> image (u, v)."""
        x_m, z_m = np.broadcast_arrays(np.asarray(x_m, float), np.asarray(z_m, float))
        road = np.stack([x_m, np.zeros_like(x_m), z_m, np.ones_like(x_m)], axis=-1)
        cam = road @ self.road_to_cam.T
        uvw = cam @ self.P2.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return uvw[..., 0] / uvw[..., 2], uvw[..., 1] / uvw[..., 2]

    def pixel_to_road(self, u: float, v: float) -> tuple[float, float] | None:
        """Intersect the pixel's viewing ray with the road plane -> (x_m, z_m).

        Returns None for a ray at or above the horizon, which never meets the
        road -- rather than a huge or negative distance.
        """
        K = self.P2[:, :3]
        offset = np.linalg.solve(K, self.P2[:, 3])  # X_cam2 = X_rect0 + offset
        direction = np.linalg.solve(K, np.array([u, v, 1.0]))
        R, t = self.cam_to_road[:3, :3], self.cam_to_road[:3, 3]
        ray = R @ direction
        origin = R @ (-offset) + t
        if abs(ray[1]) < 1e-12:
            return None
        s = -origin[1] / ray[1]
        if s <= 0:
            return None
        p = s * ray + origin
        return float(p[0]), float(p[2])


@dataclass(frozen=True)
class RoadFrame:
    """One labelled `um` frame."""

    stem: str  # e.g. "um_000012"
    root: Path = ROAD_ROOT

    @property
    def image_path(self) -> Path:
        return self.root / "image_2" / f"{self.stem}.png"

    @property
    def mask_path(self) -> Path:
        idx = self.stem.split("_")[-1]
        return self.root / "gt_image_2" / f"um_lane_{idx}.png"

    def image(self) -> np.ndarray:
        import cv2

        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path)
        return img

    def lane_mask(self) -> tuple[np.ndarray, np.ndarray]:
        """(ego_lane, labelled) boolean masks. `labelled` excludes don't-care."""
        import cv2

        m = cv2.imread(str(self.mask_path), cv2.IMREAD_COLOR)
        if m is None:
            raise FileNotFoundError(self.mask_path)
        ego = (m[:, :, 0] == 255) & (m[:, :, 1] == 0) & (m[:, :, 2] == 255)
        labelled = m.any(axis=2)
        return ego, labelled

    @cached_property
    def _calib(self) -> dict[str, np.ndarray]:
        return _parse(self.root / "calib" / f"{self.stem}.txt")

    @property
    def P2(self) -> np.ndarray:
        return self._calib["P2"].reshape(3, 4)

    def gt_geometry(self) -> RoadGeometry:
        T = np.eye(4)
        T[:3, :] = self._calib["Tr_cam_to_road"].reshape(3, 4)
        return RoadGeometry(self.P2, T)


def load_lane_frames(root: Path = ROAD_ROOT) -> list[RoadFrame]:
    """All `um` frames that have an ego-lane label."""
    masks = sorted((root / "gt_image_2").glob("um_lane_*.png"))
    if not masks:
        raise FileNotFoundError(
            f"no um_lane masks under {root}; fetch data_road.zip (see docs/decisions.md D-023)"
        )
    return [RoadFrame(f"um_{p.stem.split('_')[-1]}", root) for p in masks]
