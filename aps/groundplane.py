"""Road-plane fitting from LiDAR: camera height and camera-to-road pitch.

GROUND TRUTH ONLY. Nothing here may enter the monocular estimation path. Its
purpose is to measure the two quantities the flat-ground geometry *assumes*, so
that the assumption's cost can be reported as a number instead of a caveat:

    h_cam    the camera's height above the local road surface
    pitch    the camera's tilt relative to that surface

The second is the quantity that has been missing. OXTS reports the *vehicle's*
pitch in the navigation frame, which differs from camera-relative-to-road pitch
by suspension travel, road grade, and mounting error -- so OXTS could indicate a
problem but never size it (docs/decisions.md D-009). A plane fitted to the LiDAR
returns on the road measures exactly the right angle, in the right frame.

This is what makes Phase 3's pitch treatment a *measured sensitivity curve*
rather than a hypothetical sweep: we know the real distribution of pitch error,
so we can report what it actually costs rather than what it might.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aps.kitti.calibration import Calibration

# Region of the cloud searched for road, in the velodyne frame (x fwd, y left,
# z up). Deliberately conservative: near enough that returns are dense, far
# enough to define a plane rather than a patch, and below the sensor so that
# vehicle roofs and foliage cannot dominate the fit.
ROAD_X_RANGE_M = (5.0, 30.0)
ROAD_Y_ABS_M = 6.0
ROAD_Z_MAX_M = -1.2

RANSAC_ITERS = 200
RANSAC_INLIER_M = 0.05
MIN_INLIERS = 500


@dataclass(frozen=True)
class RoadPlane:
    """The local road surface, expressed relative to the camera."""

    height_m: float  # camera optical centre above the road
    pitch_deg: float  # camera tilt relative to road; positive = nose down
    roll_deg: float
    n_inliers: int
    normal_rect: np.ndarray  # unit normal in the rectified camera frame
    offset_rect: float

    @property
    def is_valid(self) -> bool:
        return self.n_inliers >= MIN_INLIERS and np.isfinite(self.height_m)


def fit_plane_ransac(
    points: np.ndarray,
    iters: int = RANSAC_ITERS,
    inlier_m: float = RANSAC_INLIER_M,
    seed: int = 0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC plane fit. Returns (unit normal, offset d, inlier mask) for n·x + d = 0.

    RANSAC rather than a least-squares fit because the candidate region still
    contains kerbs, low walls, and the occasional vehicle underside; least
    squares would let any of them tilt the plane, and a tilted plane is exactly
    the error being measured.
    """
    pts = np.asarray(points, dtype=np.float64)[:, :3]
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(len(pts), dtype=bool)

    rng = np.random.default_rng(seed)
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    best_inliers = np.zeros(len(pts), dtype=bool)

    for _ in range(iters):
        sample = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -normal @ sample[0]
        inliers = np.abs(pts @ normal + d) < inlier_m
        if inliers.sum() > best_inliers.sum():
            best_normal, best_d, best_inliers = normal, d, inliers

    # Refit on the consensus set: RANSAC picks the support, least squares over
    # that support gives a far better normal than three sampled points.
    if best_inliers.sum() >= 3:
        support = pts[best_inliers]
        centroid = support.mean(axis=0)
        _, _, vt = np.linalg.svd(support - centroid)
        best_normal = vt[-1]
        best_d = -best_normal @ centroid

    return best_normal, float(best_d), best_inliers


def road_candidates(points_velo: np.ndarray) -> np.ndarray:
    """Slice of the cloud plausibly belonging to the road ahead."""
    pts = np.asarray(points_velo, dtype=np.float64)
    m = (
        (pts[:, 0] > ROAD_X_RANGE_M[0])
        & (pts[:, 0] < ROAD_X_RANGE_M[1])
        & (np.abs(pts[:, 1]) < ROAD_Y_ABS_M)
        & (pts[:, 2] < ROAD_Z_MAX_M)
    )
    return pts[m][:, :3]


def fit_road_plane(points_velo: np.ndarray, calib: Calibration, cam: int = 2) -> RoadPlane:
    """Fit the road and express it relative to camera `cam`.

    A plane `n·x + d = 0` in the velodyne frame maps under `x_r = R x_v + t` to
    `n_r = R n`, `d_r = d - n_r·t` -- derived rather than approximated, because
    a sign error here would silently offset every height and pitch reading.
    """
    candidates = road_candidates(points_velo)
    if len(candidates) < 3:
        return RoadPlane(np.nan, np.nan, np.nan, 0, np.zeros(3), np.nan)

    normal_velo, d_velo, inliers = fit_plane_ransac(candidates)

    transform = calib.R_rect0 @ calib.Tr_velo_to_cam
    rotation, translation = transform[:3, :3], transform[:3, 3]
    normal_rect = rotation @ normal_velo
    d_rect = d_velo - normal_rect @ translation

    # Point the normal along -y (up, in the rectified frame) so that height and
    # pitch have a stable sign regardless of the RANSAC sample's orientation.
    if normal_rect[1] > 0:
        normal_rect, d_rect = -normal_rect, -d_rect

    origin = calib.cam_offset(cam)
    height = float(abs(normal_rect @ origin + d_rect) / np.linalg.norm(normal_rect))

    # With the camera level, the road normal in the rectified frame is (0,-1,0).
    # Tilting the camera nose-down by theta rotates it to (0, -cos, +sin), so
    # theta = atan2(n_z, -n_y). Roll is the same construction about x.
    pitch = float(np.degrees(np.arctan2(normal_rect[2], -normal_rect[1])))
    roll = float(np.degrees(np.arctan2(normal_rect[0], -normal_rect[1])))

    return RoadPlane(height, pitch, roll, int(inliers.sum()), normal_rect, float(d_rect))
