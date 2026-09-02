"""KITTI raw calibration: parsing, and the velodyne -> image projection chain.

Coordinate frames, named explicitly because every bug in this stack is a frame
bug (see docs/glossary.md):

    velo    LiDAR frame.        x forward, y left,  z up.
    cam0    Unrectified left grayscale camera.
    rect0   Rectified reference frame. x right, y down, z FORWARD (optical axis).
    cam2    Rectified left COLOUR camera -- the camera this project uses.
            Differs from rect0 by a near-pure translation along x (the baseline).

The chain, as KITTI defines it:

    X_rect0 = R_rect_00 @ (Tr_velo_to_cam @ X_velo)
    [u v w] = P_rect_02 @ [X_rect0; 1]          u_px = u/w, v_px = v/w

and the homogeneous w that falls out of the last line *is* the depth of the
point in the cam2 frame, which is why depth never needs a separate computation.

`range_m` in this project means that depth -- the longitudinal distance along
the optical axis, NOT the Euclidean distance to the point. TTC is about closing
along the direction of travel, and the monocular pinhole relation d = f*H/h
recovers exactly this quantity. Euclidean range is available as
`euclidean_range_m` for the rare case it is wanted; mixing the two is a silent
error of up to a few percent at wide image angles. See docs/decisions.md D-003.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _parse_calib_file(path: Path) -> dict[str, np.ndarray]:
    """Parse a KITTI `key: v1 v2 ...` calibration file into float arrays.

    Non-numeric values (e.g. `calib_time`) are skipped rather than raised on:
    they are metadata, and every KITTI calibration file carries one.
    """
    out: dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        try:
            out[key.strip()] = np.array([float(x) for x in rest.split()])
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class Calibration:
    """Rectified projection matrices and the velodyne extrinsic, for one KITTI date.

    Calibration is per *date*, not per drive: every drive recorded on the same
    day shares it. This is why configs/dataset.yaml pins all five drives to
    2011_09_26 -- one intrinsics set for every row in benchmarks.md.
    """

    P_rect: dict[int, np.ndarray]  # cam -> 3x4
    R_rect0: np.ndarray  # 4x4 (3x3 rectifying rotation, homogeneous)
    Tr_velo_to_cam: np.ndarray  # 4x4
    image_size: dict[int, tuple[int, int]]  # cam -> (width, height), rectified
    source_dir: Path

    # ---------------------------------------------------------------- loading

    @classmethod
    def from_dir(cls, calib_dir: Path) -> Calibration:
        calib_dir = Path(calib_dir)
        c2c = _parse_calib_file(calib_dir / "calib_cam_to_cam.txt")
        v2c = _parse_calib_file(calib_dir / "calib_velo_to_cam.txt")

        p_rect = {cam: c2c[f"P_rect_{cam:02d}"].reshape(3, 4) for cam in (0, 1, 2, 3)}

        r_rect0 = np.eye(4)
        r_rect0[:3, :3] = c2c["R_rect_00"].reshape(3, 3)

        tr = np.eye(4)
        tr[:3, :3] = v2c["R"].reshape(3, 3)
        tr[:3, 3] = v2c["T"]

        sizes = {
            cam: (int(c2c[f"S_rect_{cam:02d}"][0]), int(c2c[f"S_rect_{cam:02d}"][1]))
            for cam in (0, 1, 2, 3)
        }
        return cls(p_rect, r_rect0, tr, sizes, calib_dir)

    # ------------------------------------------------------------- intrinsics

    def intrinsics(self, cam: int = 2) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in pixels for a rectified camera."""
        p = self.P_rect[cam]
        return float(p[0, 0]), float(p[1, 1]), float(p[0, 2]), float(p[1, 2])

    @property
    def fx(self) -> float:
        return float(self.P_rect[2][0, 0])

    @property
    def fy(self) -> float:
        return float(self.P_rect[2][1, 1])

    def cam_offset(self, cam: int = 2) -> np.ndarray:
        """The translation `t` satisfying `X_cam = X_rect0 + t`, in metres.

        P_rect_cc = K_cc @ [I | t], so t = inv(K) @ P[:, 3].

        Sign convention, stated because it is easy to get backwards and the
        error is invisible: KITTI's P_rect_02[0, 3] is POSITIVE (+44.86 px),
        giving t_x = +0.06 m. That is the translation applied to a point, which
        is the negation of where the camera's optical centre sits -- cam2 is
        mounted ~6 cm to the LEFT of cam0, so world points shift right in its
        image. Measured on 2011_09_26: t = [+0.0598, -0.0004, +0.0027] m.

        The z component being ~0 is what makes rect0 depth and cam2 depth
        interchangeable. Computed rather than assumed, because "essentially
        zero" is not a unit test.
        """
        p = self.P_rect[cam]
        return np.linalg.solve(p[:, :3], p[:, 3])

    # ------------------------------------------------------------ projections

    def velo_to_rect0(self, pts_velo: np.ndarray) -> np.ndarray:
        """(N,3) points in the velodyne frame -> (N,3) in the rectified frame."""
        pts = np.asarray(pts_velo, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] not in (3, 4):
            raise ValueError(f"expected (N,3) or (N,4) points, got {pts.shape}")
        xyz = pts[:, :3]
        hom = np.hstack([xyz, np.ones((len(xyz), 1))])
        return (self.R_rect0 @ self.Tr_velo_to_cam @ hom.T).T[:, :3]

    def project_rect0(self, pts_rect0: np.ndarray, cam: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) rectified points -> ((N,2) pixels, (N,) depth in the cam frame).

        Depth is the homogeneous divisor, i.e. the z coordinate in `cam`'s own
        frame. Points at or behind the image plane come back with depth <= 0 and
        pixel coordinates that are meaningless -- callers must filter on depth,
        never on whether the pixel happens to land inside the image.
        """
        pts = np.asarray(pts_rect0, dtype=np.float64)
        hom = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
        uvw = (self.P_rect[cam] @ hom.T).T
        depth = uvw[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uvw[:, :2] / depth[:, None]
        return uv, depth

    def project_velo(self, pts_velo: np.ndarray, cam: int = 2) -> tuple[np.ndarray, np.ndarray]:
        """(N,3|4) velodyne points -> ((N,2) pixels, (N,) depth in metres)."""
        return self.project_rect0(self.velo_to_rect0(pts_velo), cam=cam)

    def in_image(self, uv: np.ndarray, depth: np.ndarray, cam: int = 2) -> np.ndarray:
        """Boolean mask: projects in front of the camera AND lands on the sensor."""
        w, h = self.image_size[cam]
        return (
            (depth > 0)
            & np.isfinite(uv).all(axis=1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )

    def euclidean_range_m(self, pts_velo: np.ndarray, cam: int = 2) -> np.ndarray:
        """Euclidean distance from `cam`'s optical centre. NOT the project's `range_m`."""
        rect = self.velo_to_rect0(pts_velo)
        return np.linalg.norm(rect - self.cam_offset(cam), axis=1)

    def summary(self) -> str:
        fx, fy, cx, cy = self.intrinsics(2)
        w, h = self.image_size[2]
        t = self.cam_offset(2)
        return (
            f"KITTI calibration from {self.source_dir.name}\n"
            f"  cam2 rectified size : {w} x {h} px\n"
            f"  fx, fy              : {fx:.3f}, {fy:.3f} px\n"
            f"  cx, cy              : {cx:.3f}, {cy:.3f} px\n"
            f"  horizontal FOV      : {2 * np.degrees(np.arctan(w / (2 * fx))):.1f} deg\n"
            f"  rect0 -> cam2 offset: [{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m"
        )
