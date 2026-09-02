"""Closed-form tests for the projection chain.

Every case here has an answer computable by hand. Projection code that is
subtly wrong still produces plausible-looking pictures and entirely wrong
metres, so "the overlay looked right" is not evidence and these are.
"""

from __future__ import annotations

import numpy as np
import pytest

from aps.kitti.calibration import Calibration

FX, FY, CX, CY = 700.0, 700.0, 600.0, 180.0
BASELINE_M = 0.06  # KITTI's P_rect_02[0,3] is positive; see Calibration.cam_offset


def make_calib(baseline_m: float = BASELINE_M) -> Calibration:
    """A pinhole camera with no rectifying rotation and a pure-x camera offset.

    Built by hand rather than loaded from KITTI so the expected pixel of any
    3D point is arithmetic, not a fixture nobody can check.
    """
    # P = K @ [I | t]. KITTI's P_rect_02[0,3] is POSITIVE, so t_x = +baseline.
    p = np.array(
        [
            [FX, 0.0, CX, FX * baseline_m],
            [0.0, FY, CY, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    return Calibration(
        P_rect=dict.fromkeys((0, 1, 2, 3), p),
        R_rect0=np.eye(4),
        Tr_velo_to_cam=np.eye(4),
        image_size=dict.fromkeys((0, 1, 2, 3), (1242, 375)),
        source_dir=__import__("pathlib").Path("synthetic"),
    )


def test_point_on_optical_axis_lands_at_principal_point():
    calib = make_calib(baseline_m=0.0)
    uv, depth = calib.project_rect0(np.array([[0.0, 0.0, 10.0]]))
    assert uv[0] == pytest.approx([CX, CY])
    assert depth[0] == pytest.approx(10.0)


def test_pixel_offset_follows_the_pinhole_relation():
    """u = fx * X/Z + cx, exactly."""
    calib = make_calib(baseline_m=0.0)
    x, y, z = 2.0, 1.5, 25.0
    uv, depth = calib.project_rect0(np.array([[x, y, z]]))
    assert uv[0, 0] == pytest.approx(FX * x / z + CX)
    assert uv[0, 1] == pytest.approx(FY * y / z + CY)
    assert depth[0] == pytest.approx(z)


def test_depth_is_the_homogeneous_divisor_not_the_euclidean_range():
    """The distinction D-003 turns on: a point off-axis has depth < range."""
    calib = make_calib(baseline_m=0.0)
    pt = np.array([[10.0, 0.0, 30.0]])
    _, depth = calib.project_rect0(pt)
    assert depth[0] == pytest.approx(30.0)
    assert np.linalg.norm(pt) == pytest.approx(np.hypot(10.0, 30.0))
    assert np.linalg.norm(pt) > depth[0]


def test_range_doubles_when_apparent_size_halves():
    """d = f*H/h -- the relation the whole monocular estimator rests on."""
    calib = make_calib(baseline_m=0.0)
    height_m = 1.5
    heights_px = []
    for z in (20.0, 40.0):
        top = calib.project_rect0(np.array([[0.0, -height_m / 2, z]]))[0]
        bottom = calib.project_rect0(np.array([[0.0, +height_m / 2, z]]))[0]
        heights_px.append(bottom[0, 1] - top[0, 1])
    assert heights_px[0] == pytest.approx(2 * heights_px[1])
    assert FY * height_m / heights_px[0] == pytest.approx(20.0)


def test_cam_offset_recovers_the_baseline():
    calib = make_calib(baseline_m=0.06)
    assert calib.cam_offset(2) == pytest.approx([0.06, 0.0, 0.0])


def test_camera_offset_shifts_pixels_but_not_depth():
    """Why rect0 depth and cam2 depth are interchangeable for KITTI.

    The pixel shift is +fx*t_x/Z: cam2 is mounted left of cam0, so a world
    point lands further RIGHT in its image. Getting this sign backwards is
    invisible in an overlay and shifts nothing in depth, which is exactly why
    it is asserted here.
    """
    pt = np.array([[0.0, 0.0, 20.0]])
    plain = make_calib(baseline_m=0.0)
    offset = make_calib(baseline_m=0.06)
    uv_a, depth_a = plain.project_rect0(pt)
    uv_b, depth_b = offset.project_rect0(pt)
    assert depth_a[0] == pytest.approx(depth_b[0])
    assert uv_b[0, 0] == pytest.approx(uv_a[0, 0] + FX * 0.06 / 20.0)


def test_points_behind_the_camera_have_nonpositive_depth():
    """They must be excluded on depth, never on whether the pixel looks sane.

    A point behind the camera projects to a finite, plausible pixel with a
    negative divisor -- filtering on image bounds alone silently admits it.
    """
    calib = make_calib()
    uv, depth = calib.project_rect0(np.array([[1.0, 0.5, -15.0]]))
    assert depth[0] < 0
    assert np.isfinite(uv).all()
    assert not calib.in_image(uv, depth)[0]


def test_in_image_rejects_out_of_frame_pixels():
    calib = make_calib(baseline_m=0.0)
    pts = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    mask = calib.in_image(*calib.project_rect0(pts))
    assert mask.tolist() == [True, False, True]


def test_velo_to_rect0_applies_the_full_chain():
    """velo -> cam -> rect, with a non-identity extrinsic and rotation."""
    calib = make_calib()
    # 90 deg about z, then translate 1 m along x.
    tr = np.eye(4)
    tr[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tr[:3, 3] = [1.0, 0.0, 0.0]
    calib = Calibration(calib.P_rect, np.eye(4), tr, calib.image_size, calib.source_dir)
    out = calib.velo_to_rect0(np.array([[0.0, 1.0, 0.0]]))
    assert out[0] == pytest.approx([0.0, 0.0, 0.0])


def test_reflectance_column_is_ignored():
    """(N,4) velodyne rows must project identically to their (N,3) prefix."""
    calib = make_calib()
    xyz = np.array([[3.0, 1.0, 0.5], [12.0, -2.0, 0.1]])
    xyzr = np.hstack([xyz, np.array([[0.9], [0.2]])])
    assert calib.velo_to_rect0(xyzr) == pytest.approx(calib.velo_to_rect0(xyz))


def test_malformed_input_raises():
    calib = make_calib()
    with pytest.raises(ValueError):
        calib.velo_to_rect0(np.zeros((5, 2)))
