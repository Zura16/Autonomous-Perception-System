"""Tests for tracklet geometry: box corners, yaw, and the two range conventions."""

from __future__ import annotations

import numpy as np
import pytest

from aps.kitti.tracklets import (
    TRUNC_BEHIND_IMAGE,
    TRUNC_IN_IMAGE,
    TRUNC_OUT_IMAGE,
    Tracklet,
    TrackletBox,
    boxes_by_frame,
)
from tests.test_calibration import make_calib


def box(
    center=(20.0, 0.0, -1.5),
    dims=(4.0, 1.8, 1.5),
    yaw=0.0,
    frame=0,
    track_id=0,
    cls="Car",
    truncation=TRUNC_IN_IMAGE,
    occlusion=0,
) -> TrackletBox:
    return TrackletBox(
        frame=frame,
        track_id=track_id,
        object_type=cls,
        center_velo=np.array(center),
        dims_lwh_m=dims,
        yaw_rad=yaw,
        truncation=truncation,
        occlusion=occlusion,
    )


def test_corners_are_the_expected_box_at_zero_yaw():
    c = box(center=(10.0, 0.0, 0.0), dims=(4.0, 2.0, 1.5), yaw=0.0).corners_velo
    assert c.shape == (8, 3)
    assert c[:, 0].min() == pytest.approx(8.0)  # 10 - length/2
    assert c[:, 0].max() == pytest.approx(12.0)
    assert c[:, 1].min() == pytest.approx(-1.0)  # width/2 either side
    assert c[:, 1].max() == pytest.approx(1.0)
    # Translation is the BOTTOM centre: the box rises from z to z + height.
    assert c[:, 2].min() == pytest.approx(0.0)
    assert c[:, 2].max() == pytest.approx(1.5)


def test_yaw_of_ninety_degrees_swaps_length_and_width():
    c = box(center=(10.0, 0.0, 0.0), dims=(4.0, 2.0, 1.5), yaw=np.pi / 2).corners_velo
    assert c[:, 0].max() - c[:, 0].min() == pytest.approx(2.0)
    assert c[:, 1].max() - c[:, 1].min() == pytest.approx(4.0)


def test_yaw_does_not_move_the_centroid():
    for yaw in (0.0, 0.7, np.pi / 2, -2.3):
        c = box(center=(15.0, 2.0, -1.0), yaw=yaw).corners_velo
        assert c[:, :2].mean(axis=0) == pytest.approx([15.0, 2.0])


def test_near_face_is_half_a_length_closer_than_the_centroid():
    """The D-003 gap, computed rather than asserted.

    Velodyne +x is forward and the identity extrinsic in make_calib maps it to
    rect0 x, so the box is set up along the axis that becomes depth.
    """
    calib = make_calib(baseline_m=0.0)
    tr = np.eye(4)
    # velo(x fwd, y left, z up) -> rect0(x right, y down, z fwd)
    tr[:3, :3] = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    from aps.kitti.calibration import Calibration

    calib = Calibration(calib.P_rect, np.eye(4), tr, calib.image_size, calib.source_dir)

    near, center = box(center=(30.0, 0.0, 0.0), dims=(4.0, 1.8, 1.5)).ranges_m(calib)
    assert near == pytest.approx(28.0)  # 30 - 4/2
    assert center == pytest.approx(30.0)
    assert center - near == pytest.approx(2.0)


def test_box_2d_is_none_when_the_object_crosses_the_image_plane():
    calib = make_calib()
    assert box(center=(0.1, 0.0, 0.0), dims=(4.0, 1.8, 1.5)).box_2d(calib) is None


def test_boxes_by_frame_indexes_by_frame_and_filters_classes():
    t_car = Tracklet(0, "Car", (4, 2, 1.5), 5, [box(frame=f, track_id=0) for f in (5, 6, 7)])
    t_tram = Tracklet(1, "Tram", (4, 2, 1.5), 5, [box(frame=5, track_id=1, cls="Tram")])
    by_frame = boxes_by_frame([t_car, t_tram], classes=frozenset({"Car"}))
    assert sorted(by_frame) == [5, 6, 7]
    assert len(by_frame[5]) == 1
    assert by_frame[5][0].object_type == "Car"


def test_boxes_by_frame_drops_objects_outside_the_image():
    """Counting them as missed detections would measure the FOV, not the detector."""
    t = Tracklet(
        0,
        "Car",
        (4, 2, 1.5),
        0,
        [
            box(frame=0, truncation=TRUNC_IN_IMAGE),
            box(frame=1, truncation=TRUNC_OUT_IMAGE),
            box(frame=2, truncation=TRUNC_BEHIND_IMAGE),
        ],
    )
    assert sorted(boxes_by_frame([t], classes=frozenset({"Car"}))) == [0]
    assert sorted(boxes_by_frame([t], classes=frozenset({"Car"}), drop_out_of_image=False)) == [
        0,
        1,
        2,
    ]
