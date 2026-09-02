"""Debug rendering. Not the HUD -- the HUD is Phase 8 and presents results.

Everything here exists to make a geometry bug visible. A projection that is
subtly wrong produces numbers that look plausible forever; projected onto the
image it is obvious in one glance.
"""

from __future__ import annotations

import cv2
import numpy as np

from aps.kitti.calibration import Calibration
from aps.kitti.tracklets import TrackletBox

# Depth colouring runs near=warm to far=cool over this span, in metres.
DEPTH_COLOUR_RANGE_M = (3.0, 60.0)

CLASS_COLOURS = {
    "Car": (80, 220, 100),
    "Van": (80, 200, 200),
    "Truck": (60, 160, 240),
    "Pedestrian": (240, 140, 90),
    "Cyclist": (220, 110, 220),
}
DEFAULT_COLOUR = (180, 180, 180)


def depth_colours(depth_m: np.ndarray) -> np.ndarray:
    """(N,) depths -> (N,3) BGR, near = warm, far = cool."""
    lo, hi = DEPTH_COLOUR_RANGE_M
    t = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    ramp = cv2.applyColorMap((255 * (1.0 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    return ramp.reshape(-1, 3)


def draw_lidar(
    img: np.ndarray,
    points_velo: np.ndarray,
    calib: Calibration,
    radius: int = 1,
    alpha: float = 0.85,
) -> np.ndarray:
    """Overlay projected LiDAR returns, coloured by depth. Returns a new image.

    If the projection chain is wrong this looks wrong immediately: points slide
    off the objects they belong to, road returns climb the image, or the cloud
    mirrors left-right. That is the entire point of drawing it.
    """
    out = img.copy()
    uv, depth = calib.project_velo(points_velo)
    mask = calib.in_image(uv, depth)
    uv, depth = uv[mask].astype(np.int32), depth[mask]
    if len(uv) == 0:
        return out

    layer = out.copy()
    for (u, v), colour in zip(uv, depth_colours(depth), strict=True):
        cv2.circle(layer, (int(u), int(v)), radius, tuple(int(c) for c in colour), -1)
    return cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)


def draw_box_2d(
    img: np.ndarray,
    box: np.ndarray,
    label: str = "",
    colour: tuple[int, int, int] = DEFAULT_COLOUR,
    thickness: int = 2,
) -> None:
    """Draw an [x1,y1,x2,y2] box with a label plate. Mutates `img`."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    if not label:
        return
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    top = max(0, y1 - th - base - 3)
    cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + base + 3), colour, -1)
    cv2.putText(
        img,
        label,
        (x1 + 3, top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def draw_box_3d(
    img: np.ndarray,
    tbox: TrackletBox,
    calib: Calibration,
    colour: tuple[int, int, int] | None = None,
    thickness: int = 1,
) -> None:
    """Project a labelled 3D box's wireframe. Mutates `img`.

    Drawn as a wireframe rather than a 2D box because a wireframe reveals a yaw
    or dimension error that an axis-aligned box hides completely.
    """
    colour = colour or CLASS_COLOURS.get(tbox.object_type, DEFAULT_COLOUR)
    uv, depth = calib.project_rect0(calib.velo_to_rect0(tbox.corners_velo))
    if (depth < 0.5).any() or not np.isfinite(uv).all():
        return
    p = uv.astype(np.int32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
        (0, 5), (1, 4),                  # front-face cross: marks the heading
    ]  # fmt: skip
    for a, b in edges:
        cv2.line(img, tuple(p[a]), tuple(p[b]), colour, thickness, cv2.LINE_AA)


def banner(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (8, 8)) -> None:
    """Top-left text block on a dark plate. Mutates `img`."""
    x, y = origin
    pad, lh = 6, 16
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] for t in lines)
    plate = img[y : y + lh * len(lines) + pad, x : x + width + 2 * pad]
    cv2.addWeighted(plate, 0.25, np.zeros_like(plate), 0.75, 0, plate)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (x + pad, y + lh * (i + 1) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
