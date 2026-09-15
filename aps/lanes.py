"""Ego-lane boundaries and lateral offset from a single camera.

Classical pipeline, and deliberately so -- a learned lane model would need
training data this project does not have, and the point of Phase 6 is a
measured failure set, not a leaderboard number:

    image --(metric bird's-eye warp)--> BEV --(top-hat on lightness)--> marking mask
          --(sliding windows from the ego position)--> boundary points
          --(quadratic fit in metres)--> left/right boundary x(z)
          --> lane-centre offset, lane width, departure warning

**The warp is metric.** Each bird's-eye pixel is a known patch of road surface
(`bev_resolution_m` square), built by projecting road-plane points through a
camera-to-road geometry. Lateral offsets therefore come out in metres directly,
and estimate and ground truth live in the same frame. A hand-picked trapezoid
warp -- the usual tutorial choice -- produces offsets in "BEV pixels" whose scale
is whatever the trapezoid happened to imply.

**Two geometries, and the difference is the point** (docs/decisions.md D-023):

  nominal  a level camera at the fixed Phase 3 height (1.655 m, measured on the
           raw drives). What a deployed camera-only system has.
  oracle   the per-frame `Tr_cam_to_road` shipped with the labels. Not available
           to a real system; used only to separate geometry error from marking-
           detection error.

**Marking extraction is a morphological top-hat, not a colour threshold.** A lane
marking is a bright stripe ~0.12 m wide. A top-hat with a structuring element
several times wider than that keeps thin bright structure and discards anything
broad -- sunlit asphalt, shadows' soft edges, the smeared body of a car ahead --
because broad structure survives the opening and is subtracted away. One
contrast parameter, with a physical meaning, instead of a stack of HLS bands.

All parameters live in configs/lanes.yaml and were frozen before the labelled set
was evaluated (D-023).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from aps.kitti.road import RoadGeometry

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LaneConfig:
    camera_height_m: float
    bev_x_min_m: float
    bev_x_max_m: float
    bev_z_min_m: float
    bev_z_max_m: float
    bev_resolution_m: float
    tophat_width_m: float
    tophat_contrast: float
    base_search_min_m: float
    base_search_max_m: float
    base_min_pixels: int
    n_windows: int
    window_half_width_m: float
    window_min_pixels: int
    fit_min_points: int
    fit_min_span_m: float
    lookahead_min_m: float
    lookahead_max_m: float
    vehicle_half_width_m: float
    departure_margin_m: float

    @classmethod
    def from_config(cls, path: Path | None = None) -> LaneConfig:
        c = yaml.safe_load((path or REPO / "configs" / "lanes.yaml").read_text())
        return cls(
            camera_height_m=float(c["geometry"]["camera_height_m"]),
            bev_x_min_m=float(c["bev"]["x_min_m"]),
            bev_x_max_m=float(c["bev"]["x_max_m"]),
            bev_z_min_m=float(c["bev"]["z_min_m"]),
            bev_z_max_m=float(c["bev"]["z_max_m"]),
            bev_resolution_m=float(c["bev"]["resolution_m"]),
            tophat_width_m=float(c["markings"]["tophat_width_m"]),
            tophat_contrast=float(c["markings"]["tophat_contrast"]),
            base_search_min_m=float(c["search"]["base_min_m"]),
            base_search_max_m=float(c["search"]["base_max_m"]),
            base_min_pixels=int(c["search"]["base_min_pixels"]),
            n_windows=int(c["search"]["n_windows"]),
            window_half_width_m=float(c["search"]["window_half_width_m"]),
            window_min_pixels=int(c["search"]["window_min_pixels"]),
            fit_min_points=int(c["fit"]["min_points"]),
            fit_min_span_m=float(c["fit"]["min_span_m"]),
            lookahead_min_m=float(c["metric"]["lookahead_min_m"]),
            lookahead_max_m=float(c["metric"]["lookahead_max_m"]),
            vehicle_half_width_m=float(c["departure"]["vehicle_half_width_m"]),
            departure_margin_m=float(c["departure"]["margin_m"]),
        )


# ── bird's-eye view ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BevGrid:
    """A metric road-surface grid. Row 0 is FAR, the last row is NEAR."""

    x_m: np.ndarray  # (W,) lateral, left negative
    z_m: np.ndarray  # (H,) forward, descending

    @classmethod
    def from_config(cls, cfg: LaneConfig) -> BevGrid:
        r = cfg.bev_resolution_m
        x = np.arange(cfg.bev_x_min_m, cfg.bev_x_max_m, r) + r / 2
        z = np.arange(cfg.bev_z_max_m, cfg.bev_z_min_m, -r) - r / 2
        return cls(x, z)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.z_m), len(self.x_m)

    def col_of(self, x_m: float) -> int:
        return int(np.clip(np.searchsorted(self.x_m, x_m), 0, len(self.x_m) - 1))


def warp_to_bev(image: np.ndarray, geometry: RoadGeometry, grid: BevGrid) -> np.ndarray:
    """Resample the image onto the road-plane grid. Off-image cells become 0."""
    xx, zz = np.meshgrid(grid.x_m, grid.z_m)
    u, v = geometry.road_to_pixel(xx, zz)
    return cv2.remap(
        image,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def marking_mask(bev_bgr: np.ndarray, cfg: LaneConfig) -> np.ndarray:
    """Thin bright stripes on the road surface -> boolean mask.

    The structuring element is horizontal because, in a bird's-eye view, lane
    markings run vertically: only their WIDTH is short. A 1-D horizontal top-hat
    therefore keeps a marking at any length while rejecting broad bright regions.
    """
    lightness = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)[:, :, 1]
    width_px = max(3, int(round(cfg.tophat_width_m / cfg.bev_resolution_m)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width_px, 1))
    tophat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, kernel)
    return tophat >= cfg.tophat_contrast


# ── boundary search and fit ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Boundary:
    """One lane boundary as x(z) = a·z² + b·z + c, in metres."""

    coeffs: np.ndarray  # (3,)
    z_min_m: float
    z_max_m: float
    n_points: int

    def x_at(self, z_m: np.ndarray | float) -> np.ndarray:
        return np.polyval(self.coeffs, z_m)

    def covers(self, z_m: float) -> bool:
        return self.z_min_m <= z_m <= self.z_max_m


def _track_boundary(
    mask: np.ndarray, grid: BevGrid, base_col: int, cfg: LaneConfig
) -> Boundary | None:
    h, _ = mask.shape
    half = max(1, int(round(cfg.window_half_width_m / cfg.bev_resolution_m)))
    win_h = h // cfg.n_windows
    ys, xs = np.nonzero(mask)
    centre = base_col
    keep_y: list[np.ndarray] = []
    keep_x: list[np.ndarray] = []

    for k in range(cfg.n_windows):
        y_hi = h - k * win_h
        y_lo = max(0, y_hi - win_h)
        inside = (ys >= y_lo) & (ys < y_hi) & (xs >= centre - half) & (xs < centre + half)
        if inside.sum() >= cfg.window_min_pixels:
            keep_y.append(ys[inside])
            keep_x.append(xs[inside])
            centre = int(np.mean(xs[inside]))
        # A window with too few pixels keeps its centre: dashed markings have
        # gaps, and re-centring on noise inside a gap walks the search off the lane.

    if not keep_y:
        return None
    rows = np.concatenate(keep_y)
    cols = np.concatenate(keep_x)
    z = grid.z_m[rows]
    x = grid.x_m[cols]
    if len(z) < cfg.fit_min_points or np.ptp(z) < cfg.fit_min_span_m:
        return None
    return Boundary(np.polyfit(z, x, 2), float(z.min()), float(z.max()), len(z))


def _base_column(
    hist: np.ndarray, grid: BevGrid, lo_m: float, hi_m: float, min_px: int
) -> int | None:
    lo, hi = sorted((grid.col_of(lo_m), grid.col_of(hi_m)))
    band = hist[lo : hi + 1]
    if band.size == 0 or band.max() < min_px:
        return None
    return lo + int(np.argmax(band))


# ── the estimate ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LaneEstimate:
    """Ego-lane boundaries and the quantities derived from them."""

    left: Boundary | None
    right: Boundary | None
    centre_offset_m: float  # lane centre x at the lookahead; + means centre is to the RIGHT
    width_m: float
    departure_warning: bool
    lookahead_z_m: np.ndarray

    @property
    def detected(self) -> bool:
        return self.left is not None and self.right is not None


def summarise(left: Boundary | None, right: Boundary | None, cfg: LaneConfig) -> LaneEstimate:
    """Derive offset, width and departure from two boundaries (estimated or GT).

    Shared by the estimator and the ground truth, so both are reduced to numbers
    by the same arithmetic -- a difference between them cannot come from two
    subtly different ways of computing the offset.
    """
    z = np.linspace(cfg.lookahead_min_m, cfg.lookahead_max_m, 7)
    if left is None or right is None:
        return LaneEstimate(left, right, float("nan"), float("nan"), False, z)
    xl, xr = left.x_at(z), right.x_at(z)
    centre = float(np.median((xl + xr) / 2))
    width = float(np.median(xr - xl))
    # Distance from the vehicle centreline to the nearer boundary, at the near
    # end of the lookahead where a departure actually happens.
    clearance = min(-float(xl[0]), float(xr[0]))
    warning = clearance < cfg.vehicle_half_width_m + cfg.departure_margin_m
    return LaneEstimate(left, right, centre, width, warning, z)


def estimate_lane(
    image: np.ndarray, geometry: RoadGeometry, cfg: LaneConfig, grid: BevGrid | None = None
) -> tuple[LaneEstimate, np.ndarray]:
    """Run the full pipeline. Returns (estimate, marking mask in BEV)."""
    grid = grid or BevGrid.from_config(cfg)
    mask = marking_mask(warp_to_bev(image, geometry, grid), cfg)

    near_rows = mask[int(mask.shape[0] * 2 / 3) :]
    hist = near_rows.sum(axis=0)
    left_base = _base_column(
        hist, grid, -cfg.base_search_max_m, -cfg.base_search_min_m, cfg.base_min_pixels
    )
    right_base = _base_column(
        hist, grid, cfg.base_search_min_m, cfg.base_search_max_m, cfg.base_min_pixels
    )

    left = _track_boundary(mask, grid, left_base, cfg) if left_base is not None else None
    right = _track_boundary(mask, grid, right_base, cfg) if right_base is not None else None
    return summarise(left, right, cfg), mask


# ── ground truth from a labelled ego-lane mask ───────────────────────────────


def lane_from_mask(
    ego: np.ndarray, geometry: RoadGeometry, cfg: LaneConfig, row_step: int = 2
) -> LaneEstimate:
    """Ego-lane boundaries from a human label, projected onto the road plane.

    A row where the labelled lane touches the image edge does not show that
    boundary -- the lane continues out of frame -- so that side is skipped for
    that row rather than recorded at the image border. Recording the border would
    make every truncated lane look narrower than it is.
    """
    h, w = ego.shape
    pts: dict[str, list[tuple[float, float]]] = {"left": [], "right": []}
    for v in range(h - 1, 0, -row_step):
        cols = np.flatnonzero(ego[v])
        if len(cols) < 3:
            continue
        for side, col, edge in (("left", cols[0], 0), ("right", cols[-1], w - 1)):
            if col == edge:
                continue
            p = geometry.pixel_to_road(float(col), float(v))
            if p is None:
                continue
            x, z = p
            if cfg.bev_z_min_m <= z <= cfg.bev_z_max_m:
                pts[side].append((z, x))

    def fit(side: str) -> Boundary | None:
        arr = np.array(pts[side])
        if len(arr) < cfg.fit_min_points or np.ptp(arr[:, 0]) < cfg.fit_min_span_m:
            return None
        return Boundary(
            np.polyfit(arr[:, 0], arr[:, 1], 2), arr[:, 0].min(), arr[:, 0].max(), len(arr)
        )

    return summarise(fit("left"), fit("right"), cfg)
