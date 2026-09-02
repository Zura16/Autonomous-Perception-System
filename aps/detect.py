"""YOLOv8 detection, and the COCO -> APS class mapping.

The mapping is the load-bearing part of this module, not the model call.

KITTI's classes and COCO's do not correspond, and pretending they do is how a
detection number becomes uninterpretable (docs/decisions.md D-014):

  * KITTI `Van` is COCO `car` for some vans and `truck` for others. There is no
    rule; the boundary is a judgement the COCO annotators made per image.
  * KITTI `Cyclist` is ONE object. COCO emits `person` AND `bicycle` for it, so
    per-class matching scores one as a hit and the other as a false alarm for an
    object the detector plainly found.
  * KITTI `Person (sitting)` has no COCO analogue at all.

So evaluation is reported at two granularities, and fine-grained KITTI classes
are NOT recovered:

  class-agnostic   the headline. FCW asks "is there an obstacle at range D
                   closing at speed V", not what taxonomy it belongs to.
  coarse group     vehicle vs VRU. Two groups whose mapping is defensible, and
                   the distinction that matters downstream, because Phase 3's
                   size prior differs enormously between them.

Recovering Car/Van/Truck separately from a COCO-trained model would require
inventing information, and any per-class mAP so produced would measure the
mapping rather than the detector.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aps.kitti.tracklets import VEHICLE_CLASSES, VRU_CLASSES

# APS coarse groups.
GROUP_VEHICLE, GROUP_VRU = "vehicle", "vru"

# COCO class id -> APS coarse group. Ids are the standard 80-class COCO order.
#
# `motorcycle` maps to VRU rather than vehicle: a motorcyclist is an unprotected
# road user with a VRU's size prior and a VRU's collision consequence, and KITTI
# groups powered two-wheelers with Cyclist. `train` maps to nothing -- KITTI's
# Tram is an ignore class, not a detection target.
COCO_TO_GROUP: dict[int, str] = {
    1: GROUP_VRU,  # bicycle
    3: GROUP_VRU,  # motorcycle
    0: GROUP_VRU,  # person
    2: GROUP_VEHICLE,  # car
    5: GROUP_VEHICLE,  # bus
    7: GROUP_VEHICLE,  # truck
}

# KITTI label -> APS coarse group.
KITTI_TO_GROUP: dict[str, str] = {c: GROUP_VEHICLE for c in VEHICLE_CLASSES} | {
    c: GROUP_VRU for c in VRU_CLASSES
}

# Labelled KITTI objects that are real but not scored. A detection landing on
# one is discarded rather than counted against precision -- the detector found
# something that exists; this evaluation simply does not grade it.
IGNORE_CLASSES = frozenset({"Tram", "Misc", "Person (sitting)"})


@dataclass(frozen=True)
class Detections:
    """One frame's detections, after mapping and class-agnostic NMS."""

    boxes: np.ndarray  # (N,4) [x1,y1,x2,y2] pixels
    scores: np.ndarray  # (N,)
    groups: np.ndarray  # (N,) 'vehicle' | 'vru'
    coco_ids: np.ndarray  # (N,) original COCO class, kept for auditing
    latency_ms: float = float("nan")

    def __len__(self) -> int:
        return len(self.boxes)

    def filter(self, mask: np.ndarray) -> Detections:
        return Detections(
            self.boxes[mask],
            self.scores[mask],
            self.groups[mask],
            self.coco_ids[mask],
            self.latency_ms,
        )

    def by_group(self, group: str) -> Detections:
        return self.filter(self.groups == group)

    def above(self, conf: float) -> Detections:
        return self.filter(self.scores >= conf)


class Detector:
    """YOLOv8 wrapper that emits APS groups rather than COCO classes.

    `conf` is deliberately low (0.05). Average precision integrates the whole
    precision-recall curve, so thresholding early truncates it and reports a
    number that depends on the threshold rather than the detector. The operating
    threshold is chosen later, from the curve, in Phase 7.
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        device: str = "mps",
        conf: float = 0.05,
        imgsz: int = 640,
        nms_iou: float = 0.6,
    ) -> None:
        from ultralytics import YOLO

        self.weights = str(weights)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.nms_iou = nms_iou
        self.model = YOLO(self.weights)
        self._synced = _make_sync(device)

    def warmup(self, image: np.ndarray, iters: int = 5) -> None:
        """Run and discard. Required before any timing.

        First inference pays lazy weight transfer and Metal shader compilation;
        including it in a latency figure measures the compiler.
        """
        for _ in range(iters):
            self.model.predict(image, device=self.device, verbose=False, conf=self.conf)
        self._synced()

    def detect(self, image: np.ndarray, time_it: bool = False) -> Detections:
        """Detect on one BGR frame.

        When `time_it`, the device is synchronised before the clock stops.
        MPS dispatch is asynchronous exactly as CUDA's is, so timing without the
        sync measures the launch rather than the work (CLAUDE.md hard rule 11).
        """
        t0 = time.perf_counter() if time_it else 0.0
        result = self.model.predict(
            image, device=self.device, verbose=False, conf=self.conf, imgsz=self.imgsz
        )[0]
        if time_it:
            self._synced()
            latency = (time.perf_counter() - t0) * 1000.0
        else:
            latency = float("nan")

        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float64)
        scores = result.boxes.conf.cpu().numpy().astype(np.float64)
        coco = result.boxes.cls.cpu().numpy().astype(int)

        keep = np.array([c in COCO_TO_GROUP for c in coco], dtype=bool)
        boxes, scores, coco = boxes[keep], scores[keep], coco[keep]
        groups = np.array([COCO_TO_GROUP[c] for c in coco], dtype=object)

        if len(boxes):
            from aps.matching import nms

            order = nms(boxes, scores, iou_threshold=self.nms_iou)
            boxes, scores, coco, groups = boxes[order], scores[order], coco[order], groups[order]

        return Detections(boxes, scores, groups.astype(str), coco, latency)


def _make_sync(device: str):
    """Return a no-arg callable that blocks until the device is idle."""
    if device == "mps":
        import torch

        return torch.mps.synchronize
    if device == "cuda":
        import torch

        return torch.cuda.synchronize
    return lambda: None
