import numpy as np
import config

class MultiObjectTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}  # id -> track dict

    def _calculate_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, detections):
        """
        Updates the tracker with new detections.
        detections: list of dicts from detector.py
        Returns: list of active tracks with updated ID, bbox, distance, velocity, etc.
        """
        # 1. Increment disappeared count for all existing tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["disappeared"] += 1
            self.tracks[track_id]["age"] += 1

        if not detections:
            # Clean up old tracks
            for track_id in list(self.tracks.keys()):
                if self.tracks[track_id]["disappeared"] > config.TRACKER_MAX_AGE:
                    del self.tracks[track_id]
            return self.get_active_tracks()

        # 2. Match detections with existing tracks using IoU
        detection_indices = list(range(len(detections)))
        track_ids = list(self.tracks.keys())

        matched_tracks = set()
        matched_detections = set()

        if track_ids:
            # Create IoU cost matrix (rows = tracks, cols = detections)
            iou_matrix = np.zeros((len(track_ids), len(detections)))
            for t_idx, track_id in enumerate(track_ids):
                for d_idx, det in enumerate(detections):
                    iou_matrix[t_idx, d_idx] = self._calculate_iou(
                        self.tracks[track_id]["bbox"], det["bbox"]
                    )

            # Greedy matching based on highest IoU
            while True:
                # Find maximum IoU value
                max_val_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_val_idx]

                if max_iou < config.TRACKER_IOU_THRESHOLD:
                    break

                t_idx, d_idx = max_val_idx
                track_id = track_ids[t_idx]

                # Match found
                matched_tracks.add(track_id)
                matched_detections.add(d_idx)

                # Update track details
                det = detections[d_idx]
                track = self.tracks[track_id]
                
                # Update velocity calculation
                prev_distance = track["distance"]
                new_distance = det.get("distance")
                
                # Update bounding box, centroid, class, and reset disappeared count
                track["bbox"] = det["bbox"]
                track["centroid"] = self._get_centroid(det["bbox"])
                track["disappeared"] = 0
                track["confidence"] = det["confidence"]
                
                if new_distance is not None:
                    track["distance"] = new_distance
                    if prev_distance is not None:
                        # Simple velocity estimation: distance delta / time delta
                        # Negative velocity means getting closer (approaching us)
                        dt = 1.0 / config.FPS
                        raw_velocity = (new_distance - prev_distance) / dt
                        
                        # Apply exponential moving average to smooth velocity readings
                        if track["velocity"] is None:
                            track["velocity"] = raw_velocity
                        else:
                            track["velocity"] = 0.7 * track["velocity"] + 0.3 * raw_velocity
                
                # Zero out row and column to prevent reuse
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # 3. For unmatched detections, create new tracks
        for d_idx in detection_indices:
            if d_idx not in matched_detections:
                det = detections[d_idx]
                centroid = self._get_centroid(det["bbox"])
                
                self.tracks[self.next_id] = {
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "centroid": centroid,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "disappeared": 0,
                    "age": 1,
                    "distance": det.get("distance"),
                    "velocity": None,
                    "ttc": None
                }
                self.next_id += 1

        # 4. Clean up tracks that have been inactive for too long
        for track_id in list(self.tracks.keys()):
            # If a track disappeared for too long, delete it
            if self.tracks[track_id]["disappeared"] > config.TRACKER_MAX_AGE:
                del self.tracks[track_id]
            # Filter out tracks that are too young and haven't hit enough frames
            elif self.tracks[track_id]["age"] < config.TRACKER_MIN_HITS and self.tracks[track_id]["disappeared"] > 0:
                # Discard temporary noise
                del self.tracks[track_id]

        return self.get_active_tracks()

    def get_active_tracks(self):
        """
        Returns a list of tracks that are active and have met the minimum hit criteria.
        """
        active_tracks = []
        for track_id, track in self.tracks.items():
            # Only return confirmed tracks (age >= min_hits) and currently visible (disappeared == 0)
            if track["age"] >= config.TRACKER_MIN_HITS and track["disappeared"] == 0:
                active_tracks.append(track)
        return active_tracks
