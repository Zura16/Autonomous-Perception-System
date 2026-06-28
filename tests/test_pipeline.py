import unittest
import numpy as np
import config
from src.tracker import MultiObjectTracker
from src.lane_detector import LaneDetector
from src.depth_estimator import DepthEstimator
from src.decision_logic import DecisionLogic

class TestPerceptionPipeline(unittest.TestCase):

    def setUp(self):
        self.tracker = MultiObjectTracker()
        self.depth_estimator = DepthEstimator(use_dl_depth=False)
        self.decision_logic = DecisionLogic()

    def test_depth_estimation(self):
        """
        Verify that the geometric depth estimator correctly implements the pinhole camera equations.
        """
        # Set up a mock car bounding box
        # height = 150 pixels, real car height = 1.5m, focal length = 950px
        # d = (950 * 1.5) / 150 = 9.5 meters
        bbox = [100, 100, 250, 250] # width=150, height=150
        distance = self.depth_estimator.estimate_distance(bbox, "car")
        self.assertAlmostEqual(distance, 9.5, places=2)

    def test_tracker_association(self):
        """
        Verify that the tracker correctly associates overlapping boxes across frames.
        """
        # Frame 1 detections
        det_frame1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_name": "car"}
        ]
        
        # Initial update (under min_hits threshold, so track not returned as active yet)
        tracks = self.tracker.update(det_frame1)
        self.assertEqual(len(tracks), 0)
        self.assertEqual(len(self.tracker.tracks), 1)
        track_id = list(self.tracker.tracks.keys())[0]
        self.assertEqual(track_id, 1)

        # Force age update to bypass min_hits (simulate 3 consecutive frames)
        self.tracker.tracks[1]["age"] = config.TRACKER_MIN_HITS
        self.tracker.tracks[1]["disappeared"] = 0

        # Frame 2: overlapping bounding box
        det_frame2 = [
            {"bbox": [105, 102, 205, 202], "confidence": 0.92, "class_name": "car"}
        ]
        tracks = self.tracker.update(det_frame2)
        
        # Verify track ID remains 1 (associated) and track count remains 1
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["id"], 1)
        self.assertEqual(tracks[0]["bbox"], [105, 102, 205, 202])

    def test_decision_logic_lane_departure(self):
        """
        Verify that the decision logic triggers lane departure warnings when crossing thresholds.
        """
        # Case A: Inside lane center
        decision = self.decision_logic.evaluate(
            tracked_objects=[], lane_curvature=2000.0, lane_offset=0.1
        )
        self.assertEqual(decision["action"], "MOVE")
        self.assertEqual(len(decision["warnings"]), 0)

        # Case B: Drifting Left
        decision = self.decision_logic.evaluate(
            tracked_objects=[], lane_curvature=2000.0, lane_offset=-0.6
        )
        self.assertEqual(decision["action"], "WARN")
        self.assertTrue(any("LEFT" in w for w in decision["warnings"]))

        # Case C: Drifting Right
        decision = self.decision_logic.evaluate(
            tracked_objects=[], lane_curvature=2000.0, lane_offset=0.6
        )
        self.assertEqual(decision["action"], "WARN")
        self.assertTrue(any("RIGHT" in w for w in decision["warnings"]))

    def test_decision_logic_collision(self):
        """
        Verify that Time-to-Collision (TTC) evaluations trigger warnings and braking commands.
        """
        # Case A: Ahead vehicle in-lane, far away and moving at same speed (rel velocity = 0)
        # distance = 30m, lateral offset = 0 (in center)
        cx = config.FRAME_WIDTH / 2.0
        mock_tracks = [{
            "id": 1,
            "class_name": "car",
            "centroid": (cx, 500),
            "distance": 30.0,
            "velocity": 0.0,
            "age": 5,
            "disappeared": 0
        }]
        
        decision = self.decision_logic.evaluate(mock_tracks, 5000.0, 0.0)
        self.assertEqual(decision["action"], "MOVE")

        # Case B: Ahead vehicle in-lane, closing in rapidly (distance = 15m, rel velocity = -8m/s)
        # TTC = 15 / 8 = 1.875 seconds -> should trigger Warning (TTC < 2.5s)
        mock_tracks[0]["distance"] = 15.0
        mock_tracks[0]["velocity"] = -8.0
        decision = self.decision_logic.evaluate(mock_tracks, 5000.0, 0.0)
        self.assertEqual(decision["action"], "WARN")
        self.assertTrue(any("COLLISION" in w for w in decision["warnings"]))

        # Case C: Critical Collision Risk (distance = 7m, rel velocity = -8m/s)
        # TTC = 7 / 8 = 0.875 seconds -> should trigger BRAKE (TTC < 1.2s or distance < 8m)
        mock_tracks[0]["distance"] = 7.0
        mock_tracks[0]["velocity"] = -8.0
        decision = self.decision_logic.evaluate(mock_tracks, 5000.0, 0.0)
        self.assertEqual(decision["action"], "BRAKE")
        self.assertTrue(any("CRITICAL" in w for w in decision["warnings"]))

        # Case D: Out of lane vehicle - should not trigger warning
        # centroid placed far to the side (x = cx + 300px)
        # At distance 10m, cx+300px corresponds to 300 * 10 / 950 = 3.15 meters lateral offset (outside lane)
        mock_tracks[0]["centroid"] = (cx + 300, 500)
        mock_tracks[0]["distance"] = 10.0
        mock_tracks[0]["velocity"] = -8.0
        decision = self.decision_logic.evaluate(mock_tracks, 5000.0, 0.0)
        self.assertEqual(decision["action"], "MOVE")

if __name__ == "__main__":
    unittest.main()
