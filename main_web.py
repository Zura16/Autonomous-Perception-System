import os
import time
import threading
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

import config
from src.web_stream_handler import WebStreamHandler, generate_mjpeg_stream
from src.video_stream import VideoStream
from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.lane_detector import LaneDetector
from src.depth_estimator import DepthEstimator
from src.decision_logic import DecisionLogic
import src.utils as utils

app = Flask(__name__, static_folder="web")
CORS(app)

# Thread-safe stream handler
stream_handler = WebStreamHandler()
perception_thread = None

def perception_worker(handler):
    """
    Background worker thread running the real-time perception pipeline.
    """
    print("Background perception thread started.")
    
    # 1. Initialize streams and models
    # We load source parameters dynamically from app config or use default
    video_source = app.config.get("VIDEO_SOURCE", None)
    use_dl_depth = app.config.get("USE_DL_DEPTH", False)
    
    stream = VideoStream(video_source, loop=True) # loop inside server so video never terminates
    detector = YOLODetector()
    tracker = MultiObjectTracker()
    lane_detector = LaneDetector()
    depth_estimator = DepthEstimator(use_dl_depth=use_dl_depth)
    decision_logic = DecisionLogic()

    frame_count = 0
    
    try:
        while handler.is_running:
            success, frame = stream.read()
            if not success or frame is None:
                # Video ended or capture failed - reset stream if loop enabled
                print("Failed to read frame from stream. Re-initializing...")
                stream.release()
                stream = VideoStream(video_source, loop=True)
                continue
                
            frame_count += 1
            
            # --- RUN PERCEPTION PIPELINE ---
            
            # Lane Detection
            left_fit, right_fit, ploty, Minv, curvature, offset = lane_detector.process(frame)
            
            # Object Detection
            raw_detections = detector.detect(frame, frame_idx=frame_count)
            
            # Depth Estimation
            for det in raw_detections:
                distance = depth_estimator.estimate_distance(det["bbox"], det["class_name"])
                det["distance"] = distance

            # Multi-Object Tracking
            tracked_objects = tracker.update(raw_detections)
            
            # Safety Decision Logic
            decision = decision_logic.evaluate(tracked_objects, curvature, offset)
            
            # --- TELEMETRY EXPORT ---
            telemetry_objects = []
            for obj in tracked_objects:
                centroid_x, _ = obj["centroid"]
                distance = obj.get("distance", 0.0)
                
                # Compute physical lateral coordinate in meters relative to camera center
                # X = (centroid_x - optical_center_x) * distance / focal_length
                cx = config.FRAME_WIDTH / 2.0
                lateral_pos = (centroid_x - cx) * distance / config.FOCAL_LENGTH_PX
                
                telemetry_objects.append({
                    "id": obj["id"],
                    "class_name": obj["class_name"],
                    "distance": float(distance),
                    "velocity": float(obj["velocity"]) if obj["velocity"] is not None else 0.0,
                    "lateral_pos": float(lateral_pos),
                    "ttc": float(obj["ttc"]) if obj["ttc"] is not None else None
                })
                
            telemetry = {
                "action": decision["action"],
                "lane_offset": float(offset),
                "lane_curvature": float(curvature),
                "warnings": decision["warnings"],
                "objects": telemetry_objects
            }
            handler.update_telemetry(telemetry)

            # --- RENDER OVERLAYS ---
            annotated_frame = utils.draw_lane_overlay(frame, left_fit, right_fit, ploty, Minv)
            annotated_frame = utils.draw_tracked_objects(annotated_frame, tracked_objects)
            annotated_frame = utils.draw_dashboard(annotated_frame, decision)
            
            # Push frame to web handler
            handler.update_frame(annotated_frame)
            
            # Throttle thread to match video FPS (30 FPS -> ~33ms delay)
            time.sleep(0.03)
            
    except Exception as e:
        print(f"Error in background perception worker: {e}")
    finally:
        print("Releasing background video stream...")
        stream.release()
        handler.update_frame(None)
        handler.is_running = False
        print("Background perception thread ended.")


# --- FLASK ENDPOINTS ---

@app.route("/")
def index():
    """Serves the dashboard frontend page."""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_assets(path):
    """Serves static frontend assets (css, js, images)."""
    return send_from_directory(app.static_folder, path)

@app.route("/video_feed")
def video_feed():
    """MJPEG Live video stream feed."""
    return Response(
        generate_mjpeg_stream(stream_handler),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    """Delivers real-time JSON telemetry of lanes and tracked objects."""
    return jsonify(stream_handler.get_telemetry())

@app.route("/api/status", methods=["GET"])
def get_status():
    """Checks the pipeline thread state."""
    return jsonify({
        "is_running": stream_handler.is_running,
        "video_source": app.config.get("VIDEO_SOURCE", "Default (Synthetic)")
    })

@app.route("/api/control", methods=["POST"])
def post_control():
    """Handles starting and stopping the background pipeline worker thread."""
    global perception_thread
    
    data = request.json or {}
    command = data.get("command", "")
    
    if command == "start":
        if not stream_handler.is_running:
            # Load dynamic configs
            video_source = data.get("video_source", None)
            use_dl_depth = data.get("use_dl_depth", False)
            
            # Map default source
            if video_source == "synthetic" or not video_source:
                app.config["VIDEO_SOURCE"] = None
            else:
                app.config["VIDEO_SOURCE"] = video_source
                
            app.config["USE_DL_DEPTH"] = use_dl_depth
            
            stream_handler.is_running = True
            perception_thread = threading.Thread(
                target=perception_worker, 
                args=(stream_handler,),
                daemon=True
            )
            perception_thread.start()
            return jsonify({"status": "started", "message": "Pipeline thread launched successfully."})
        else:
            return jsonify({"status": "running", "message": "Pipeline already active."})
            
    elif command == "stop":
        if stream_handler.is_running:
            stream_handler.is_running = False
            if perception_thread is not None:
                perception_thread.join(timeout=3.0)
            return jsonify({"status": "stopped", "message": "Pipeline thread terminated."})
        else:
            return jsonify({"status": "idle", "message": "Pipeline not running."})
            
    return jsonify({"error": "Invalid command"}), 400

if __name__ == "__main__":
    # Launch server on port 5000
    print("====================================================")
    print("      LAUNCHING PERCEPTION SYSTEM WEB BACKEND")
    print("      Open http://127.0.0.1:5000 in your browser")
    print("====================================================")
    
    # Automatically start pipeline on start
    stream_handler.is_running = True
    perception_thread = threading.Thread(
        target=perception_worker, 
        args=(stream_handler,),
        daemon=True
    )
    perception_thread.start()
    
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
