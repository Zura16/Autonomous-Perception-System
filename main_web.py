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

def project_custom_obstacle_to_bbox(distance, lateral_pos, class_name="car"):
    """
    Projects physical meters coordinates (distance, lateral_pos) to 2D image bounding box (x1, y1, x2, y2).
    """
    dist = max(float(distance), 1.0)
    lat = float(lateral_pos)
    
    # Camera Intrinsic / Geometry Config
    focal = config.FOCAL_LENGTH_PX
    cam_h = config.CAMERA_HEIGHT_M
    cx = config.FRAME_WIDTH / 2.0
    cy = config.FRAME_HEIGHT / 2.0
    
    # Physical object heights and widths
    if class_name == "person":
        h_obj, w_obj = 1.7, 0.6
    elif class_name == "barrier":
        h_obj, w_obj = 1.0, 1.2
    else: # car / truck
        h_obj, w_obj = 1.5, 1.8
        
    # Projected pixel heights and widths
    h_box = (focal * h_obj) / dist
    w_box = (focal * w_obj) / dist
    
    # Projected bottom y coordinate (horizon is cy)
    y_bottom = cy + (focal * cam_h) / dist
    x_center = cx + (lat * focal) / dist
    
    x1 = int(max(0, min(config.FRAME_WIDTH - 1, x_center - w_box / 2.0)))
    x2 = int(max(1, min(config.FRAME_WIDTH, x_center + w_box / 2.0)))
    y2 = int(max(0, min(config.FRAME_HEIGHT, y_bottom)))
    y1 = int(max(0, min(config.FRAME_HEIGHT - 1, y_bottom - h_box)))
    
    return [x1, y1, x2, y2]


def perception_worker(handler):
    """
    Background worker thread running the real-time perception pipeline.
    """
    print("Background perception thread started.")
    
    video_source = app.config.get("VIDEO_SOURCE", None)
    use_dl_depth = app.config.get("USE_DL_DEPTH", False)
    
    stream = VideoStream(video_source, loop=True)
    detector = YOLODetector()
    tracker = MultiObjectTracker()
    lane_detector = LaneDetector()
    depth_estimator = DepthEstimator(use_dl_depth=use_dl_depth)
    decision_logic = DecisionLogic()

    frame_count = 0
    last_processed_frame = None
    
    try:
        while handler.is_running:
            # Handle Pause Mode
            if handler.is_paused:
                time.sleep(0.05)
                continue

            success, frame = stream.read()
            if not success or frame is None:
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
            
            # --- INJECT CUSTOM STREET OBSTACLES ---
            custom_obs_list = handler.get_custom_obstacles()
            for c_obs in custom_obs_list:
                bbox = project_custom_obstacle_to_bbox(c_obs["distance"], c_obs["lateral_pos"], c_obs["class_name"])
                # Compute Time-To-Collision (TTC) for custom obstacle
                dist = c_obs["distance"]
                vel = c_obs.get("velocity", 0.0)
                ttc = (dist / vel) if vel > 0 else (dist / 1.0 if dist < 8.0 else None)
                
                tracked_objects.append({
                    "id": c_obs["id"],
                    "bbox": bbox,
                    "class_name": c_obs["class_name"],
                    "distance": dist,
                    "velocity": vel,
                    "centroid": ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                    "ttc": ttc,
                    "custom": True
                })
            
            # Safety Decision Logic
            decision = decision_logic.evaluate(tracked_objects, curvature, offset)
            
            # --- TELEMETRY EXPORT ---
            telemetry_objects = []
            for obj in tracked_objects:
                centroid_x, _ = obj["centroid"]
                distance = obj.get("distance", 0.0)
                
                cx = config.FRAME_WIDTH / 2.0
                lateral_pos = (centroid_x - cx) * distance / config.FOCAL_LENGTH_PX if "lateral_pos" not in obj else obj["lateral_pos"]
                
                telemetry_objects.append({
                    "id": obj["id"],
                    "class_name": obj["class_name"],
                    "distance": float(distance),
                    "velocity": float(obj.get("velocity", 0.0) or 0.0),
                    "lateral_pos": float(lateral_pos),
                    "ttc": float(obj["ttc"]) if obj.get("ttc") is not None else None,
                    "custom": obj.get("custom", False)
                })
                
            telemetry = {
                "action": decision["action"],
                "lane_offset": float(offset),
                "lane_curvature": float(curvature),
                "warnings": decision["warnings"],
                "objects": telemetry_objects,
                "is_paused": handler.is_paused
            }
            handler.update_telemetry(telemetry)

            # --- RENDER OVERLAYS ---
            annotated_frame = utils.draw_lane_overlay(frame, left_fit, right_fit, ploty, Minv)
            annotated_frame = utils.draw_tracked_objects(annotated_frame, tracked_objects)
            annotated_frame = utils.draw_dashboard(annotated_frame, decision)
            
            # Push frame to web handler
            handler.update_frame(annotated_frame)
            
            time.sleep(0.03)
            
    except Exception as e:
        print(f"Error in background perception worker: {e}")
    finally:
        print("Releasing background video stream...")
        stream.release()
        handler.update_frame(None)
        handler.is_running = False
        handler.is_paused = False
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
        "is_paused": stream_handler.is_paused,
        "video_source": app.config.get("VIDEO_SOURCE", "Default (Synthetic)"),
        "custom_obstacles": stream_handler.get_custom_obstacles()
    })

@app.route("/api/control", methods=["POST"])
def post_control():
    """Handles play, pause, start, and stop of the background pipeline thread."""
    global perception_thread
    
    data = request.json or {}
    command = data.get("command", "")
    
    if command in ["start", "play"]:
        handler_was_paused = stream_handler.is_paused
        stream_handler.is_paused = False
        
        if not stream_handler.is_running:
            video_source = data.get("video_source", None)
            use_dl_depth = data.get("use_dl_depth", False)
            
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
            return jsonify({"status": "playing", "message": "Pipeline thread started."})
        else:
            return jsonify({"status": "playing", "message": "Pipeline resumed."})

    elif command == "pause":
        if stream_handler.is_running:
            stream_handler.is_paused = True
            return jsonify({"status": "paused", "message": "Pipeline playback paused."})
        else:
            return jsonify({"status": "stopped", "message": "Pipeline is not running."})

    elif command == "stop":
        stream_handler.is_paused = False
        if stream_handler.is_running:
            stream_handler.is_running = False
            if perception_thread is not None:
                perception_thread.join(timeout=3.0)
            return jsonify({"status": "stopped", "message": "Pipeline thread stopped."})
        else:
            return jsonify({"status": "idle", "message": "Pipeline not running."})
            
    return jsonify({"error": "Invalid command"}), 400

@app.route("/api/obstacles", methods=["POST", "DELETE"])
def handle_obstacles():
    """Endpoint to add or clear custom street obstacles."""
    if request.method == "POST":
        data = request.json or {}
        class_name = data.get("class_name", "car")
        distance = float(data.get("distance", 10.0))
        lateral_pos = float(data.get("lateral_pos", 0.0))
        velocity = float(data.get("velocity", 0.0))
        
        obs = stream_handler.add_custom_obstacle(
            class_name=class_name,
            distance=distance,
            lateral_pos=lateral_pos,
            velocity=velocity
        )
        return jsonify({"status": "added", "obstacle": obs})
        
    elif request.method == "DELETE":
        stream_handler.clear_custom_obstacles()
        return jsonify({"status": "cleared", "message": "All custom obstacles cleared."})

if __name__ == "__main__":
    print("====================================================")
    print("      LAUNCHING PERCEPTION SYSTEM WEB BACKEND")
    print("      Open http://127.0.0.1:5000 in your browser")
    print("====================================================")
    
    stream_handler.is_running = True
    perception_thread = threading.Thread(
        target=perception_worker, 
        args=(stream_handler,),
        daemon=True
    )
    perception_thread.start()
    
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
