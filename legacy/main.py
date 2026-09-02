import os
import cv2
import argparse
import time
import config
from src.video_stream import VideoStream
from src.detector import YOLODetector
from src.tracker import MultiObjectTracker
from src.lane_detector import LaneDetector
from src.depth_estimator import DepthEstimator
from src.decision_logic import DecisionLogic
import src.utils as utils

def run_pipeline(video_source=None, output_path=None, show_gui=True, use_dl_depth=False):
    print("====================================================")
    print("      STARTING AUTONOMOUS PERCEPTION PIPELINE")
    print("====================================================")
    
    # 1. Initialize Video Stream
    stream = VideoStream(video_source, loop=show_gui)
    width, height, fps = stream.get_metadata()
    print(f"Input Stream: Resolution={width}x{height}, FPS={fps}")
    
    # 2. Setup Output Video Writer
    if output_path is None:
        output_path = os.path.join(config.OUTPUT_DIR, "processed_drive.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    print(f"Saving processed output video to: {output_path}")

    # 3. Initialize Modules
    detector = YOLODetector()
    tracker = MultiObjectTracker()
    lane_detector = LaneDetector()
    depth_estimator = DepthEstimator(use_dl_depth=use_dl_depth)
    decision_logic = DecisionLogic()

    frame_count = 0
    start_time = time.time()
    
    gui_active = show_gui
    
    try:
        while True:
            success, frame = stream.read()
            if not success or frame is None:
                print("End of video stream or failed to read frame.")
                break
                
            frame_count += 1
            
            # --- PERCEPTION PIPELINE ---
            
            # Step A: Lane Detection
            # Returns: left_fit, right_fit, ploty, Minv, curvature, offset
            left_fit, right_fit, ploty, Minv, curvature, offset = lane_detector.process(frame)
            
            # Step B: Object Detection
            # Returns raw bounding boxes: [{'bbox', 'confidence', 'class_id', 'class_name'}]
            # Pass frame_count as frame_idx for the traditional CV fallback detector
            raw_detections = detector.detect(frame, frame_idx=frame_count)
            
            # Step C: Depth Estimation (calculate distances for raw detections)
            for det in raw_detections:
                distance = depth_estimator.estimate_distance(det["bbox"], det["class_name"])
                det["distance"] = distance

            # Step D: Multi-Object Tracking & Velocity Estimation
            # Returns tracked active objects: [{'id', 'bbox', 'centroid', 'class_name', 'distance', 'velocity', 'ttc'}]
            tracked_objects = tracker.update(raw_detections)
            
            # Step E: Decision Making
            # Returns safety commands/alerts: {'action', 'lane_offset', 'lane_curvature', 'warnings'}
            decision = decision_logic.evaluate(tracked_objects, curvature, offset)
            
            # --- VISUALIZATION / RENDER LAYER ---
            
            # Overlay detected lane
            annotated_frame = utils.draw_lane_overlay(frame, left_fit, right_fit, ploty, Minv)
            
            # Overlay tracked objects with IDs, distances, velocities and warning boxes
            annotated_frame = utils.draw_tracked_objects(annotated_frame, tracked_objects)
            
            # Overlay telemetry dashboard (system action, lane departure, curvature)
            annotated_frame = utils.draw_dashboard(annotated_frame, decision)
            
            # Write annotated frame to output video file
            writer.write(annotated_frame)
            
            # Print periodic logs (every 60 frames / 2 seconds)
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                warnings_str = ", ".join(decision["warnings"]) if decision["warnings"] else "None"
                print(f"Frame {frame_count:04d} | FPS: {current_fps:.1f} | Action: {decision['action']} | Alerts: {warnings_str}")

            # Show GUI if enabled
            if gui_active:
                cv2.imshow("Autonomous Perception System HUD", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("User terminated pipeline.")
                    break
                    
    except Exception as e:
        print(f"\nPipeline runtime error occurred: {e}")
        # If imshow failed due to no display device (e.g. headless CI environment), continue headlessly
        if "imshow" in str(e) or "display" in str(e):
            print("Display device not found. Switching to headless mode (disabling GUI visualization).")
            gui_active = False
            # Re-run or handle frame loop (actually we just caught it, so let's clean up)
            
    finally:
        # Cleanup
        stream.release()
        writer.release()
        if gui_active:
            cv2.destroyAllWindows()
            
        elapsed_total = time.time() - start_time
        avg_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
        print("\n====================================================")
        print("              PIPELINE EXECUTION COMPLETE")
        print("====================================================")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average Execution Speed: {avg_fps:.1f} FPS")
        print(f"Processed output saved to: {output_path}")
        print("====================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Driving Vision Perception Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (default: synthetic generator)")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")
    parser.add_argument("--no-view", action="store_true", help="Run in headless mode without GUI rendering window")
    parser.add_argument("--use-dl-depth", action="store_true", help="Use deep-learning MiDaS model for depth mapping")
    
    args = parser.parse_args()
    
    # Check if a display exists
    has_display = True
    if args.no_view or ("DISPLAY" not in os.environ and os.uname().sysname == "Linux"):
        has_display = False
        
    run_pipeline(
        video_source=args.video,
        output_path=args.output,
        show_gui=has_display,
        use_dl_depth=args.use_dl_depth
    )
