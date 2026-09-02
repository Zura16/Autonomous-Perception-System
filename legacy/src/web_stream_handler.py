import time
import threading
import cv2

class WebStreamHandler:
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.telemetry_lock = threading.Lock()
        self.obstacle_lock = threading.Lock()
        self.latest_frame = None
        self.latest_telemetry = {
            "action": "MOVE",
            "lane_offset": 0.0,
            "lane_curvature": 0.0,
            "warnings": [],
            "objects": []
        }
        self.is_running = False
        self.is_paused = False
        self.custom_obstacles = []
        self._next_obstacle_id = 900

    def update_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame.copy() if frame is not None else None

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame

    def update_telemetry(self, telemetry):
        with self.telemetry_lock:
            self.latest_telemetry = telemetry.copy() if telemetry is not None else {}

    def get_telemetry(self):
        with self.telemetry_lock:
            return self.latest_telemetry

    def add_custom_obstacle(self, class_name="car", distance=10.0, lateral_pos=0.0, velocity=0.0):
        with self.obstacle_lock:
            obs = {
                "id": self._next_obstacle_id,
                "class_name": class_name,
                "distance": float(distance),
                "lateral_pos": float(lateral_pos),
                "velocity": float(velocity),
                "custom": True
            }
            self._next_obstacle_id += 1
            self.custom_obstacles.append(obs)
            return obs

    def clear_custom_obstacles(self):
        with self.obstacle_lock:
            self.custom_obstacles.clear()

    def get_custom_obstacles(self):
        with self.obstacle_lock:
            return [obs.copy() for obs in self.custom_obstacles]


def generate_mjpeg_stream(stream_handler):
    """
    Generator that continually yields JPEG-encoded frames from the handler 
    formatted as a multipart MJPEG stream.
    """
    while True:
        frame = stream_handler.get_frame()
        if frame is not None:
            # Encode frame to JPEG
            success, jpeg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg_buffer.tobytes() + b'\r\n')
        else:
            # Fallback black frame if stream not started
            black_frame = cv2.imencode('.jpg', cv2.resize(cv2.imread(''), (640, 360)) if False else None)[1] # mock
            # Yielding small delay to prevent CPU spinning
            pass
        time.sleep(0.04) # ~25 FPS stream
