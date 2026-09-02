import os
import numpy as np

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Camera intrinsic calibration parameters (mock dashboard camera)
# K: Camera matrix
CAMERA_MATRIX = np.array([
    [950.0, 0.0, FRAME_WIDTH / 2.0],
    [0.0, 950.0, FRAME_HEIGHT / 2.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Distortion coefficients
DISTORTION_COEFFS = np.zeros((5, 1), dtype=np.float32)

# Focal length in pixels (for depth estimation helper)
FOCAL_LENGTH_PX = 950.0
# Approximate real dimensions of common object classes (in meters)
# [width, height, depth]
OBJECT_DIMENSIONS = {
    "car": (1.8, 1.5, 4.2),        # typical sedan
    "truck": (2.5, 3.2, 10.0),     # commercial truck
    "bus": (2.6, 3.2, 12.0),       # city bus
    "motorcycle": (0.8, 1.0, 2.0),  # motorcycle
    "person": (0.6, 1.7, 0.3),     # human
}

# Lane Detection configuration
LANE_ROI_VERTICES = np.array([
    [(200, 680), (550, 460), (730, 460), (1150, 680)]
], dtype=np.int32)

# Source and destination points for perspective warp (Bird's eye view)
LANE_SRC_PTS = np.float32([
    [580, 460],   # Top left
    [700, 460],   # Top right
    [1050, 680],  # Bottom right
    [270, 680]    # Bottom left
])

LANE_DST_PTS = np.float32([
    [300, 0],            # Top left
    [980, 0],            # Top right
    [980, FRAME_HEIGHT], # Bottom right
    [300, FRAME_HEIGHT]  # Bottom left
])

# Color thresholding (HLS bounds for yellow & white lane lines)
YELLOW_LANE_MIN = np.array([15, 38, 115], dtype=np.uint8)
YELLOW_LANE_MAX = np.array([35, 204, 255], dtype=np.uint8)

WHITE_LANE_MIN = np.array([0, 200, 0], dtype=np.uint8)
WHITE_LANE_MAX = np.array([180, 255, 255], dtype=np.uint8)

# Object Detection configuration (YOLOv8)
YOLO_MODEL_NAME = "yolov8n.pt"  # Lightweight model, auto-downloaded
DETECTION_CONF_THRESHOLD = 0.35
# COCO classes to detect: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

# Multi-Object Tracking configuration
TRACKER_MAX_AGE = 15      # Frames to keep track active without update
TRACKER_MIN_HITS = 3      # Consecutive frames needed to confirm track
TRACKER_IOU_THRESHOLD = 0.25

# Decision Layer configuration
SAFE_TTC_THRESHOLD = 2.5       # Seconds before collision warning
CRITICAL_TTC_THRESHOLD = 1.2   # Seconds before emergency brake command
CRITICAL_DISTANCE = 8.0        # Meters (hard minimum safe distance)
LANE_DEPARTURE_THRESHOLD = 0.5  # Meters vehicle can deviate from center

# Graphics and Visualization
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
