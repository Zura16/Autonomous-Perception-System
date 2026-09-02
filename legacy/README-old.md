# Autonomous Perception System

A vision-based autonomous vehicle perception pipeline built using **Python, OpenCV, and PyTorch / YOLOv8**.

This system processes camera video inputs to detect driving-relevant objects (vehicles, pedestrians), track them across frames, estimate their lateral and depth distances, track lane lines, and compute safe control actions (such as emergency braking and lane departure warnings).

---

## Key Features

1. **Object Detection**: Leverages YOLOv8 (with a lightweight fallback traditional CV detector using OpenCV color segmentation) to recognize cars, trucks, buses, motorcycles, and pedestrians.
2. **Multi-Object Tracking**: Centroid and Intersection-over-Union (IoU) tracker that assigns stable tracking IDs across frames and handles short-term occlusions.
3. **Monocular Depth / Distance Estimation**: Computes physical obstacle distance (in meters) from camera height projection equations ($d = \frac{f \cdot H}{h}$) and provides a model hook for MiDaS-based depth maps.
4. **Lane Line Detection**: Performs perspective warp (bird's-eye view), HLS color thresholding, sliding window searching, second-order polynomial curve fitting, and offset tracking.
5. **Safety Decision Layer**: Estimates closing speed relative velocities, evaluates Time-to-Collision (TTC) to trigger emergency braking (`BRAKE`), and checks vehicle deviation to trigger Lane Departure Warnings (`WARN`).
6. **Telemetry HUD Overlay**: Draws real-time vehicle tracks, lane overlays, and a dashboard panel summarizing system alerts.
7. **Release Scheduler**: Integrates a staging compiler (`commit_scheduler.py`) to commit and push the codebase in daily increments of approximately 15% to maintain a gradual commit log.

---

## Project Structure

```
Autonomous Perception System/
├── requirements.txt             # Package dependencies
├── .gitignore                   # Files to ignore in Git
├── config.py                    # Configurations & camera matrix calibration
├── generate_synthetic_video.py  # Generates synthetic test video data
├── main.py                      # Main orchestrator pipeline
├── commit_scheduler.py          # Daily commit scheduler
├── commit_state.json            # Scheduler tracking file
├── README.md                    # Project documentation
├── src/                         # Pipeline implementation package
│   ├── __init__.py
│   ├── utils.py                 # HUD and dashboard overlay tools
│   ├── video_stream.py          # Video capture & generator wrapper
│   ├── detector.py              # YOLOv8 / OpenCV color detector
│   ├── tracker.py               # Centroid & IoU track manager
│   ├── lane_detector.py         # OpenCV perspective lane solver
│   ├── depth_estimator.py       # Geometric & MiDaS depth estimator
│   └── decision_logic.py        # TTC and lateral position decision layer
└── tests/                       # Unit tests package
    ├── __init__.py
    └── test_pipeline.py         # Math and logic verification tests
```

---

## Setup & Installation

1. Create a local virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Pipeline

### Step 1: Generate Test Data
If you don't have CARLA or recorded driving videos, generate a realistic synthetic driving sequence (curving lanes, ahead braking vehicle, and crossing pedestrian):
```bash
python generate_synthetic_video.py
```
This writes a sample video file to `data/driving_sample.mp4`.

### Step 2: Start the Perception Engine
Run the main orchestrator (automatically loads the synthetic video if no external source is supplied):
```bash
python main.py
```
*Press `q` on the GUI window to exit.*

### Command Options
- **Run on a custom video file**:
  ```bash
  python main.py --video path/to/your_video.mp4
  ```
- **Run in headless mode** (saves the processed output video to `output/processed_drive.mp4` without showing a display GUI):
  ```bash
  python main.py --no-view
  ```
- **Enable Deep Learning Depth Mapping** (uses PyTorch Hub to run the MiDaS monocular depth model):
  ```bash
  python main.py --use-dl-depth
  ```


## Running Unit Tests

Run the unit test suite to verify math projection coordinate conversions, tracking associations, and collision warning logic:
```bash
python -m unittest tests/test_pipeline.py
```


