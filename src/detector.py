import cv2
import numpy as np
import config

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class YOLODetector:
    def __init__(self):
        self.model = None
        if HAS_YOLO:
            try:
                print(f"Loading YOLOv8 model: {config.YOLO_MODEL_NAME}...")
                self.model = YOLO(config.YOLO_MODEL_NAME)
                print("YOLOv8 loaded successfully.")
            except Exception as e:
                print(f"Error loading YOLOv8 model: {e}. Falling back to OpenCV CV-based detector.")
                self.model = None
        else:
            print("ultralytics package not found. Falling back to OpenCV CV-based detector.")

    def detect(self, frame, frame_idx=None):
        """
        Detects objects in the frame.
        Returns a list of dicts: [{'bbox': [x1, y1, x2, y2], 'confidence': c, 'class_id': id, 'class_name': name}]
        """
        if self.model is not None:
            return self._detect_yolo(frame)
        else:
            return self._detect_opencv_heuristics(frame, frame_idx)

    def _detect_yolo(self, frame):
        results = self.model.predict(
            source=frame,
            conf=config.DETECTION_CONF_THRESHOLD,
            classes=config.TARGET_CLASSES,
            verbose=False
        )
        
        detections = []
        if len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })
            
        return detections

    def _detect_opencv_heuristics(self, frame, frame_idx):
        """
        OpenCV Color-segmentation based detector.
        Detects the synthetic lead car (red body) and synthetic pedestrian (green shirt, red legs) 
        using color thresholds, acting as a lightweight traditional CV fallback.
        """
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Segment Red Lead Car
        # Red has two HSV ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        
        # Only look in our road region (crop upper sky & far sides to reduce noise)
        roi_mask = np.zeros_like(mask_red)
        # Bounding box limits for road area in synthetic video
        roi_mask[int(config.FRAME_HEIGHT*0.55):, :] = 255
        mask_red = cv2.bitwise_and(mask_red, roi_mask)
        
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 120:  # filter tiny noise spots
                x, y, w, h = cv2.boundingRect(cnt)
                # Expand box slightly to cover whole car (windshield & tires)
                x1 = max(0, x - int(w*0.1))
                y1 = max(0, y - int(h*0.5))
                x2 = min(config.FRAME_WIDTH, x + w + int(w*0.1))
                y2 = min(config.FRAME_HEIGHT, y + h + int(h*0.1))
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95,
                    "class_id": 2, # Car
                    "class_name": "car"
                })
                
        # 2. Segment Pedestrian (Green shirt / torso)
        lower_green = np.array([35, 80, 80])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_green = cv2.bitwise_and(mask_green, roi_mask)
        
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_green:
            area = cv2.contourArea(cnt)
            if 30 < area < 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                # Expand box to cover head and legs
                x1 = max(0, x - w)
                y1 = max(0, y - h)
                x2 = min(config.FRAME_WIDTH, x + 2*w)
                y2 = min(config.FRAME_HEIGHT, y + 2*h)
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.90,
                    "class_id": 0, # Person
                    "class_name": "person"
                })
                
        return detections
