import numpy as np
import config

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class DepthEstimator:
    def __init__(self, use_dl_depth=False):
        self.use_dl_depth = use_dl_depth and HAS_TORCH
        self.midas_model = None
        self.midas_transforms = None
        self.device = None
        
        if self.use_dl_depth:
            try:
                print("Loading MiDaS depth estimation model...")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # Load lightweight MiDaS model
                self.midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                self.midas_model.to(self.device)
                self.midas_model.eval()
                
                # Load appropriate transforms
                midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.midas_transforms = midas_transforms.small_transform
                print("MiDaS model loaded successfully.")
            except Exception as e:
                print(f"Error loading MiDaS: {e}. Falling back to geometric distance estimation.")
                self.use_dl_depth = False

    def estimate_distance(self, bbox, class_name):
        """
        Estimates the distance to the object using camera geometry.
        bbox: [x1, y1, x2, y2]
        class_name: string (e.g. 'car', 'person')
        Returns: distance in meters
        """
        x1, y1, x2, y2 = bbox
        pixel_height = max(1.0, y2 - y1)
        pixel_width = max(1.0, x2 - x1)
        
        # Get standard real-world dimensions (default to passenger car if class unknown)
        real_width, real_height, _ = config.OBJECT_DIMENSIONS.get(
            class_name, config.OBJECT_DIMENSIONS["car"]
        )
        
        # Calculate distance based on height: d = (f * H) / h
        distance_height = (config.FOCAL_LENGTH_PX * real_height) / pixel_height
        
        # Calculate distance based on width: d = (f * W) / w
        distance_width = (config.FOCAL_LENGTH_PX * real_width) / pixel_width
        
        # Height is generally more stable than width for vehicles since aspect ratios 
        # change depending on view angle (rear vs. side), but height remains constant.
        # However, for pedestrians, height can also be compressed (bending down, etc.).
        # We'll use height-based distance as our primary metric.
        if class_name == "person":
            # For persons, weight height-based distance heavily
            distance = 0.8 * distance_height + 0.2 * distance_width
        else:
            # For vehicles, height is highly stable
            distance = distance_height
            
        return float(distance)

    def estimate_dl_depth_map(self, frame):
        """
        Runs MiDaS depth estimation on the entire frame.
        Returns: a normalized depth map
        """
        if not self.use_dl_depth or self.midas_model is None:
            return None
            
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.midas_transforms(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Normalize for visualization (0-255)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 255.0
        return depth_map.astype(np.uint8)
