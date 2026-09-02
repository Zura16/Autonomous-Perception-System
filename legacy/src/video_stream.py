import os
import cv2
import config
from generate_synthetic_video import create_synthetic_frame, generate_video

class VideoStream:
    def __init__(self, source_path=None, loop=False):
        self.source_path = source_path
        self.loop = loop
        self.cap = None
        self.is_synthetic = False
        self.frame_idx = 0
        self.total_synthetic_frames = config.FPS * 10 # 10 seconds default

        # Resolve input source
        if source_path is None:
            # Check if synthetic file exists, if not generate it
            synthetic_path = os.path.join(config.DATA_DIR, "driving_sample.mp4")
            if not os.path.exists(synthetic_path):
                generate_video()
            self.source_path = synthetic_path
            
        # Try to open capture stream
        if isinstance(self.source_path, str) and not os.path.exists(self.source_path):
            print(f"Warning: Video file '{self.source_path}' not found. Falling back to real-time synthetic generator.")
            self.is_synthetic = True
        else:
            self.cap = cv2.VideoCapture(self.source_path)
            if not self.cap.isOpened():
                print("Warning: Failed to open video file. Falling back to real-time synthetic generator.")
                self.is_synthetic = True

    def read(self):
        """
        Reads the next frame. Returns (success, frame).
        """
        if self.is_synthetic:
            if self.frame_idx >= self.total_synthetic_frames:
                if not self.loop:
                    return False, None
                # Loop synthetic video
                self.frame_idx = 0
            frame = create_synthetic_frame(self.frame_idx, self.total_synthetic_frames)
            self.frame_idx += 1
            return True, frame
        else:
            success, frame = self.cap.read()
            if not success:
                if not self.loop:
                    return False, None
                # End of video - check if we should loop
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.cap.read()
            
            if success:
                # Resize if necessary to match config standard size
                if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
                    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            return success, frame

    def get_metadata(self):
        """
        Returns (width, height, fps).
        """
        if self.is_synthetic:
            return config.FRAME_WIDTH, config.FRAME_HEIGHT, config.FPS
        else:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
            fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or config.FPS
            return width, height, fps

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
