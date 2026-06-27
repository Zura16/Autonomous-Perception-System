import cv2
import numpy as np
import config

class LaneDetector:
    def __init__(self):
        # Homography matrices
        self.M = cv2.getPerspectiveTransform(config.LANE_SRC_PTS, config.LANE_DST_PTS)
        self.Minv = cv2.getPerspectiveTransform(config.LANE_DST_PTS, config.LANE_SRC_PTS)
        
        # Scaling factors from pixels to meters
        self.ym_per_pix = 30.0 / 720.0  # 30 meters for 720 vertical pixels
        self.xm_per_pix = 3.7 / 680.0   # 3.7m standard lane width for ~680 horizontal pixels
        
        # Keep track of previous fits for smoothing / fallbacks
        self.prev_left_fit = None
        self.prev_right_fit = None

    def preprocess(self, frame):
        """
        Applies HLS color thresholding to isolate white and yellow lane lines.
        """
        # Convert to HLS color space
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        
        # Yellow line thresholding
        mask_yellow = cv2.inRange(hls, config.YELLOW_LANE_MIN, config.YELLOW_LANE_MAX)
        
        # White line thresholding
        mask_white = cv2.inRange(hls, config.WHITE_LANE_MIN, config.WHITE_LANE_MAX)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(mask_yellow, mask_white)
        
        # Apply Region of Interest Mask (in original perspective)
        roi_mask = np.zeros_like(combined_mask)
        cv2.fillPoly(roi_mask, config.LANE_ROI_VERTICES, 255)
        masked_image = cv2.bitwise_and(combined_mask, roi_mask)
        
        return masked_image

    def warp(self, img):
        """
        Warps the image to a bird's-eye view.
        """
        return cv2.warpPerspective(img, self.M, (config.FRAME_WIDTH, config.FRAME_HEIGHT), flags=cv2.INTER_LINEAR)

    def find_lane_pixels(self, warped_binary):
        """
        Uses sliding window search to find left and right lane line pixels.
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped_binary[warped_binary.shape[0]//2:, :], axis=0)
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0]//2)
        # Search within reasonable bounds to avoid borders
        leftx_base = np.argmax(histogram[100:midpoint]) + 100
        rightx_base = np.argmax(histogram[midpoint:config.FRAME_WIDTH-100]) + midpoint

        # Sliding window parameters
        nwindows = 9
        window_height = int(warped_binary.shape[0] // nwindows)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_binary.shape[0] - (window + 1) * window_height
            win_y_high = warped_binary.shape[0] - window * window_height
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (avoid error if empty)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_polynomial(self, warped_binary):
        """
        Fits a second-order polynomial to the detected lane pixels.
        """
        leftx, lefty, rightx, righty = self.find_lane_pixels(warped_binary)
        
        # Fit a second order polynomial to each
        # Left line
        if len(leftx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
            self.prev_left_fit = left_fit
        else:
            # Fallback to previous fit or dummy if none exists
            left_fit = self.prev_left_fit if self.prev_left_fit is not None else np.array([0.0, 0.0, 350.0])
            
        # Right line
        if len(rightx) > 100:
            right_fit = np.polyfit(righty, rightx, 2)
            self.prev_right_fit = right_fit
        else:
            # Fallback
            right_fit = self.prev_right_fit if self.prev_right_fit is not None else np.array([0.0, 0.0, 930.0])
            
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
        
        return left_fit, right_fit, ploty

    def calculate_curvature_and_offset(self, left_fit, right_fit, ploty):
        """
        Computes the road radius of curvature and lateral vehicle offset from the lane center.
        """
        # Define y-value where we want to calculate curvature (bottom of the image)
        y_eval = np.max(ploty)
        
        # Fit polynomials in real world space (meters)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fitx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fitx * self.xm_per_pix, 2)
        
        # Calculate R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Average curvature
        curvature = (left_curverad + right_curverad) / 2.0
        
        # Calculate Vehicle Offset (lateral distance from center of lane)
        # Position of lane center at the bottom of frame (closest to vehicle)
        left_base = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_base = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        
        lane_center = (left_base + right_base) / 2.0
        car_center = config.FRAME_WIDTH / 2.0
        
        # Offset is negative if drifting left, positive if drifting right
        offset = (car_center - lane_center) * self.xm_per_pix
        
        return curvature, offset

    def process(self, frame):
        """
        Main pipeline process: preprocess, warp, fit curves, calculate metrics.
        """
        binary = self.preprocess(frame)
        warped = self.warp(binary)
        
        # Fit curve
        left_fit, right_fit, ploty = self.fit_polynomial(warped)
        
        # Calculate curvature and vehicle offset
        curvature, offset = self.calculate_curvature_and_offset(left_fit, right_fit, ploty)
        
        # Return fits and inverse homography matrix to overlay on original frame
        return left_fit, right_fit, ploty, self.Minv, curvature, offset
