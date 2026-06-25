import cv2
import numpy as np
import config

def draw_tracked_objects(frame, objects):
    """
    Draws tracked bounding boxes, IDs, estimated distance, relative speed, and warnings.
    """
    for obj in objects:
        bbox = obj.get("bbox")  # (x1, y1, x2, y2)
        if bbox is None:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        obj_id = obj.get("id")
        class_name = obj.get("class_name", "object")
        distance = obj.get("distance")
        velocity = obj.get("velocity", 0.0)
        ttc = obj.get("ttc")

        # Select color based on warning state
        color = config.COLOR_GREEN
        warning_text = ""
        
        if ttc is not None:
            if ttc < config.CRITICAL_TTC_THRESHOLD or (distance is not None and distance < config.CRITICAL_DISTANCE):
                color = config.COLOR_RED
                warning_text = "CRITICAL BRAKE!"
            elif ttc < config.SAFE_TTC_THRESHOLD:
                color = config.COLOR_YELLOW
                warning_text = "COLLISION WARNING"
                
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"ID {obj_id} | {class_name.capitalize()}"
        if distance is not None:
            label += f" | {distance:.1f}m"
        if velocity is not None and abs(velocity) > 0.1:
            label += f" | {velocity:+.1f}m/s"
            
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + len(label) * 8 + 10, y1), color, -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 7), config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
        
        # Draw warning alert above the box
        if warning_text:
            cv2.rectangle(frame, (x1, y1 - 44), (x1 + len(warning_text) * 8 + 10, y1 - 22), config.COLOR_RED, -1)
            cv2.putText(frame, warning_text, (x1 + 5, y1 - 29), config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
            
    return frame

def draw_lane_overlay(frame, left_fit, right_fit, ploty, Minv):
    """
    Fills the detected lane region on a bird's-eye view, unwarps it, and blends with the original frame.
    """
    if left_fit is None or right_fit is None or ploty is None:
        return frame
        
    # Generate points for left and right curves
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right)).astype(np.int32)
    
    # Draw the lane onto the warped blank image (green lane fill)
    cv2.fillPoly(warp_zero, [pts], (0, 100, 0))
    
    # Draw outer lane lines (yellow left, white right)
    pts_left_line = pts_left.astype(np.int32)
    pts_right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int32)
    cv2.polylines(warp_zero, pts_left_line, False, (0, 215, 255), 15)
    cv2.polylines(warp_zero, pts_right_line, False, (255, 255, 255), 15)
    
    # Warp back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, Minv, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    
    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1.0, newwarp, 0.3, 0)
    return result

def draw_dashboard(frame, decision):
    """
    Overlays a neat semi-transparent dashboard showing telemetry, alerts, and commands.
    """
    overlay = frame.copy()
    
    # Draw background panel at the top
    cv2.rectangle(overlay, (20, 20), (450, 190), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Dashboard border
    cv2.rectangle(frame, (20, 20), (450, 190), (100, 100, 100), 1)
    
    # Text positions
    x, y_start, spacing = 35, 45, 22
    
    # Title
    cv2.putText(frame, "PERCEPTION SYSTEM TELEMETRY", (x, y_start), config.FONT, 0.55, config.COLOR_YELLOW, 2, cv2.LINE_AA)
    
    # Extract decision stats
    action = decision.get("action", "MOVE")
    offset = decision.get("lane_offset", 0.0)
    curvature = decision.get("lane_curvature", 0.0)
    warnings = decision.get("warnings", [])
    
    # Color action text appropriately
    action_color = config.COLOR_GREEN
    if action == "BRAKE":
        action_color = config.COLOR_RED
    elif action == "DESTRUCTIVE" or "COLLISION" in "".join(warnings):
        action_color = config.COLOR_YELLOW
        
    # Overlay metrics
    cv2.putText(frame, f"System Action  : {action}", (x, y_start + spacing), config.FONT, 0.45, action_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lane Curvature : {curvature:.1f} m", (x, y_start + 2*spacing), config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Offset fr Center: {offset:+.2f} m", (x, y_start + 3*spacing), config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
    
    # Warnings Display
    warning_y = y_start + 4*spacing
    cv2.putText(frame, "Alerts/Warnings:", (x, warning_y), config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)
    
    if not warnings:
        cv2.putText(frame, "None - Road Clear", (x + 130, warning_y), config.FONT, 0.45, config.COLOR_GREEN, 1, cv2.LINE_AA)
    else:
        warn_str = ", ".join(warnings[:2])  # display top 2 warnings
        cv2.putText(frame, warn_str, (x + 130, warning_y), config.FONT, 0.45, config.COLOR_RED, 1, cv2.LINE_AA)
        
    # Draw Target HUD at the center of the frame
    # Pitch line / Center crosshair
    cx, cy = config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), config.COLOR_GREEN, 1)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), config.COLOR_GREEN, 1)
    
    return frame
