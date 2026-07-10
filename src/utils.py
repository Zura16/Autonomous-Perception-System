import cv2
import numpy as np
import config

def draw_hud_corners(img, bbox, color, length=12, thickness=2):
    """Draws tech-style HUD corners around a bounding box instead of a full rectangle."""
    x1, y1, x2, y2 = map(int, bbox)
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_tracked_objects(frame, objects):
    """
    Draws tracked targets with transparent status badges and high-tech corner brackets.
    """
    # Create semi-transparent overlay layer for labels
    overlay = frame.copy()
    
    for obj in objects:
        bbox = obj.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        obj_id = obj.get("id")
        class_name = obj.get("class_name", "object")
        distance = obj.get("distance")
        velocity = obj.get("velocity", 0.0)
        ttc = obj.get("ttc")

        # Select color based on threat level
        color = config.COLOR_GREEN
        warning_text = ""
        
        if ttc is not None:
            if ttc < config.CRITICAL_TTC_THRESHOLD or (distance is not None and distance < config.CRITICAL_DISTANCE):
                color = config.COLOR_RED
                warning_text = "COLLISION RISK!"
            elif ttc < config.SAFE_TTC_THRESHOLD:
                color = config.COLOR_YELLOW
                warning_text = "CLOSE APPROACH"
        elif distance is not None and distance < config.CRITICAL_DISTANCE:
            color = config.COLOR_RED
            warning_text = "CRITICAL PROXIMITY"
                
        # Draw high-tech HUD corner brackets
        draw_hud_corners(frame, bbox, color, length=12, thickness=2)
        
        # Soft transparent bbox fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        # Compile info text
        label = f"[ID {obj_id}] {class_name.upper()}"
        if distance is not None:
            label += f" | {distance:.1f}m"
        if velocity is not None and abs(velocity) > 0.1:
            label += f" | {velocity:+.1f}m/s"

        # Text size calculation
        (w, h), _ = cv2.getTextSize(label, config.FONT, 0.35, 1)
        
        # Draw label background box
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 12, y1), (15, 15, 15), -1)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 12, y1), color, 1)
        cv2.putText(frame, label, (x1 + 6, y1 - 5), config.FONT, 0.35, config.COLOR_WHITE, 1, cv2.LINE_AA)
        
        # Draw Warning Tag above box
        if warning_text:
            (ww, wh), _ = cv2.getTextSize(warning_text, config.FONT, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - h - 25 - wh), (x1 + ww + 12, y1 - h - 10), (15, 15, 220) if color == config.COLOR_RED else (15, 180, 180), -1)
            cv2.putText(frame, warning_text, (x1 + 6, y1 - h - 17), config.FONT, 0.35, config.COLOR_WHITE, 1, cv2.LINE_AA)

    # Blend soft filled boxes slightly
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    return frame

def draw_lane_overlay(frame, left_fit, right_fit, ploty, Minv):
    """
    Fills the detected lane region with a neon-cyan cyber flow and unwarps it.
    """
    if left_fit is None or right_fit is None or ploty is None:
        return frame
        
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    warp_zero = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right)).astype(np.int32)
    
    # Glowing neon-cyan lane interior (BGR: 180, 120, 0)
    cv2.fillPoly(warp_zero, [pts], (180, 120, 0))
    
    # Draw left boundary line (neon gold/amber BGR: 20, 105, 139 to match portfolio site)
    pts_left_line = pts_left.astype(np.int32)
    cv2.polylines(warp_zero, pts_left_line, False, (20, 105, 139), 8)
    
    # Draw right boundary line (bright white/cyan BGR: 240, 240, 240)
    pts_right_line = np.array([np.transpose(np.vstack([right_fitx, ploty]))]).astype(np.int32)
    cv2.polylines(warp_zero, pts_right_line, False, (240, 240, 240), 8)
    
    # Warp back to original perspective
    newwarp = cv2.warpPerspective(warp_zero, Minv, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    
    # Alpha blend for a glowing cyber-lane effect
    result = cv2.addWeighted(frame, 1.0, newwarp, 0.35, 0)
    return result

def draw_dashboard(frame, decision):
    """
    Overlays a telemetry stats dashboard matching the gold portfolio aesthetic.
    """
    overlay = frame.copy()
    
    # Draw dashboard box (Glassmorphic dark styling)
    cv2.rectangle(overlay, (20, 20), (450, 185), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    
    # Gold border matching aalind.org accent colors (BGR: 20, 105, 139)
    cv2.rectangle(frame, (20, 20), (450, 185), (20, 105, 139), 1)
    
    x, y_start, spacing = 35, 45, 22
    
    # Title
    cv2.putText(frame, "AUTONOMOUS VISION HUD TELEMETRY", (x, y_start), config.FONT, 0.45, (20, 105, 139), 1, cv2.LINE_AA)
    
    action = decision.get("action", "MOVE")
    offset = decision.get("lane_offset", 0.0)
    curvature = decision.get("lane_curvature", 0.0)
    warnings = decision.get("warnings", [])
    
    # Action state colors
    action_color = config.COLOR_GREEN
    if action == "BRAKE":
        action_color = config.COLOR_RED
    elif action == "WARN":
        action_color = config.COLOR_YELLOW
        
    cv2.putText(frame, f"Ego Command   : {action}", (x, y_start + spacing), config.FONT, 0.40, action_color, 1, cv2.LINE_AA)
    
    curve_txt = "Straight" if curvature > 3000 else f"{curvature:.1f} m"
    cv2.putText(frame, f"Road Curvature: {curve_txt}", (x, y_start + 2*spacing), config.FONT, 0.40, config.COLOR_WHITE, 1, cv2.LINE_AA)
    
    offset_sign = "+" if offset >= 0 else ""
    cv2.putText(frame, f"Lateral Offset: {offset_sign}{offset:.2f} m", (x, y_start + 3*spacing), config.FONT, 0.40, config.COLOR_WHITE, 1, cv2.LINE_AA)
    
    # Active threats ledger
    warn_y = y_start + 4*spacing
    cv2.putText(frame, "Threat Alerts :", (x, warn_y), config.FONT, 0.40, config.COLOR_WHITE, 1, cv2.LINE_AA)
    if not warnings:
        cv2.putText(frame, "ROAD CLEAR", (x + 130, warn_y), config.FONT, 0.40, config.COLOR_GREEN, 1, cv2.LINE_AA)
    else:
        warn_txt = ", ".join(warnings[:2])
        cv2.putText(frame, warn_txt, (x + 130, warn_y), config.FONT, 0.40, config.COLOR_RED, 1, cv2.LINE_AA)
        
    # Draw futuristic targeting reticle crosshair in the center
    cx, cy = config.FRAME_WIDTH // 2, config.FRAME_HEIGHT // 2
    # Draw outer reticle ring
    cv2.circle(frame, (cx, cy), 18, (0, 200, 0), 1, cv2.LINE_AA)
    # Draw tick coordinates
    cv2.line(frame, (cx - 6, cy), (cx + 6, cy), (0, 200, 0), 1)
    cv2.line(frame, (cx, cy - 6), (cx, cy + 6), (0, 200, 0), 1)
    
    return frame
