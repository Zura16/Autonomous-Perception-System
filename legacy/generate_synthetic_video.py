import os
import cv2
import numpy as np
import config

def create_synthetic_frame(frame_idx, num_frames):
    # Create empty frame (dark gray asphalt road, blue sky, green scenery)
    frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
    
    # Define vertical regions
    horizon = int(config.FRAME_HEIGHT * 0.6)
    
    # 1. Draw Sky (top portion)
    frame[0:horizon, :] = [210, 160, 100]  # Light blue/sky color (BGR)
    
    # 2. Draw Ground Scenery (bottom portion)
    frame[horizon:, :] = [70, 130, 50]  # Green grass/terrain
    
    # 3. Draw Road (trapezoid from horizon to bottom)
    road_poly = np.array([
        [550, horizon],
        [730, horizon],
        [config.FRAME_WIDTH - 100, config.FRAME_HEIGHT],
        [100, config.FRAME_HEIGHT]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [road_poly], [60, 60, 60])  # Asphalt grey
    
    # 4. Draw lane markers moving downward to simulate vehicle motion
    # Lane curvature simulation (curving right starting at frame 120, peaking at 220)
    curve_offset = 0
    if 100 < frame_idx < 250:
        # Sine wave transition to simulate a curved road segment
        curve_offset = int(60 * np.sin(np.pi * (frame_idx - 100) / 150))
    elif frame_idx >= 250:
        curve_offset = int(60 * np.sin(np.pi * 150 / 150)) # hold curve offset
        
    # Apply curve offset to the horizon road points
    horizon_left = 550 + curve_offset
    horizon_right = 730 + curve_offset
    
    # Re-draw asphalt with curvature
    curved_road = np.array([
        [horizon_left, horizon],
        [horizon_right, horizon],
        [config.FRAME_WIDTH - 100 + curve_offset, config.FRAME_HEIGHT],
        [100 + curve_offset, config.FRAME_HEIGHT]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [curved_road], [60, 60, 60])
    
    # Left Lane Line (Yellow - solid)
    # We will interpolate points from horizon to bottom
    left_pts = []
    right_pts = []
    center_pts = []
    
    # Calculate lane boundaries at different y heights
    for y in range(horizon, config.FRAME_HEIGHT, 10):
        t = (y - horizon) / (config.FRAME_HEIGHT - horizon)
        
        # Horizontal positions
        lx = int((1 - t) * horizon_left + t * (200 + curve_offset))
        rx = int((1 - t) * horizon_right + t * (1080 + curve_offset))
        cx = (lx + rx) // 2
        
        left_pts.append((lx, y))
        right_pts.append((rx, y))
        center_pts.append((cx, y))
        
    # Draw yellow solid left lane
    for i in range(len(left_pts) - 1):
        cv2.line(frame, left_pts[i], left_pts[i+1], [0, 215, 255], 6)
        
    # Draw white dashed right lane (moving dash animation)
    dash_speed = 4
    dash_phase = (frame_idx * dash_speed) % 30
    for i in range(len(right_pts) - 1):
        if (i + dash_phase) % 12 < 6:
            cv2.line(frame, right_pts[i], right_pts[i+1], [255, 255, 255], 6)
            
    # Draw dashed center line (optional divider)
    for i in range(len(center_pts) - 1):
        if (i + dash_phase) % 20 < 10:
            cv2.line(frame, center_pts[i], center_pts[i+1], [200, 200, 200], 2)
            
    # 5. Draw a Lead Car in our lane
    # It starts far away and slows down, getting closer (from distance = 50m to 7m)
    # At frame 220, it accelerates again.
    lead_car_y_target = 0
    if frame_idx < 180:
        # Distance gets closer: 60m down to 9m
        distance = 60.0 - 51.0 * (frame_idx / 180.0)
    elif frame_idx < 230:
        # Stays close (critical braking test): 9m down to 7.5m
        distance = 9.0 - 1.5 * ((frame_idx - 180.0) / 50.0)
    else:
        # Speeds up/moves away: 7.5m back to 30m
        distance = 7.5 + 22.5 * ((frame_idx - 230.0) / 70.0)
        
    # Convert distance to screen size & vertical position (perspective projection)
    # distance = f * real_height / pixel_height
    # pixel_height = f * real_height / distance
    car_real_height = 1.4
    car_real_width = 1.8
    
    car_pixel_height = int(config.FOCAL_LENGTH_PX * car_real_height / distance)
    car_pixel_width = int(config.FOCAL_LENGTH_PX * car_real_width / distance)
    
    # Calculate vehicle position (bottom centered in the lane)
    # Find lane center at projected y
    t_proj = (config.FRAME_HEIGHT - car_pixel_height/2 - horizon) / (config.FRAME_HEIGHT - horizon)
    # constrain t_proj between 0 and 1
    t_proj = max(0.0, min(1.0, t_proj))
    
    lane_center_x = int((1 - t_proj) * (horizon_left + horizon_right)/2 + t_proj * (640 + curve_offset))
    
    car_x1 = lane_center_x - car_pixel_width // 2
    car_y1 = int(horizon + t_proj * (config.FRAME_HEIGHT - horizon) - car_pixel_height)
    car_x2 = car_x1 + car_pixel_width
    car_y2 = car_y1 + car_pixel_height
    
    # Draw lead car box (only if reasonable size)
    if car_pixel_width > 10 and car_y1 > horizon:
        # Body shadow
        cv2.ellipse(frame, (lane_center_x, car_y2), (car_pixel_width//2, car_pixel_height//6), 0, 0, 360, [20, 20, 20], -1)
        # Main body (dark red/gray car back)
        cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), [30, 30, 180], -1)
        # Bumper
        cv2.rectangle(frame, (car_x1, car_y2 - int(car_pixel_height*0.25)), (car_x2, car_y2), [50, 50, 50], -1)
        # License plate
        plate_w = int(car_pixel_width * 0.3)
        plate_h = int(car_pixel_height * 0.12)
        cv2.rectangle(frame, (lane_center_x - plate_w//2, car_y2 - int(car_pixel_height*0.2)),
                      (lane_center_x + plate_w//2, car_y2 - int(car_pixel_height*0.2) + plate_h), [255, 255, 255], -1)
        # Rear windshield
        windshield_h = int(car_pixel_height * 0.35)
        cv2.rectangle(frame, (car_x1 + int(car_pixel_width*0.1), car_y1 + int(car_pixel_height*0.1)),
                      (car_x2 - int(car_pixel_width*0.1), car_y1 + int(car_pixel_height*0.1) + windshield_h), [80, 80, 80], -1)
        # Brake lights
        light_w = int(car_pixel_width * 0.12)
        light_h = int(car_pixel_height * 0.12)
        # If braking (distance decreasing or very close), brake lights are bright red/pink
        is_braking = (frame_idx > 100 and frame_idx < 230)
        light_color = [50, 50, 255] if is_braking else [20, 20, 120]
        
        cv2.rectangle(frame, (car_x1 + int(car_pixel_width*0.05), car_y2 - int(car_pixel_height*0.35)),
                      (car_x1 + int(car_pixel_width*0.05) + light_w, car_y2 - int(car_pixel_height*0.35) + light_h), light_color, -1)
        cv2.rectangle(frame, (car_x2 - int(car_pixel_width*0.05) - light_w, car_y2 - int(car_pixel_height*0.35)),
                      (car_x2 - int(car_pixel_width*0.05), car_y2 - int(car_pixel_height*0.35) + light_h), light_color, -1)

    # 6. Draw a Pedestrian crossing the road from the right shoulder
    # Starts at frame 150 on shoulder, crosses during 160-220, returns/finishes crossing by 260
    if 140 < frame_idx < 270:
        ped_distance = 18.0 # constant distance to simulate pedestrian crossing in front
        ped_real_height = 1.7
        ped_real_width = 0.6
        
        ped_pixel_height = int(config.FOCAL_LENGTH_PX * ped_real_height / ped_distance)
        ped_pixel_width = int(config.FOCAL_LENGTH_PX * ped_real_width / ped_distance)
        
        # Pedestrian movement: walk from right (x=1000) to left lane (x=500)
        # Walk stage: frame 150 -> 240
        t_walk = (frame_idx - 140) / 100.0
        # Walk from right lane line to center/left
        ped_x_start = 850 + curve_offset
        ped_x_end = 450 + curve_offset
        ped_x = int((1 - t_walk) * ped_x_start + t_walk * ped_x_end)
        
        ped_y2 = int(horizon + (ped_distance / 60.0) * (config.FRAME_HEIGHT - horizon))
        ped_y2 = min(config.FRAME_HEIGHT - 10, max(horizon + 20, ped_y2))
        ped_y1 = ped_y2 - ped_pixel_height
        
        # Draw simple stick figure / pedestrian shape
        cv2.ellipse(frame, (ped_x, ped_y2), (ped_pixel_width//2, ped_pixel_height//12), 0, 0, 360, [10, 10, 10], -1) # shadow
        # Legs
        cv2.line(frame, (ped_x - ped_pixel_width//4, ped_y2), (ped_x, ped_y2 - ped_pixel_height//3), [200, 50, 50], 3)
        cv2.line(frame, (ped_x + ped_pixel_width//4, ped_y2), (ped_x, ped_y2 - ped_pixel_height//3), [200, 50, 50], 3)
        # Torso
        cv2.rectangle(frame, (ped_x - ped_pixel_width//3, ped_y1 + ped_pixel_height//3),
                      (ped_x + ped_pixel_width//3, ped_y2 - ped_pixel_height//3), [50, 180, 50], -1)
        # Head
        cv2.circle(frame, (ped_x, ped_y1 + ped_pixel_height//4), ped_pixel_width//3, [220, 200, 150], -1)
        
    return frame

def generate_video():
    video_path = os.path.join(config.DATA_DIR, "driving_sample.mp4")
    print(f"Generating synthetic driving video at: {video_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, config.FPS, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    
    num_frames = config.FPS * 10  # 10 seconds of video
    for i in range(num_frames):
        frame = create_synthetic_frame(i, num_frames)
        out.write(frame)
        
    out.release()
    print("Synthetic video generated successfully!")

if __name__ == "__main__":
    generate_video()
