import config

class DecisionLogic:
    def __init__(self):
        # Default ego vehicle speed in m/s (approx 54 km/h)
        self.ego_speed = 15.0

    def evaluate(self, tracked_objects, lane_curvature, lane_offset):
        """
        Combines object tracks and lane state to produce warning signals and control actions.
        """
        action = "MOVE"
        warnings = []
        
        # 1. Check Lane Departure
        if abs(lane_offset) > config.LANE_DEPARTURE_THRESHOLD:
            if lane_offset < 0:
                warnings.append("LANE DEPARTURE (LEFT)")
                action = "WARN"
            else:
                warnings.append("LANE DEPARTURE (RIGHT)")
                action = "WARN"

        # 2. Check Object Collision Risks
        for obj in tracked_objects:
            distance = obj.get("distance")
            if distance is None:
                continue
                
            centroid_x, centroid_y = obj["centroid"]
            
            # Compute physical horizontal position of the object in meters relative to our center
            # X = (x_pixel - optical_center_x) * distance / focal_length
            cx = config.FRAME_WIDTH / 2.0
            lateral_pos = (centroid_x - cx) * distance / config.FOCAL_LENGTH_PX
            
            # Check if object is in our path/lane (lateral offset from lane center < half lane width)
            # Adjust lateral position by our vehicle's current lane offset
            relative_lane_pos = lateral_pos - lane_offset
            
            # Standard lane width is 3.7 meters. If object is within ~1.85m of lane center, it is in-lane.
            is_in_lane = abs(relative_lane_pos) < 1.95
            
            # Check for pedestrians close to our lane even if not strictly inside it (margin for safety)
            is_pedestrian_threat = (obj["class_name"] == "person" and abs(relative_lane_pos) < 2.5)
            
            if is_in_lane or is_pedestrian_threat:
                velocity = obj.get("velocity")
                
                # Check for critical proximity regardless of velocity
                if distance < config.CRITICAL_DISTANCE:
                    action = "BRAKE"
                    warnings.append(f"CRITICAL PROXIMITY: {obj['class_name'].upper()} ID {obj['id']}")
                    continue
                
                # If the object is approaching us (relative velocity is negative)
                if velocity is not None and velocity < -0.1:
                    closing_rate = abs(velocity)
                    ttc = distance / closing_rate
                    obj["ttc"] = ttc
                    
                    if ttc < config.CRITICAL_TTC_THRESHOLD:
                        action = "BRAKE"
                        warnings.append(f"CRITICAL TTC ({ttc:.1f}s): {obj['class_name'].upper()} ID {obj['id']}")
                    elif ttc < config.SAFE_TTC_THRESHOLD:
                        if action != "BRAKE":
                            action = "WARN"
                        warnings.append(f"COLLISION RISK ({ttc:.1f}s): {obj['class_name'].upper()} ID {obj['id']}")
                else:
                    # If object is static or moving away, check if we would hit it at our speed if they are stopped
                    # Assume relative speed is our full speed if they are stationary
                    # (Used as a secondary precaution if tracker velocity is not fully converged)
                    if distance < 15.0 and (velocity is None or abs(velocity) < 1.0):
                        ttc_static = distance / self.ego_speed
                        obj["ttc"] = ttc_static
                        if ttc_static < config.SAFE_TTC_THRESHOLD:
                            if action != "BRAKE" and ttc_static < config.CRITICAL_TTC_THRESHOLD:
                                action = "WARN"
                                warnings.append(f"STATIC OBSTACLE ({distance:.1f}m): {obj['class_name'].upper()} ID {obj['id']}")
        
        # Override action to BRAKE if warnings contain critical alerts
        if any("CRITICAL" in w for w in warnings):
            action = "BRAKE"

        return {
            "action": action,
            "lane_offset": lane_offset,
            "lane_curvature": lane_curvature,
            "warnings": warnings
        }
