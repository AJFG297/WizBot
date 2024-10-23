from collections import Counter
import numpy as np
import cv2

class DirectionDetector:
    def __init__(self):
        self.prev_direction = None
        self.direction_counter = 0
        self.DIRECTION_THRESHOLD = 3
        
        # Screen dimensions
        self.SCREEN_WIDTH = 1920
        self.SCREEN_HEIGHT = 1080
        
        # Define quadrant boundaries (with offset)
        self.CENTER_X = self.SCREEN_WIDTH // 2
        self.CENTER_Y = self.SCREEN_HEIGHT // 2
        self.screen_center = (self.CENTER_X, self.CENTER_Y)
        
        # Define dead zone
        self.DEAD_ZONE = 50  # pixels from center
        
        # Thresholds for direction changes
        self.SIDE_THRESHOLD = 100  # pixels from center to consider left/right
        self.TOP_POINTS_COUNT = 3  # number of top points to consider
        self.ANGLE_THRESHOLD = 30  # degrees from vertical to consider left/right
        self.MIN_AREA = 100  # minimum contour area
        
    def get_direction(self, contour):
        # Find the tip of the arrow using convex hull
        hull = cv2.convexHull(contour)
        
        # Get centroid
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return "none", [], (0, 0), (0, 0)
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Find points that form the arrow tip
        tip_points = []
        hull_points = [point[0] for point in hull]
        
        # Find three points near the tip (points with smallest y values)
        hull_points.sort(key=lambda p: p[1])  # Sort by y-coordinate
        tip_points = hull_points[:3]
        
        # Calculate average position of tip points
        if tip_points:
            avg_x = sum(p[0] for p in tip_points) / len(tip_points)
            avg_y = sum(p[1] for p in tip_points) / len(tip_points)
            
            # Calculate angle from vertical
            dx = avg_x - cx
            dy = cy - avg_y  # Inverted because y increases downward
            angle = np.degrees(np.arctan2(dx, dy))
            
            # Determine direction based on angle
            if abs(angle) < self.ANGLE_THRESHOLD:  # within threshold of vertical
                current_direction = 'up'
            elif abs(angle) > 180 - self.ANGLE_THRESHOLD:
                current_direction = 'down'
            else:
                current_direction = 'right' if angle > 0 else 'left'
                
            # Apply direction smoothing
            if current_direction == self.prev_direction:
                self.direction_counter += 1
            else:
                self.direction_counter = 0
            
            # Only change direction after threshold is met
            if self.direction_counter >= self.DIRECTION_THRESHOLD:
                final_direction = current_direction
            else:
                final_direction = self.prev_direction if self.prev_direction else current_direction
                
            self.prev_direction = final_direction
            
            # Debug info
            print(f"Angle: {angle:.1f}, Direction: {final_direction}")
            
            return final_direction, tip_points, (cx, cy), (dx, dy)
        
        return "none", [], (0, 0), (0, 0)

