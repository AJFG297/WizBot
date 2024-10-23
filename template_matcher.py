import cv2
import numpy as np
from pathlib import Path

class TemplateMatcher:
    def __init__(self):
        self.templates = {}
        self.directions = [
            'up', 'up_right', 'right', 'down_right',
            'down', 'down_left', 'left', 'up_left'
        ]
        self.threshold = 0.7
        
        # HSV range for yellow arrow
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])
        
        # Load templates after initializing all attributes
        self.load_templates()
    
    def preprocess_image(self, image):
        """Extract yellow arrow from image"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for yellow color
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def load_templates(self):
        """Load arrow templates from the templates directory"""
        template_dir = Path('templates')
        if not template_dir.exists():
            template_dir.mkdir(exist_ok=True)
            print("Created templates directory. Please add template images for:")
            for direction in self.directions:
                print(f"- {direction}.png")
            return
            
        # Load each template
        for direction in self.directions:
            path = template_dir / f"{direction}.png"
            if path.exists():
                template = cv2.imread(str(path))
                if template is not None:
                    # Preprocess template to get just the yellow arrow
                    template = self.preprocess_image(template)
                    self.templates[direction] = template
                    print(f"Loaded template for direction: {direction}")
            else:
                print(f"Missing template for {direction}")
    
    def capture_template(self, frame, direction, scale=0.5):
        """Save current frame as a template"""
        if direction not in self.directions:
            print(f"Invalid direction. Must be one of: {self.directions}")
            return
            
        # Create window for region selection
        selection_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.namedWindow('Select Arrow Region')
        roi = cv2.selectROI('Select Arrow Region', selection_frame)
        cv2.destroyWindow('Select Arrow Region')
        
        # Scale ROI back to original size
        x = int(roi[0] / scale)
        y = int(roi[1] / scale)
        w = int(roi[2] / scale)
        h = int(roi[3] / scale)
        
        # Extract region
        template = frame[y:y+h, x:x+w]
        
        if template.size == 0:
            print("Invalid region selected")
            return
            
        template_dir = Path('templates')
        template_dir.mkdir(exist_ok=True)
        
        path = template_dir / f"{direction}.png"
        cv2.imwrite(str(path), template)
        print(f"Saved template for {direction}")
        
        # Reload templates
        self.load_templates()
    
    def get_direction(self, frame):
        """Find the best matching direction in the frame"""
        # Preprocess frame to get just the yellow arrow
        frame_processed = self.preprocess_image(frame)
        
        best_match = None
        best_score = -1
        best_loc = None
        
        # Try each template
        for direction, template in self.templates.items():
            try:
                # Make sure both images are the same type (8-bit single channel)
                if len(template.shape) > 2:  # If template has channels
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                if len(frame_processed.shape) > 2:  # If frame has channels
                    frame_processed = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
                
                # Apply template matching
                result = cv2.matchTemplate(frame_processed, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = direction
                    best_loc = max_loc
                    
                print(f"{direction}: {max_val:.3f}")  # Debug print
                
            except cv2.error as e:
                print(f"Error matching template {direction}: {e}")
                print(f"Frame shape: {frame_processed.shape}, Template shape: {template.shape}")
                continue
        
        # If match is good enough, return the direction
        if best_score >= self.threshold:
            return best_match, best_loc, best_score
        
        return None, None, best_score