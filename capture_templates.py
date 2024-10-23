import cv2
import numpy as np
from mss import mss
from template_matcher import TemplateMatcher

def capture_templates():
    # Initialize screen capture
    sct = mss()
    monitor = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    
    # Initialize template matcher
    matcher = TemplateMatcher()
    
    print("\nTemplate Capture Mode:")
    print("For each direction:")
    print("1. Wait for the arrow to point in that direction")
    print("2. Press 'c' to start capture")
    print("3. Draw a box around the arrow")
    print("4. Press SPACE or ENTER to confirm selection")
    print("\nPress 'q' to quit")
    
    capture_index = 0
    
    while True:
        # Capture screen
        frame = np.array(sct.grab(monitor))
        
        # Show current direction to capture
        if capture_index < len(matcher.directions):
            current_direction = matcher.directions[capture_index]
            print(f"\nReady to capture: {current_direction}")
            print("Press 'c' when ready...")
        
        # Resize frame for display (1600x900)
        display_frame = cv2.resize(frame, (1600, 900))
        
        # Show frame
        cv2.imshow('Capture', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and capture_index < len(matcher.directions):
            # For selection, use same size as display
            selection_scale = 1600/1920  # Scale factor for 1600x900
            
            # Capture template
            matcher.capture_template(frame, matcher.directions[capture_index], selection_scale)
            capture_index += 1
            
            if capture_index >= len(matcher.directions):
                print("\nAll templates captured!")
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_templates()
