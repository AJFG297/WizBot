import cv2
import numpy as np
from mss import mss
from template_matcher import TemplateMatcher

def test_template_matching():
    # Initialize screen capture
    sct = mss()
    monitor = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    
    # Initialize template matcher
    matcher = TemplateMatcher()
    
    # Current direction for template capture
    capture_index = 0
    
    print("\nTemplate Capture Mode:")
    print("Press 'c' to capture current frame as template")
    print("Press 'q' to quit")
    print("\nDirections to capture:")
    for i, direction in enumerate(matcher.directions):
        print(f"{i+1}. {direction}")
    
    while True:
        # Capture screen
        frame = np.array(sct.grab(monitor))
        debug_frame = frame.copy()
        
        # Get direction
        direction, location, score = matcher.get_direction(frame)
        
        if direction and location:
            # Draw rectangle around match
            h, w = matcher.templates[direction].shape[:2]
            cv2.rectangle(debug_frame, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)
            
            # Draw direction text
            cv2.putText(debug_frame, f"{direction} ({score:.2f})", 
                       (location[0], location[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"Detected Direction: {direction} (Score: {score:.2f})")
        
        # Draw capture mode info
        if capture_index < len(matcher.directions):
            cv2.putText(debug_frame, f"Capture Mode: {matcher.directions[capture_index]}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Resize frame for display
        display_scale = 0.5
        debug_frame_small = cv2.resize(debug_frame, None, fx=display_scale, fy=display_scale)
        
        # Show window
        cv2.imshow('Template Matching', debug_frame_small)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and capture_index < len(matcher.directions):
            # Capture current frame as template
            direction = matcher.directions[capture_index]
            matcher.capture_template(frame, direction)
            capture_index += 1
            if capture_index < len(matcher.directions):
                print(f"\nReady to capture: {matcher.directions[capture_index]}")
            else:
                print("\nAll templates captured!")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_template_matching()

