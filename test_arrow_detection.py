import cv2
import numpy as np
from mss import mss
from template_matcher import TemplateMatcher

def test_arrow_detection():
    # Initialize screen capture
    sct = mss()
    monitor = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}
    
    # Initialize template matcher
    matcher = TemplateMatcher()
    
    print("\nArrow Detection Mode:")
    print("Press 'q' to quit")
    print("Yellow mask window shows what's being matched")
    
    while True:
        # Capture screen
        frame = np.array(sct.grab(monitor))
        debug_frame = frame.copy()
        
        # Get direction
        direction, location, score = matcher.get_direction(frame)
        
        # Show preprocessed frame
        processed_frame = matcher.preprocess_image(frame)
        
        if direction and location:
            # Draw rectangle around match
            h, w = matcher.templates[direction].shape[:2]
            cv2.rectangle(debug_frame, location, (location[0] + w, location[1] + h), (0, 255, 0), 2)
            
            # Draw direction text
            cv2.putText(debug_frame, f"{direction} ({score:.2f})", 
                       (location[0], location[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"\nDetected Direction: {direction} (Score: {score:.2f})")
        else:
            print(f"\nNo direction detected. Best score: {score}")
        
        # Resize frames for display
        display_scale = 0.5
        debug_frame_small = cv2.resize(debug_frame, None, fx=display_scale, fy=display_scale)
        processed_small = cv2.resize(processed_frame, None, fx=display_scale, fy=display_scale)
        
        # Show windows
        cv2.imshow('Original with Detection', debug_frame_small)
        cv2.imshow('Yellow Mask', processed_small)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_arrow_detection()