import cv2
import numpy as np
import pyautogui
import time
from mss import mss
from typing import Tuple, Optional

class WizardBot:
    def __init__(self):
        self.screen_capture = mss()
        self.game_state = 'exploring'
        self.last_arrow_pos = None
        
        # Screen region for the game window
        self.game_window = {
            'left': 0,
            'top': 0,
            'width': 1920,
            'height': 1080
        }
        
        # Initialize movement controls
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1  # Add small delay between actions
        
    def capture_screen(self) -> np.ndarray:
        """Capture the game window and return as numpy array"""
        screenshot = self.screen_capture.grab(self.game_window)
        return np.array(screenshot)
    
    def detect_quest_arrow(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the quest arrow in the given frame
        Returns (x, y) coordinates of the arrow or None if not found
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define gold/yellow color range for quest arrow
        lower_gold = np.array([26, 175, 70])
        upper_gold = np.array([29, 230, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_gold, upper_gold)
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest gold-colored object (likely the quest arrow)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum size threshold
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
        return None
    
    def move_towards(self, target_x: int, target_y: int):
        """Move character towards the target coordinates"""
        # Get screen center
        center_x = self.game_window['width'] // 2
        center_y = self.game_window['height'] // 2
        
        # Calculate direction
        dx = target_x - center_x
        dy = target_y - center_y
        
        # Basic movement logic
        if abs(dx) > 10:  # Add threshold to prevent jitter
            if dx > 0:
                pyautogui.keyDown('d')
                time.sleep(0.1)
                pyautogui.keyUp('d')
            else:
                pyautogui.keyDown('a')
                time.sleep(0.1)
                pyautogui.keyUp('a')
                
        if abs(dy) > 10:
            if dy > 0:
                pyautogui.keyDown('s')
                time.sleep(0.1)
                pyautogui.keyUp('s')
            else:
                pyautogui.keyDown('w')
                time.sleep(0.1)
                pyautogui.keyUp('w')
    
    def run(self):
        """Main bot loop"""
        try:
            while True:
                # Capture current screen
                frame = self.capture_screen()
                
                # Detect quest arrow
                arrow_pos = self.detect_quest_arrow(frame)
                
                if arrow_pos:
                    print(f"Quest arrow detected at: {arrow_pos}")
                    self.move_towards(*arrow_pos)
                    self.last_arrow_pos = arrow_pos
                else:
                    print("No quest arrow detected")
                
                # Add small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Bot stopped by user")
            
if __name__ == "__main__":
    # Create and run bot
    bot = WizardBot()
    bot.run()

