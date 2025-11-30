import cv2
import numpy as np


class BackgroundSubtraction:
    def __init__(self, threshold=25):
        self.threshold = threshold
        self.background = None
    
    def init_background(self, frame):
        """Initializes the background model with the given frame."""
        self.background = frame.astype("float")
    
    def apply(self, frame):
        """Returns the foreground mask for the given frame."""
        if self.background is None:
            self.init_background(frame)
            return np.zeros(frame.shape[:2], dtype=np.uint8)
        
        bg_uint8 = self.background.astype(np.uint8)
        gr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gr_bg = cv2.cvtColor(bg_uint8, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gr_frame, gr_bg)
        
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def bbox(self, frame):
        """Returns the bounding box of the foreground object in the frame."""
        mask = self.apply(frame)

        y_coords, x_coords = np.where(mask > 0)

        if len(x_coords) < 20:
            return None

        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))


        return (x_min, y_min, x_max, y_max)