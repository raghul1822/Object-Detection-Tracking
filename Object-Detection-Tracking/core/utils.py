import cv2
import numpy as np

# A list of premium, modern BGR colors for object tracks
COLOR_PALETTE = [
    (255, 99, 71),    # Tomato Red
    (30, 144, 255),   # Dodger Blue
    (50, 205, 50),    # Lime Green
    (255, 165, 0),    # Orange
    (138, 43, 226),   # Blue Violet
    (0, 255, 255),    # Cyan
    (255, 20, 147),   # Deep Pink
    (0, 250, 154),    # Medium Spring Green
    (240, 230, 140),  # Khaki
    (255, 105, 180),  # Hot Pink
    (72, 209, 204),   # Medium Turquoise
    (255, 215, 0)     # Gold
]

# Theme colors for specific classes in BGR format
CLASS_COLORS_BGR = {
    "person": (209, 255, 0),       # #00FFD1 -> RGB (0, 255, 209) -> BGR (209, 255, 0)
    "car": (255, 139, 79),          # #4F8BFF -> RGB (79, 139, 255) -> BGR (255, 139, 79)
    "bicycle": (255, 77, 124),      # #7C4DFF -> RGB (124, 77, 255) -> BGR (255, 77, 124)
    "motorcycle": (3, 183, 255),    # #FFB703 -> RGB (255, 183, 3) -> BGR (3, 183, 255)
    "bus": (73, 120, 255),         # #FF7849 -> RGB (255, 120, 73) -> BGR (73, 120, 255)
    "truck": (111, 71, 239),       # #EF476F -> RGB (239, 71, 111) -> BGR (111, 71, 239)
    "dog": (248, 189, 56),         # #38BDF8 -> RGB (56, 189, 248) -> BGR (248, 189, 56)
    "cat": (247, 85, 168),         # #A855F7 -> RGB (168, 85, 247) -> BGR (247, 85, 168)
}

def get_color_for_id(track_id):
    """
    Returns a persistent, visually pleasing color for a given track ID.
    """
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]

def get_color_for_class(class_name, track_id=None):
    """
    Returns the theme color for a class name. Falls back to track ID coloring if class is not in theme palette.
    """
    class_lower = str(class_name).lower()
    if class_lower in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[class_lower]
    
    # Fallback to ID-based hashing if not standard class
    if track_id is not None:
        return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]
    return (255, 255, 255) # white fallback

def draw_corner_rect(img, bbox, color, thickness=2, d=15, alpha=1.0):
    """
    Draws a bounding box with high-tech corner brackets and a faint full rectangle,
    supporting an alpha transparency parameter for fade-in animations.
    """
    if alpha <= 0.0:
        return
        
    x1, y1, x2, y2 = map(int, bbox)
    
    # 1. Draw a faint background rectangle for the box
    rect_overlay = img.copy()
    cv2.rectangle(rect_overlay, (x1, y1), (x2, y2), color, 1)
    cv2.addWeighted(rect_overlay, 0.25 * alpha, img, 1.0 - (0.25 * alpha), 0, img)
    
    # 2. Draw high-tech corner brackets
    corners_overlay = img.copy()
    
    # Top-Left
    cv2.line(corners_overlay, (x1, y1), (x1 + d, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(corners_overlay, (x1, y1), (x1, y1 + d), color, thickness, lineType=cv2.LINE_AA)
    
    # Top-Right
    cv2.line(corners_overlay, (x2, y1), (x2 - d, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(corners_overlay, (x2, y1), (x2, y1 + d), color, thickness, lineType=cv2.LINE_AA)
    
    # Bottom-Left
    cv2.line(corners_overlay, (x1, y2), (x1 + d, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(corners_overlay, (x1, y2), (x1, y2 - d), color, thickness, lineType=cv2.LINE_AA)
    
    # Bottom-Right
    cv2.line(corners_overlay, (x2, y2), (x2 - d, y2), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(corners_overlay, (x2, y2), (x2, y2 - d), color, thickness, lineType=cv2.LINE_AA)
    
    cv2.addWeighted(corners_overlay, alpha, img, 1.0 - alpha, 0, img)

def draw_glassy_label(img, text, position, color, text_color=(255, 255, 255), font_scale=0.45, alpha=1.0):
    """
    Draws a text label with a rounded, semi-transparent background (glassmorphism effect),
    supporting an alpha transparency parameter.
    """
    if alpha <= 0.0:
        return
        
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Define label coordinates
    pad_x, pad_y = 6, 6
    x1, y1 = x - pad_x, y - text_h - pad_y - baseline
    x2, y2 = x + text_w + pad_x, y + baseline
    
    # Draw glass shadow/border
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1) # Dark fill
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA) # Color border
    
    # Blend overlay with original
    cv2.addWeighted(overlay, 0.65 * alpha, img, 1.0 - (0.65 * alpha), 0, img)
    
    # Write text on top of blended image
    text_overlay = img.copy()
    cv2.putText(text_overlay, text, (x, y - baseline), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(text_overlay, alpha, img, 1.0 - alpha, 0, img)

def draw_fading_trail(img, trail, color, max_trail_len=30):
    """
    Draws a fading trail behind tracked objects, fading from 20% opacity at the oldest point
    to 100% opacity at the newest point.
    """
    if len(trail) < 2:
        return
        
    for i in range(1, len(trail)):
        pt1 = trail[i - 1]
        pt2 = trail[i]
        
        # Calculate fading opacity from 20% to 100%
        if len(trail) > 2:
            alpha = 0.2 + 0.8 * (float(i - 1) / (len(trail) - 2))
        else:
            alpha = 1.0
            
        thickness = int(max(1, int(4 * (float(i) / len(trail)))))
        
        # Draw line segment
        overlay = img.copy()
        cv2.line(overlay, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.7, img, 1.0 - (alpha * 0.7), 0, img)

class FPSCalculator:
    """
    Helper class to measure and compute running FPS.
    """
    def __init__(self, buffer_len=30):
        self.buffer_len = buffer_len
        self.times = []
        
    def update(self):
        import time
        current_time = time.time()
        self.times.append(current_time)
        if len(self.times) > self.buffer_len:
            self.times.pop(0)
            
    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        delta = self.times[-1] - self.times[0]
        if delta == 0:
            return 0.0
        return (len(self.times) - 1) / delta

def draw_hud(img, tracker_type, fps, active_count, total_count, classes_names_str="All"):
    """
    Draws a clean dashboard HUD overlay at the top left showing key metrics.
    """
    h, w, _ = img.shape
    hud_w, hud_h = 240, 120
    
    # Draw faint panel box
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (15, 15, 15), -1)
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (100, 100, 100), 1, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (240, 240, 240)
    
    # Text positions inside HUD
    cv2.putText(img, f"SYSTEM MONITOR", (20, 28), font, 0.45, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.line(img, (20, 35), (20 + hud_w - 20, 35), (100, 100, 100), 1)
    
    cv2.putText(img, f"Tracker: {tracker_type.upper()}", (20, 52), font, font_scale, color, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 70), font, font_scale, color, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Active Targets: {active_count}", (20, 88), font, font_scale, color, 1, lineType=cv2.LINE_AA)
    cv2.putText(img, f"Total Detections: {total_count}", (20, 106), font, font_scale, color, 1, lineType=cv2.LINE_AA)
