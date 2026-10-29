import os
import cv2
import numpy as np
from core.detector import YOLODetector
from core.tracker_bridge import TrackerBridge

def test_pipeline():
    print("Starting pipeline verification test (50 frames)...")
    
    video_path = "data/sample.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Sample video {video_path} does not exist.")
        return False
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: OpenCV could not open sample video.")
        return False
        
    print("Loading YOLO model...")
    detector = YOLODetector(model_name="yolov8n.pt", device="cpu")
    
    print("Initializing SORT tracker...")
    tracker = TrackerBridge(tracker_type="sort", min_hits=1) # Set min_hits=1 for testing immediate tracks!
    
    print("Processing 50 frames...")
    for i in range(50):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Reached end of video early at frame {i}")
            break
            
        # Check raw detections first
        raw_dets = detector.detect(frame, conf_threshold=0.25)
        
        # Track
        tracks = tracker.update(
            frame,
            detector,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        
        if len(raw_dets) > 0 or len(tracks) > 0:
            print(f"Frame {i+1:02d}: Raw Detections = {len(raw_dets)}, Tracked Objects = {len(tracks)}")
            for track in tracks[:3]: # print first 3 tracks
                x1, y1, x2, y2, track_id, class_id, score = track
                print(f"  - Track ID {int(track_id)}: Class {int(class_id)} (conf: {score:.2f})")
            
    cap.release()
    print("Pipeline verification test completed!")
    return True

if __name__ == "__main__":
    test_pipeline()
