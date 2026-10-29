import argparse
import sys
import cv2
import torch
import numpy as np

from core.detector import YOLODetector
from core.tracker_bridge import TrackerBridge
from core.utils import (
    get_color_for_id,
    get_color_for_class,
    draw_corner_rect,
    draw_glassy_label,
    draw_fading_trail,
    FPSCalculator,
    draw_hud
)

def run_tracking(args):
    # Initialize Detector
    try:
        detector = YOLODetector(model_name=args.model, device=args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check your internet connection (to download model) or model path.")
        sys.exit(1)
        
    class_names = detector.get_class_names()
    
    # Initialize Tracker Bridge
    tracker = TrackerBridge(
        tracker_type=args.tracker,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold
    )
    
    # Open Video Source
    # Check if webcam (usually integer index) or video path
    source = args.source
    if source.isdigit():
        source = int(source)
        print(f"Opening Webcam source: {source}")
    else:
        print(f"Opening Video source: {source}")
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)
        
    # Get video properties for writer
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0 or np.isnan(orig_fps):
        orig_fps = 30.0
        
    # Initialize Video Writer if output path is provided
    out_writer = None
    if args.output:
        print(f"Saving processed video to: {args.output}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, orig_fps, (w, h))
        
    # Parse class filters if provided
    classes_filter = None
    if args.classes:
        classes_filter = [int(c) for c in args.classes.split(',')]
        print(f"Filtering for classes: {[class_names.get(c, c) for c in classes_filter]}")
        
    fps_calc = FPSCalculator()
    total_unique_ids = set()
    show_hud_toggle = True
    show_trails_toggle = True
    paused = False
    
    print("\n--- Controls ---")
    print("  'q' or 'ESC' : Quit")
    print("  'p' or 'SPACE': Pause/Resume")
    print("  't'          : Cycle Tracker (SORT -> ByteTrack -> BoT-SORT)")
    print("  'h'          : Toggle metrics HUD overlay")
    print("  'l'          : Toggle movement trails")
    print("----------------\n")
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break
                
            # Process Frame
            tracks = tracker.update(
                frame,
                detector,
                conf_threshold=args.conf,
                iou_threshold=args.iou_threshold,
                classes=classes_filter
            )
            
            # Update metric counts and track colors/ages
            active_ids_in_frame = []
            track_colors = {}
            track_alphas = {}
            for track in tracks:
                x1, y1, x2, y2, track_id, class_id, score = track
                track_id = int(track_id)
                active_ids_in_frame.append(track_id)
                total_unique_ids.add(track_id)
                
                class_label = class_names.get(class_id, f"ID {class_id}")
                color = get_color_for_class(class_label, track_id=track_id)
                track_colors[track_id] = color
                
                # Fade in opacity based on age
                age = tracker.get_track_age(track_id)
                track_alphas[track_id] = min(1.0, age / 5.0)
                
            # Draw visual overlays
            # 1. Draw trails first (so bounding boxes are drawn on top)
            if show_trails_toggle:
                for track_id in active_ids_in_frame:
                    trail = tracker.get_trail(track_id)
                    color = track_colors.get(track_id, get_color_for_id(track_id))
                    draw_fading_trail(frame, trail, color, max_trail_len=args.max_age * 2)
                    
            # 2. Draw boxes and labels
            for track in tracks:
                x1, y1, x2, y2, track_id, class_id, score = track
                track_id = int(track_id)
                
                color = track_colors.get(track_id, get_color_for_id(track_id))
                alpha = track_alphas.get(track_id, 1.0)
                class_label = class_names.get(class_id, f"ID {class_id}")
                label_text = f"#{track_id} {class_label} {score:.2f}"
                
                # Draw box
                draw_corner_rect(frame, [x1, y1, x2, y2], color, thickness=2, alpha=alpha)
                # Draw label
                draw_glassy_label(frame, label_text, (int(x1) + 2, int(y1) - 4), color, alpha=alpha)
                
            # Update and get FPS
            fps_calc.update()
            fps = fps_calc.get_fps()
            
            # 3. Draw HUD dashboard
            if show_hud_toggle:
                draw_hud(
                    frame,
                    tracker.tracker_type,
                    fps,
                    len(active_ids_in_frame),
                    len(total_unique_ids)
                )
                
            # Save frame if output writer is set
            if out_writer is not None:
                out_writer.write(frame)
                
        # Display window
        cv2.imshow("Real-Time Object Detection & Tracking", frame)
        
        # Capture keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('p') or key == ord(' '):  # 'p' or SPACE
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('t'):  # Cycle tracker type
            current = tracker.tracker_type
            next_tracker = 'bytetrack' if current == 'sort' else ('botsort' if current == 'bytetrack' else 'sort')
            print(f"Switching tracker to: {next_tracker.upper()}")
            # Recreate tracker with new type
            tracker = TrackerBridge(
                tracker_type=next_tracker,
                max_age=args.max_age,
                min_hits=args.min_hits,
                iou_threshold=args.iou_threshold
            )
        elif key == ord('h'):  # Toggle HUD
            show_hud_toggle = not show_hud_toggle
            print("HUD Overlay:", "ENABLED" if show_hud_toggle else "DISABLED")
        elif key == ord('l'):  # Toggle trails
            show_trails_toggle = not show_trails_toggle
            print("Movement Trails:", "ENABLED" if show_trails_toggle else "DISABLED")
            
    # Cleanup
    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing Complete. Total unique objects tracked: {len(total_unique_ids)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI-Powered Real-Time Object Detection & Tracking CLI")
    
    parser.add_argument('--source', type=str, default='0', 
                        help="Video source: '0' for webcam, or path to video file")
    parser.add_argument('--tracker', type=str, default='sort', choices=['sort', 'bytetrack', 'botsort'],
                        help="Object tracking algorithm")
    parser.add_argument('--model', type=str, default='yolov8n.pt', 
                        help="YOLO model checkpoint name (e.g. yolov8n.pt, yolov8s.pt, yolov11n.pt)")
    parser.add_argument('--conf', type=float, default=0.35, 
                        help="Confidence threshold for detections")
    parser.add_argument('--iou-threshold', type=float, default=0.45, 
                        help="Intersection-over-Union threshold for association/NMS")
    parser.add_argument('--classes', type=str, default=None, 
                        help="Comma-separated class IDs to detect/track (e.g. '0,2' for person, car). Default: all")
    
    # Tracker parameters
    parser.add_argument('--max-age', type=int, default=15, 
                        help="SORT: Max frames to keep dead tracks before deletion")
    parser.add_argument('--min-hits', type=int, default=3, 
                        help="SORT: Min hits to establish tracking")
    parser.add_argument('--output', type=str, default=None, 
                        help="Path to save output video file (.mp4)")
    parser.add_argument('--device', type=str, default=None, 
                        help="Device: 'cpu', 'cuda', 'mps'. Default: Auto-detect")
                        
    args = parser.parse_args()
    run_tracking(args)
