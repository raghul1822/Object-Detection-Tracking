import numpy as np
from core.tracker_sort import SortTracker

class TrackerBridge:
    """
    A unified interface for object tracking.
    Bridges between:
      1. Custom SORT Tracker (using core.tracker_sort)
      2. Ultralytics ByteTrack (using model.track)
      3. Ultralytics BoT-SORT (using model.track)
    """
    def __init__(self, tracker_type='sort', max_age=15, min_hits=3, iou_threshold=0.3):
        self.tracker_type = tracker_type.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Initialize SORT tracker if chosen
        self.sort_tracker = SortTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        
        # Universal trail storage: { track_id: [(x, y), ...] }
        self.trails = {}
        # Max capacity of trail points
        self.max_trail_len = 30
        
        # Track age support
        self.track_first_seen = {}
        self.frame_count = 0

    def update(self, frame, detector, conf_threshold=0.35, iou_threshold=0.45, classes=None):
        """
        Processes a single frame: performs detection & tracking based on tracker type.
        
        Returns:
          A numpy array of shape (N, 7) in the format:
          [[x1, y1, x2, y2, tracker_id, class_id, score], ...]
        """
        self.frame_count += 1
        results_formatted = np.empty((0, 7))
        
        if self.tracker_type == 'sort':
            # Custom SORT Tracking
            # 1. Detect objects using detector wrapper
            dets = detector.detect(
                frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                classes=classes
            )
            # 2. Update custom SORT tracker
            # Input to SORT update: [[x1, y1, x2, y2, score, class_id], ...]
            if len(dets) > 0:
                # Reorder to [x1, y1, x2, y2, score, class_id]
                sort_inputs = dets[:, [0, 1, 2, 3, 4, 5]]
                results_formatted = self.sort_tracker.update(sort_inputs)
            else:
                results_formatted = self.sort_tracker.update(np.empty((0, 6)))
                
        elif self.tracker_type in ['bytetrack', 'botsort']:
            # Ultralytics built-in tracking
            # YOLO tracker configurations: 'bytetrack.yaml' or 'botsort.yaml'
            tracker_cfg = f"{self.tracker_type}.yaml"
            
            # Run model.track on the current frame
            tracks = detector.model.track(
                source=frame,
                persist=True,
                tracker=tracker_cfg,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=classes,
                device=detector.device,
                verbose=False
            )
            
            if len(tracks) > 0 and tracks[0].boxes is not None:
                boxes = tracks[0].boxes
                
                # Check if IDs are assigned (sometimes in empty/early frames they are not)
                if boxes.id is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int)
                    clss = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    
                    # Construct output array of format: [x1, y1, x2, y2, tracker_id, class_id, score]
                    results_formatted = np.zeros((len(ids), 7))
                    results_formatted[:, :4] = xyxy
                    results_formatted[:, 4] = ids
                    results_formatted[:, 5] = clss
                    results_formatted[:, 6] = confs
                    
        # Update universal trails dictionary
        active_ids = set()
        for row in results_formatted:
            x1, y1, x2, y2, track_id, class_id, score = row
            track_id = int(track_id)
            active_ids.add(track_id)
            
            # Record first seen frame
            if track_id not in self.track_first_seen:
                self.track_first_seen[track_id] = self.frame_count
            
            # Calculate center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            if track_id not in self.trails:
                self.trails[track_id] = []
            self.trails[track_id].append((cx, cy))
            
            if len(self.trails[track_id]) > self.max_trail_len:
                self.trails[track_id].pop(0)
                
        # Clean up trails of tracks that are no longer active
        inactive_ids = set(self.trails.keys()) - active_ids
        for inactive_id in inactive_ids:
            del self.trails[inactive_id]
            
        return results_formatted

    def get_trail(self, track_id):
        """
        Returns the history of centers for a given track ID.
        """
        return self.trails.get(int(track_id), [])
        
    def get_track_age(self, track_id):
        """
        Returns the number of frames the track has been active.
        """
        first_frame = self.track_first_seen.get(int(track_id))
        if first_frame is None:
            return 0
        return self.frame_count - first_frame
        
    def reset(self):
        """
        Resets trackers, trails, and age stats.
        """
        self.sort_tracker = SortTracker(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold
        )
        self.trails.clear()
        self.track_first_seen.clear()
        self.frame_count = 0
