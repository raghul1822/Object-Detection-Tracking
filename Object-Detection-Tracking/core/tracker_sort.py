import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    bb_test: [N, 4] (predictions) - [[x1, y1, x2, y2], ...]
    bb_gt: [M, 4] (detections) - [[x1, y1, x2, y2], ...]
    Returns: [N, M] matrix of IoUs
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


class KalmanFilter2D:
    """
    A simple, self-contained 2D Kalman Filter tracking the state vector:
    x = [u, v, s, r, u_dot, v_dot, s_dot]^T
    where (u, v) is center, s is scale (area), r is aspect ratio.
    """
    def __init__(self, bbox):
        """
        Initialize the state from a bounding box [x1, y1, x2, y2].
        """
        # State vector: [u, v, s, r, u_dot, v_dot, s_dot]
        self.x = np.zeros((7, 1))
        
        # Convert bounding box to state format
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        u = bbox[0] + w / 2.
        v = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 0.
        
        self.x[0, 0] = u
        self.x[1, 0] = v
        self.x[2, 0] = s
        self.x[3, 0] = r
        
        # State covariance matrix P
        self.P = np.diag([10., 10., 10., 10., 1000., 1000., 1000.])
        
        # State transition matrix F
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise covariance matrix R
        self.R = np.diag([1., 1., 10., 10.])
        
        # Process noise covariance matrix Q
        self.Q = np.diag([1., 1., 1., 1., 0.01, 0.01, 0.0001])

    def predict(self):
        """
        Predict state to next time step.
        """
        # x' = F * x
        self.x = np.dot(self.F, self.x)
        # P' = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_state()

    def update(self, bbox):
        """
        Update state with a new measurement [x1, y1, x2, y2].
        """
        # Convert bounding box to measurement z
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        u = bbox[0] + w / 2.
        v = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 0.
        z = np.array([[u], [v], [s], [r]])
        
        # Innovation y = z - H * x
        y = z - np.dot(self.H, self.x)
        
        # Innovation covariance S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state x = x + K * y
        self.x = self.x + np.dot(K, y)
        
        # Update covariance P = (I - K * H) * P
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """
        Convert internal state back to bounding box [x1, y1, x2, y2].
        """
        u = self.x[0, 0]
        v = self.x[1, 0]
        s = self.x[2, 0]
        r = self.x[3, 0]
        
        if np.isnan(s) or np.isnan(r) or s <= 0 or r <= 0:
            return np.array([0., 0., 0., 0.])
            
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0.
        
        x1 = u - w / 2.
        y1 = v - h / 2.
        x2 = u + w / 2.
        y2 = v + h / 2.
        
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            return np.array([0., 0., 0., 0.])
            
        return np.array([x1, y1, x2, y2])


class KalmanBoxTracker:
    """
    Represents the state of a single tracked object using a Kalman Filter.
    """
    count = 0
    
    def __init__(self, bbox, cls_id=0, conf=0.0):
        """
        Initialises a tracker using initial bounding box.
        """
        self.kf = KalmanFilter2D(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = int(cls_id)
        self.conf = float(conf)
        # Store tracking points trail (max 30 points)
        self.trail = []

    def update(self, bbox, cls_id=None, conf=None):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox)
        if cls_id is not None:
            self.cls_id = int(cls_id)
        if conf is not None:
            self.conf = float(conf)
            
        # Add center point to trail if not NaN or empty
        state = self.kf.get_state()
        if not np.any(np.isnan(state)) and not np.all(state == 0):
            cx = int((state[0] + state[2]) / 2)
            cy = int((state[1] + state[3]) / 2)
            self.trail.append((cx, cy))
            if len(self.trail) > 30:
                self.trail.pop(0)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        bbox = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(bbox)
        
        # Add predicted center to trail when not updated and not NaN
        if not np.any(np.isnan(bbox)) and not np.all(bbox == 0):
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            self.trail.append((cx, cy))
            if len(self.trail) > 30:
                self.trail.pop(0)
            
        return bbox

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.get_state()


class SortTracker:
    """
    SORT Multi-Object Tracker.
    Associates detections to tracks using Hungarian algorithm and Kalman Filter prediction.
    """
    def __init__(self, max_age=15, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        # Reset ID counter when tracker is initialized
        KalmanBoxTracker.count = 0

    def update(self, dets):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,class_id],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns a similar array where the last element is the object ID:
          [[x1,y1,x2,y2,id,class_id,score],...]
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove trackers with NaN states
        for index in sorted(to_del, reverse=True):
            self.trackers.pop(index)
            
        trks = np.delete(trks, to_del, axis=0)
        
        if len(dets) > 0:
            dets_xyxy = dets[:, :4]
        else:
            dets_xyxy = np.empty((0, 4))
            
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            dets_xyxy, trks, self.iou_threshold
        )
        
        # Update matched trackers with corresponding detections
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(
                dets[det_idx, :4],
                cls_id=dets[det_idx, 5] if dets.shape[1] > 5 else 0,
                conf=dets[det_idx, 4] if dets.shape[1] > 4 else 1.0
            )
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            cls_id = dets[i, 5] if dets.shape[1] > 5 else 0
            conf = dets[i, 4] if dets.shape[1] > 4 else 1.0
            trk = KalmanBoxTracker(dets[i, :4], cls_id=cls_id, conf=conf)
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Format: [x1, y1, x2, y2, id, class_id, score]
                ret.append(np.concatenate((d, [trk.id, trk.cls_id, trk.conf])).reshape(1, -1))
            i -= 1
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    @staticmethod
    def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes).
        Returns:
          matched: array of tuples (detection_idx, tracker_idx)
          unmatched_detections: array of indices of unmatched detections
          unmatched_trackers: array of indices of unmatched trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 1), dtype=int)
            
        iou_matrix = iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold)
            if a.all(axis=0).any() or a.all(axis=1).any():
                # Hungarian assignment
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)
            else:
                matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
