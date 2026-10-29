import cv2
import numpy as np
from ultralytics import YOLO
import torch

class YOLODetector:
    """
    A wrapper class for YOLO model inference (Ultralytics).
    Manages loading, confidence filtering, class filtering, and prediction.
    """
    def __init__(self, model_name="yolov8n.pt", device=None):
        """
        Loads the YOLO model on the specified device (CPU/GPU).
        """
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model '{model_name}' on device: {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        # Get class names
        self.names = self.model.names

    def detect(self, frame, conf_threshold=0.35, iou_threshold=0.45, classes=None):
        """
        Runs object detection on the input frame.
        
        Params:
          frame: BGR NumPy array (from cv2.imread / cv2.VideoCapture)
          conf_threshold: confidence score threshold
          iou_threshold: NMS overlap threshold
          classes: list of class IDs to filter detections (e.g., [0, 2] for person, car).
                   If None, all classes are detected.
                   
        Returns:
          A numpy array of shape (N, 6) in the format [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get bounding box coordinates in xyxy format
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                cls_id = box.cls[0].cpu().item()
                
                detections.append([
                    xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                    conf,
                    cls_id
                ])
                
        if len(detections) == 0:
            return np.empty((0, 6))
            
        return np.array(detections)
        
    def get_class_names(self):
        """
        Returns a dictionary of class names.
        """
        return self.names
