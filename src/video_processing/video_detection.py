##import torch
##import torchvision
from typing import Dict
from .models.detection_model import DetectionModel
import cv2
from .models.moo_yolo import MOO_YoloDetector
class VideoDetector:
    def __init__(self):
        self.detector = MOO_YoloDetector(model_size='n')
        
    def process_frame(self, frame) -> dict:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.process_frame(frame_rgb)
        
        return {
            "objects": [{
                "name": d['name'],
                "confidence": d['confidence'],
                "bbox": d['bbox']
            } for d in detections],
            "environment": {
                "light_intensity": self._get_light_intensity(frame),
                "time_of_day": self._get_time_of_day(frame)
            }
        }
    def _get_light_intensity(self, frame) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = cv2.mean(gray)[0]
        if avg_brightness > 200: return "high"
        if avg_brightness > 100: return "medium"
        return "low"

    def _get_time_of_day(self, frame) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return "day" if cv2.mean(gray)[0] > 128 else "night"