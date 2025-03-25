import cv2
import torch
import torchvision
from typing import Dict
from models.detection_model import DetectionModel

class VideoDetector:
    def __init__(self):
        self.model = DetectionModel()
        
    def process_frame(self, frame) -> Dict:
        """Process a single frame for object detection"""
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get detection results
        detections = self.model.detect_objects(frame_rgb)
        
        # Get environment analysis
        light_intensity = self._get_light_intensity(frame)
        time_of_day = self._get_time_of_day(frame)
        
        return {
            "objects": detections,
            "environment": {
                "light_intensity": light_intensity,
                "time_of_day": time_of_day
            }
        }

    def _get_light_intensity(self, frame) -> str:
        """Estimate light intensity based on average brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = cv2.mean(gray)[0]
        if avg_brightness > 200: return "high"
        if avg_brightness > 100: return "medium"
        return "low"

    def _get_time_of_day(self, frame) -> str:
        """Classify time of day based on overall brightness"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return "day" if cv2.mean(gray)[0] > 128 else "night"