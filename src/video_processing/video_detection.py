from models.detection_model import DetectionModel
from utils.frame_processor import process_frame

class VideoDetection:
    def __init__(self):
        self.detection_model = DetectionModel()

    def detect_objects_in_video(self, frame):
        processed_frame = process_frame(frame)
        return self.detection_model.detect_objects(processed_frame)