from models.segmentation_model import SegmentationModel
from utils.frame_processor import process_frame

class VideoSegmentation:
    def __init__(self):
        self.segmentation_model = SegmentationModel()

    def segment_video_frame(self, frame):
        processed_frame = process_frame(frame)
        return self.segmentation_model.segment_image(processed_frame)