from models.segmentation_model import SegmentationModel
from typing import Dict
import json

class VideoSegmentor:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        from ..utils.config_loader import load_config
        self.config = load_config(config_path)
        self.model = SegmentationModel(self.config)
        
    def process_frame(self, frame) -> Dict:
        """Process a single frame for segmentation"""
        return self.model.segment_frame(frame)