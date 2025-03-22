import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class DetectionModel:
    def __init__(self):
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def detect_objects(self, frame):
        with torch.no_grad():
            results = self.model([frame])
        return results[0]