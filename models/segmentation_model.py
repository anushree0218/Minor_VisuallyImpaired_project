import torch
from torchvision.models.segmentation import deeplabv3_resnet50

class SegmentationModel:
    def __init__(self):
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.eval()

    def segment_image(self, frame):
        with torch.no_grad():
            results = self.model([frame])
        return results[0]