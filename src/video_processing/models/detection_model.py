import torch
import torchvision
from typing import List, Dict

class DetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.coco_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

    def detect_objects(self, frame) -> List[Dict]:
        """Detect objects in a frame using Faster R-CNN"""
        tensor = torchvision.transforms.functional.to_tensor(frame).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([tensor])[0]
        
        objects = []
        for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
            if score > 0.5:  # Confidence threshold
                objects.append({
                    "name": self.coco_names[label.item()],
                    "confidence": round(score.item(), 2),
                    "bbox": [round(coord.item()) for coord in box]
                })
                
        return objects