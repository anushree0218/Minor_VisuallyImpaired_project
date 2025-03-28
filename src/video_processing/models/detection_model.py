import torch
import torchvision
from typing import List, Dict

class DetectionModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda': 
            print("CUDA is available and will be used.")
         
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT').to(self.device)
        self.model.eval()
        self.coco_names = [
            "person",
            "backpack",
            "bottle",
            "chair",
            "tv",
            "laptop",
            "cell phone",
            "book",
            "clock",
            "notebook",
            "fan",
            "desk",
            "curtains",
            "table",
            "bench",
            "whiteboard",
            "ceiling light",
            "ceiling fan",
            "television",
            "door",
            "ac vents",
            "curtain",
            "window",
            "windows"
        ]
        # Define allowed objects for your application (adjust as needed)
        self.allowed_objects = {"person", "chair", "table", "door", "tv", "laptop", "cell phone", "bottle"}

    def detect_objects(self, frame) -> List[Dict]:
        """Detect objects in a frame using Faster R-CNN"""
        tensor = torchvision.transforms.functional.to_tensor(frame).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([tensor])[0]
        
        objects = []
        for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
            if score > 0.6:  # Increased threshold to filter out low-confidence detections
                obj_name = self.coco_names[label.item()]
                if obj_name not in self.allowed_objects:
                    continue
                objects.append({
                    "name": obj_name,
                    "confidence": round(score.item(), 2),
                    "bbox": [round(coord.item()) for coord in box]
                })
                
        return objects
