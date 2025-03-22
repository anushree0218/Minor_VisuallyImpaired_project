import cv2
import torch
import torchvision
from typing import Union, List, Dict

class DetectionModel:
    def __init__(self, model_type: str = "haar", model_path: str = None):
        """
        Initialize the detection model.

        Args:
            model_type (str): Type of detection model to use. Options: "haar" (Haar Cascade) or "maskrcnn" (Mask R-CNN).
            model_path (str): Path to the custom model weights (for Mask R-CNN).
        """
        self.model_type = model_type

        if self.model_type == "haar":
            # Load Haar Cascade for basic object detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif self.model_type == "maskrcnn":
            # Load a pre-trained Mask R-CNN model
            self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            
            # Load custom weights if a model path is provided
            if model_path:
                self.model.load_state_dict(torch.load(model_path))
            
            # Set the model to evaluation mode
            self.model.eval()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def detect(self, frame) -> List[Dict]:
        """
        Detect objects in a given frame.

        Args:
            frame: Input frame (numpy array).

        Returns:
            List of detected objects, each represented as a dictionary.
        """
        if self.model_type == "haar":
            return self._detect_haar(frame)
        elif self.model_type == "maskrcnn":
            return self._detect_maskrcnn(frame)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _detect_haar(self, frame) -> List[Dict]:
        """
        Detect objects using Haar Cascade.

        Args:
            frame: Input frame (numpy array).

        Returns:
            List of detected objects, each represented as a dictionary.
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects (e.g., faces)
        objects = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Format the results
        results = [{"type": "face", "bbox": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in objects]
        return results

    def _detect_maskrcnn(self, frame) -> List[Dict]:
        """
        Detect objects using Mask R-CNN.

        Args:
            frame: Input frame (numpy array).

        Returns:
            List of detected objects, each represented as a dictionary.
        """
        # Convert frame to a tensor and normalize
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        tensor = transform(frame).unsqueeze(0)
        
        # Perform object detection
        with torch.no_grad():
            predictions = self.model(tensor)[0]
        
        # Format the results
        results = []
        for i in range(len(predictions["boxes"])):
            box = predictions["boxes"][i].tolist()
            label = predictions["labels"][i].item()
            score = predictions["scores"][i].item()
            
            results.append({
                "type": label,
                "bbox": box,
                "confidence": score
            })
        
        return results

def load_detection_model(model_type: str = "haar", model_path: str = None) -> DetectionModel:
    """
    Load a detection model.

    Args:
        model_type (str): Type of detection model to use. Options: "haar" (Haar Cascade) or "maskrcnn" (Mask R-CNN).
        model_path (str): Path to the custom model weights (for Mask R-CNN).

    Returns:
        DetectionModel: An instance of the detection model.
    """
    return DetectionModel(model_type=model_type, model_path=model_path)