import torch
import torchvision
import cv2
import numpy as np
from typing import Dict, List
from datetime import datetime

class SegmentationModel:
    def __init__(self, config: dict):
        self.device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()
        self.min_mask_area = config['min_mask_area']
        self.coco_names = [
            'unlabeled', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
            'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
            'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def segment_frame(self, frame) -> Dict:
        """Perform image segmentation using Mask RCNN"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_of_day = self._detect_time_of_day(frame)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torchvision.transforms.functional.to_tensor(frame_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([tensor])[0]
        
        segmentation_data = self._process_predictions(frame, predictions)
        segmentation_data['timestamp'] = timestamp
        segmentation_data['time_of_day'] = time_of_day
        
        return segmentation_data

    def _process_predictions(self, frame, predictions) -> Dict:
        segmentation_data = {
            "partial_objects": [],
            "light_direction": self._estimate_light_direction(frame),
            "environment": {
                "dominant_colors": self._get_dominant_colors(frame),
                "texture_complexity": self._calculate_texture_complexity(frame)
            },
            "detailed_masks": []
        }

        masks = predictions['masks'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        for i in range(len(scores)):
            if scores[i] > 0.5:  # Confidence threshold
                mask = masks[i, 0]
                label = self.coco_names[labels[i]]
                
                if mask.sum() > self.min_mask_area:
                    segmentation_data['detailed_masks'].append({
                        'class': label,
                        'mask': mask.astype(np.uint8).tolist(),
                        'confidence': float(scores[i])
                    })

        segmentation_data['partial_objects'] = self._detect_partial_objects(masks)
        return segmentation_data

    def _estimate_light_direction(self, frame) -> str:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        moments = cv2.moments(edges)
        cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 0
        return 'left' if cx < frame.shape[1]//3 else 'right' if cx > 2*frame.shape[1]//3 else 'center'

    def _detect_partial_objects(self, masks) -> List:
        partial_objects = []
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < mask.shape[1]*0.3 or h < mask.shape[0]*0.3:
                    partial_objects.append('partial_object')
        return partial_objects
    
    def _get_dominant_colors(self, frame, num_colors=3):
        """Extract dominant colors using k-means clustering"""
        pixels = frame.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centers[np.bincount(labels.flatten()).argsort()[::-1]].astype(int).tolist()

    def _calculate_texture_complexity(self, frame):
        """Calculate texture complexity using variance of Laplacian"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _detect_time_of_day(self, frame):
        """Detect if the current frame is in the morning, night, or dim-lighted setting."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness > 150:
            return "morning"
        elif 50 <= brightness <= 150:
            return "dim_light"
        else:
            return "night"