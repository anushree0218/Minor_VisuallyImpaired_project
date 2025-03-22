import torch
from torchvision.transforms import functional as F

def process_frame(frame):
    frame = F.to_tensor(frame)
    frame = F.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return frame