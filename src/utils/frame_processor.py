import cv2

def preprocess_frame(frame):
    """Basic frame preprocessing"""
    return cv2.resize(frame, (640, 480))  # Resize for consistent processing