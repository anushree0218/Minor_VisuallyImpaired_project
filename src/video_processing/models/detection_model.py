import cv2
import numpy as np

class DetectionModel:
    def __init__(self):
        # Load MobileNet-SSD model
        self.prototxt = "C:/Users/KIIT/OneDrive/Documents/GitHub/Minor_VisuallyImpaired_project/src/video_processing/models/deploy.prototxt"  # Path to model architecture
        self.model = "C:/Users/KIIT/OneDrive/Documents/GitHub/Minor_VisuallyImpaired_project/src/video_processing/models/mobilenet_iter_73000.caffemodel" # Path to pre-trained weights
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        if self.net.empty():
            raise ValueError("Failed to load MobileNet-SSD model.")

        # COCO class labels
        self.classes = [
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
            "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
            "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
            "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def detect(self, frame):
        """Detect objects in a frame using MobileNet-SSD."""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        class_ids = []
        confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                boxes.append([startX, startY, endX - startX, endY - startY])
                class_ids.append(class_id)
                confidences.append(float(confidence))

        return boxes, class_ids, confidences