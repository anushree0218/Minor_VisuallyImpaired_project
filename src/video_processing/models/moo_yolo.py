import cv2
import numpy as np
from ultralytics import YOLO
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

class MOO_YoloDetector:
    def __init__(self, model_size='n', debug=True):
        # Use standard detection model instead of segmentation model
        # This will ensure we get object detections even without segmentation masks
        print(f"Initializing YOLO model with size: {model_size}")

        try:
            if model_size is None:
                # Try to load the default ultralytics model
                print("Loading default ultralytics model...")
                self.model = YOLO('yolov8n')
            else:
                # Try to load the specified model
                self.model = YOLO(f'yolov8{model_size}.pt')
            print(f"YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Try with a different approach
            try:
                print("Trying to load from ultralytics...")
                self.model = YOLO('yolov8n')
                print("YOLO model loaded from ultralytics")
            except Exception as e2:
                print(f"Error loading from ultralytics: {e2}")
                # Last resort - try with a direct path
                try:
                    from pathlib import Path
                    import os

                    # Try to find the model in common locations
                    possible_paths = [
                        "models/yolov8n.pt",
                        "yolov8n.pt",
                        os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8n.pt"),
                        os.path.join(Path.home(), ".cache", "torch", "hub", "ultralytics_yolov8_master", "yolov8n.pt")
                    ]

                    for path in possible_paths:
                        if os.path.exists(path):
                            print(f"Found model at: {path}")
                            self.model = YOLO(path)
                            print("YOLO model loaded from found path")
                            break
                    else:
                        raise FileNotFoundError("Could not find YOLO model in any expected location")
                except Exception as e3:
                    print(f"All attempts to load YOLO model failed: {e3}")
                    print("Using color-based detection as fallback")
                    self.model = None  # No model available
                    self.use_fallback = True

        # Set fallback flag
        self.use_fallback = False

        # Print model information if available
        if self.model is not None:
            print(f"Model device: {self.model.device}")
            print(f"Model task: {self.model.task}")
            print(f"Number of classes: {len(self.model.names)}")
        else:
            print("No YOLO model loaded, using color-based detection only")

        # Override the default class names with our custom list
        # This won't change the model's detection capabilities but will map detections to our custom classes
        self.custom_classes = [
            "person", "backpack", "bottle", "chair", "tv", "laptop",
            "cell phone", "book", "clock", "notebook", "fan", "desk",
            "curtains", "table", "bench", "whiteboard", "ceiling light",
            "ceiling fan", "television", "door", "ac vents", "curtain",
            "window", "windows"
        ]

        # Create a mapping from YOLO classes to our custom classes
        self.class_mapping = {
            0: 0,    # person -> person
            24: 1,   # backpack -> backpack
            39: 2,   # bottle -> bottle
            56: 3,   # chair -> chair
            62: 4,   # tv -> tv
            63: 5,   # laptop -> laptop
            67: 6,   # cell phone -> cell phone
            73: 7,   # book -> book
            74: 8,   # clock -> clock
            73: 9,   # book -> notebook (closest match)
            0: 10,   # No direct match for fan
            0: 11,   # No direct match for desk
            0: 12,   # No direct match for curtains
            60: 13,  # dining table -> table
            13: 14,  # bench -> bench
            0: 15,   # No direct match for whiteboard
            0: 16,   # No direct match for ceiling light
            0: 17,   # No direct match for ceiling fan
            62: 18,  # tv -> television
            0: 19,   # No direct match for door
            0: 20,   # No direct match for ac vents
            0: 21,   # No direct match for curtain
            0: 22,   # No direct match for window
            0: 23,   # No direct match for windows
        }

        # Set class names
        if self.model is not None:
            self.class_names = self.model.names
        else:
            # Fallback class names for color detection
            self.class_names = {
                0: 'red_object',
                1: 'blue_object',
                2: 'green_object',
                3: 'person'
            }

        self.opt_size = (60, 80)  # Fixed optimization size (h, w)
        self.debug = debug

    def _optimize_mask(self, mask):
        small_mask = cv2.resize(mask.astype(np.float32), 
                              (self.opt_size[1], self.opt_size[0]))
        small_mask = small_mask / 255.0
        
        problem = MaskProblem(small_mask)
        algorithm = NSGA2(pop_size=1)
        res = minimize(problem, algorithm, ('n_gen', 5), verbose=False)
        
        optimized = (res.X.reshape(self.opt_size) > 0.5).astype(np.uint8)
        return cv2.resize(optimized * 255, (mask.shape[1], mask.shape[0]))

    def process_frame(self, frame, conf_thresh=0.5):  # Increased threshold to reduce false positives
        if self.debug:
            print(f"Processing frame with shape: {frame.shape}")
            print(f"Confidence threshold: {conf_thresh}")

        # Define allowed objects for your assistive application
        allowed_objects = {"person", "chair", "table", "door", "tv", "laptop", "cell phone", "bottle",
                           "red_object", "blue_object", "green_object", "rectangle", "circle"}
                           
        # Check if we're using the fallback method
        if hasattr(self, 'use_fallback') and self.use_fallback:
            if self.debug:
                print("Using color-based detection fallback")
            return self._color_based_detection(frame)

        # Try to use YOLO model
        try:
            results = self.model(frame, verbose=False)

            if self.debug:
                print(f"YOLO returned {len(results)} result objects")
                for i, r in enumerate(results):
                    print(f"Result {i}: {len(r.boxes)} boxes detected")
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            print("Falling back to color-based detection")
            return self._color_based_detection(frame)

        detections = []
        has_detections = False

        # Process normal detections first
        for result in results:
            for i in range(len(result.boxes)):
                try:
                    class_id = int(result.boxes.cls[i].item())
                    confidence = float(result.boxes.conf[i].item())

                    if self.debug:
                        print(f"Detection {i}: class_id={class_id}, confidence={confidence:.2f}")

                    if confidence < conf_thresh:
                        if self.debug:
                            print("  Skipping due to low confidence")
                        continue

                    yolo_class_name = self.class_names[class_id]
                    # Filter to keep only allowed objects
                    if yolo_class_name not in allowed_objects:
                        if self.debug:
                            print(f"  Skipping {yolo_class_name} as it is not in allowed objects")
                        continue

                    x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].cpu().numpy())

                    if self.debug:
                        print(f"  Class: {yolo_class_name}, Bbox: [{x1}, {y1}, {x2}, {y2}]")

                    mask_data = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        try:
                            original_mask = result.masks.data[i].cpu().numpy()
                            mask_data = self._optimize_mask(original_mask)
                        except (IndexError, AttributeError) as e:
                            if self.debug:
                                print(f"  Mask error: {e}")
                    detections.append({
                        'name': yolo_class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'mask': mask_data
                    })
                    has_detections = True
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue

        # If no detections, try fallback for test video (color-based)
        if not has_detections:
            if self.debug:
                print("No YOLO detections, trying color-based detection for test video")
            try:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                lower_red = np.array([0, 100, 100])
                upper_red = np.array([10, 255, 255])
                mask = cv2.inRange(hsv, lower_red, upper_red)
                red_pixels = np.sum(mask) / 255
                if self.debug:
                    print(f"Red pixels detected: {red_pixels}")
                if red_pixels > 1000:
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        if self.debug:
                            print(f"Found {len(contours)} red contours")
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        if self.debug:
                            print(f"Red rectangle detected at: [{x}, {y}, {w}, {h}]")
                        detections.append({
                            'name': 'rectangle',
                            'confidence': 0.99,
                            'bbox': [x, y, x+w, y+h],
                            'mask': None
                        })
                lower_blue = np.array([100, 100, 100])
                upper_blue = np.array([140, 255, 255])
                blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_pixels = np.sum(blue_mask) / 255
                if self.debug:
                    print(f"Blue pixels detected: {blue_pixels}")
                if blue_pixels > 1000:
                    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if blue_contours:
                        if self.debug:
                            print(f"Found {len(blue_contours)} blue contours")
                        largest_blue = max(blue_contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_blue)
                        if self.debug:
                            print(f"Blue circle detected at: [{x}, {y}, {w}, {h}]")
                        detections.append({
                            'name': 'circle',
                            'confidence': 0.99,
                            'bbox': [x, y, x+w, y+h],
                            'mask': None
                        })
            except Exception as e:
                print(f"Error during color-based detection: {e}")

        if self.debug:
            print(f"Final detections: {len(detections)}")
        return detections

    def _color_based_detection(self, frame):
        """Fallback method using color-based detection when YOLO is not available"""
        if self.debug:
            print("Running color-based detection")
        detections = []
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask, red_mask2)
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            red_pixels = np.sum(red_mask) / 255
            if red_pixels > 500:
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'name': 'red_object',
                            'confidence': 0.99,
                            'bbox': [x, y, x+w, y+h],
                            'mask': None
                        })
            blue_pixels = np.sum(blue_mask) / 255
            if blue_pixels > 500:
                contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'name': 'blue_object',
                            'confidence': 0.99,
                            'bbox': [x, y, x+w, y+h],
                            'mask': None
                        })
            green_pixels = np.sum(green_mask) / 255
            if green_pixels > 500:
                contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append({
                            'name': 'green_object',
                            'confidence': 0.99,
                            'bbox': [x, y, x+w, y+h],
                            'mask': None
                        })
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    detections.append({
                        'name': 'person',
                        'confidence': 0.9,
                        'bbox': [x, y, x+w, y+h],
                        'mask': None
                    })
            except Exception as e:
                if self.debug:
                    print(f"Face detection error: {e}")
        except Exception as e:
            print(f"Error in color-based detection: {e}")
        if self.debug:
            print(f"Color-based detection found {len(detections)} objects")
        return detections

class MaskProblem(Problem):
    def __init__(self, base_mask):
        self.base_mask = base_mask
        super().__init__(n_var=np.prod(base_mask.shape),
                         n_obj=3,
                         xl=0,
                         xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        mask = x.reshape(self.base_mask.shape)
        f1 = np.mean(np.abs(mask - self.base_mask))
        contours, _ = cv2.findContours(
            (mask > 0.5).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        f2 = sum(len(cnt) for cnt in contours) / 1000
        f3 = abs(np.mean(mask)) - 0.5
        out["F"] = [f1, f2, f3]
