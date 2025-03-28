import cv2
import numpy as np
import os
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.video_processing.models.moo_yolo import MOO_YoloDetector

def draw_detection_boxes(frame, detections):
    """Draw bounding boxes and labels for detected objects"""
    detection_count = len(detections)

    # Draw counter in top right
    counter_text = f"Objects: {detection_count}"
    cv2.rectangle(frame, (frame.shape[1] - 160, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, counter_text, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Define fixed colors for common classes
    color_map = {
        "person": (0, 255, 0),      # Green
        "chair": (0, 165, 255),     # Orange
        "table": (0, 0, 255),       # Red
        "laptop": (255, 0, 0),      # Blue
        "cell phone": (255, 0, 255), # Purple
        "book": (255, 255, 0),      # Cyan
        "tv": (128, 0, 128),        # Purple
        "bottle": (0, 128, 128),    # Brown
        "backpack": (128, 128, 0),  # Olive
        "clock": (0, 255, 255),     # Yellow
    }

    # Draw each detection
    for detection in detections:
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']

        # Get color for this class
        class_name = detection['name']
        if class_name in color_map:
            color = color_map[class_name]
        else:
            # Generate a color for unknown classes
            color_hash = hash(class_name) % 255
            color = (color_hash, 255 - color_hash, 128)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f"{class_name}: {detection['confidence']:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)

        # Draw text
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame

def process_video(video_path):
    """Process video and save frames with detection visualization"""
    # Initialize the detector
    try:
        # First try with the default model
        detector = MOO_YoloDetector(model_size='n')
    except Exception as e:
        print(f"Error initializing detector with default model: {e}")
        print("Trying with ultralytics model...")
        # If that fails, try with the ultralytics model
        detector = MOO_YoloDetector(model_size=None)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output directory for frames
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "frames")
    os.makedirs(output_dir, exist_ok=True)

    # Create output video writer
    output_video_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "output", "detection_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detections = 0

    print(f"Processing video: {video_path}")
    print(f"Saving output to: {output_video_path}")
    print(f"Saving frames to: {output_dir}")
    print("Processing frames...")

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with detector
        print(f"\nProcessing frame {frame_count+1}...")

        # Convert frame to RGB for YOLO processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get raw YOLO results for debugging
        raw_results = detector.model(frame_rgb, verbose=False)
        print(f"Raw YOLO detections: {len(raw_results[0].boxes) if len(raw_results) > 0 else 0}")

        if len(raw_results) > 0 and len(raw_results[0].boxes) > 0:
            print("YOLO detected classes:")
            for i in range(len(raw_results[0].boxes)):
                class_id = int(raw_results[0].boxes.cls[i].item())
                conf = float(raw_results[0].boxes.conf[i].item())
                class_name = detector.class_names[class_id]
                print(f"  - {class_name} (id: {class_id}, conf: {conf:.2f})")

        # Process with our detector
        detections = detector.process_frame(frame_rgb)

        # Update statistics
        frame_count += 1
        total_detections += len(detections)

        # Draw detection boxes on the frame
        display_frame = draw_detection_boxes(frame.copy(), detections)

        # Add frame counter
        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save frame to output directory (every 5 frames to save space)
        if frame_count % 5 == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, display_frame)

        # Write frame to output video
        out.write(display_frame)

        # Print progress
        print(f"Final detections for frame {frame_count}:")
        if detections:
            for obj in detections:
                print(f"  - {obj['name']} (confidence: {obj['confidence']:.2f})")
        else:
            print("  No objects detected in this frame")

    # Release resources
    cap.release()
    out.release()

    # Print summary
    print(f"\nVideo Processing Complete:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total objects detected: {total_detections}")
    print(f"Average objects per frame: {total_detections / frame_count:.2f}")
    print(f"\nOutput video saved to: {output_video_path}")
    print(f"Sample frames saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="GUI application for object detection in videos")
    parser.add_argument("--video", type=str, default="src/utils/data/input_videos/output_temp_file.mp4",
                        help="Path to the video file")
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        print("Creating a test video file...")

        # Make sure the directory exists
        os.makedirs(os.path.dirname(args.video), exist_ok=True)

        # Import and use the create_test_video function from main.py
        from src.main import create_test_video
        create_test_video(args.video)
    
    process_video(args.video)

if __name__ == "__main__":
    main()