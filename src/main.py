from video_processing.video_detection import VideoDetector
from video_processing.video_segmentation import VideoSegmentor
from utils.data_combiner import DataCombiner
import cv2
import json
import os
import numpy as np
import argparse

def create_test_video(output_path):
    """Create a simple test video file for testing purposes"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a simple video with a moving rectangle
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds - longer duration for better detection

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(fps * duration):
        # Create a frame with a moving rectangle
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        x = int(width * (i / (fps * duration)))

        # Make the rectangle larger and more prominent for better detection
        cv2.rectangle(frame, (x, height//3), (x + 100, 2*height//3), (0, 0, 255), -1)

        # Add a circle too for more object variety
        cv2.circle(frame, (width//2, height//2), 50, (255, 0, 0), -1)

        # Add some text
        cv2.putText(frame, "Test Video", (width//4, height//4),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"Created test video at: {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process video for visually impaired assistance")
    parser.add_argument("--gui", action="store_true", help="Run with GUI visualization")
    args = parser.parse_args()

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Hardcoded video path (no more URL downloads!)
    video_path = os.path.join(project_root, "src", "utils", "data", "input_videos","data", "input_videos", "output_temp_file.mp4")

    # Fix path if it's incorrect (remove duplicate "data/input_videos")
    if not os.path.exists(video_path):
        alternative_path = os.path.join(project_root, "src", "utils", "data", "input_videos", "output_temp_file.mp4")
        if os.path.exists(alternative_path):
            video_path = alternative_path

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Creating a sample video file for testing...")

        # Ensure directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # Create a simple test video if the original doesn't exist
        create_test_video(video_path)

    # If GUI mode is requested, run the video processing with visualization
    if args.gui:
        from gui_app import process_video as gui_process_video
        print("Starting video processing with visualization...")
        print("(Note: This will save processed frames and video to the output directory)")
        gui_process_video(video_path)
        return

    # Ensure output folder exists
    os.makedirs(os.path.join(project_root, "output"), exist_ok=True)

    # Initialize components
    data_combiner = DataCombiner(output_dir=os.path.join(project_root, "output", "description"))
    detector = VideoDetector()
    segmentor = VideoSegmentor(config_path=os.path.join(project_root, "configs", "settings.yaml"))

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with detection and segmentation modules
        detection_data = detector.process_frame(frame)
        segmentation_data = segmentor.process_frame(frame)

        # Print detection information for debugging
        if detection_data["objects"]:
            print(f"Frame {frame_count}: Detected {len(detection_data['objects'])} objects:")
            for obj in detection_data["objects"]:
                print(f"  - {obj['name']} (confidence: {obj['confidence']:.2f})")
        else:
            print(f"Frame {frame_count}: No objects detected")

        # Combine results from both processing steps
        data_combiner.add_frame_data(detection_data, segmentation_data)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    # Release video capture
    cap.release()

    # Save final combined output data
    output_path = os.path.join(project_root, "output", "data.json")
    with open(output_path, 'w') as f:
        json.dump(data_combiner.combined_data, f, indent=2)

    print(f"Final output saved to: {output_path}")

    # Print summary of objects detected
    all_objects = []
    for frame in data_combiner.combined_data["frames"]:
        all_objects.extend(frame["objects"])

    print("\n===== DETECTION SUMMARY =====")
    print(f"Total frames processed: {frame_count}")
    print(f"Total objects detected: {len(all_objects)}")

    # Count objects by class
    object_counts = {}
    for obj in all_objects:
        name = obj["name"]
        object_counts[name] = object_counts.get(name, 0) + 1

    if object_counts:
        print("\nObjects by class:")
        for name, count in object_counts.items():
            print(f"  - {name}: {count}")
    else:
        print("\nNo objects were detected in the video.")

    print("============================\n")

    # Also save using the DataCombiner's save_output method
    json_file, txt_file = data_combiner.save_output()
    print(f"Detailed output saved to: {json_file} and {txt_file}")


if __name__ == "__main__":
    main()
