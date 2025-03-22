import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.download_video import download_and_extract
from src.video_processing.video_detection import VideoDetector
from src.video_processing.video_segmentation import VideoSegmentor
from src.utils.data_combiner import DataCombiner
import argparse
import cv2
from datetime import datetime

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_url', type=str, help='Google Drive URL')
    args = parser.parse_args()

    # Initialize components
    data_combiner = DataCombiner()
    video_path = handle_video_input(args.drive_url)
    
    detector = VideoDetector()
    segmentor = VideoSegmentor()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Process frame for detection
        detection_data = detector.process_frame(frame)
        detection_data['timestamp'] = timestamp
        detection_data['time_of_day'] = detector.detect_time_of_day(frame)
        
        # Process frame for segmentation
        segmentation_data = segmentor.process_frame(frame)
        segmentation_data['timestamp'] = timestamp
        segmentation_data['time_of_day'] = segmentor.detect_time_of_day(frame)
        
        # Combine and store results
        data_combiner.add_frame_data(detection_data, segmentation_data)
        
        # Print progress every 10 frames
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Release the video capture object
    cap.release()
    
    # Save final output
    json_file, txt_file = data_combiner.save_output()
    print(f"Analysis complete! Results saved to:")
    print(f"- JSON: output/description/json/{json_file}")
    print(f"- TXT: output/description/text/{txt_file}")

def handle_video_input(drive_url: str) -> str:
    """
    Handle video input by downloading from Google Drive or using a default local video.
    """
    if drive_url:
        return download_and_extract(drive_url)
    else:
        return "data/input_videos/default.mp4"

if __name__ == "__main__":
    main()