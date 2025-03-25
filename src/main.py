from utils.download_video import download_and_extract
from video_processing.video_detection import VideoDetector
from video_processing.video_segmentation import VideoSegmentor
from utils.data_combiner import DataCombiner
import argparse
import re
import cv2
import json
import os
import gdown

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive_url', type=str, required=True, help='Google Drive URL')
    args = parser.parse_args()

    # Initialize components
    data_combiner = DataCombiner()
    video_path = handle_video_input(args.drive_url)
    
    detector = VideoDetector()
    segmentor = VideoSegmentor()

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        detection_data = detector.process_frame(frame)
        segmentation_data = segmentor.process_frame(frame)
        
        # Combine and store results
        data_combiner.add_frame_data(detection_data, segmentation_data)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    # Save final output
    output_path = os.path.join("output", "data.json")
    with open(output_path, 'w') as f:
        json.dump(data_combiner.combined_data, f, indent=2)
        
    print(f"Final output saved to: {output_path}")



def download_and_extract(drive_url: str, output_dir: str) -> str:
    # Extract the file ID from the Google Drive URL
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', drive_url)
    if not match:
        raise ValueError("Invalid Google Drive URL format. Expected URL with '/d/{file_id}'.")
    file_id = match.group(1)
    
    # Construct the direct download URL
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the file into the output directory using gdown
    file_path = gdown.download(download_url, output=output_dir, quiet=True)
    
    if file_path is None:
        raise RuntimeError("Failed to download the file from Google Drive.")
    
    return file_path

def handle_video_input(drive_url: str) -> str:
    return download_and_extract(drive_url, output_dir="output/input_videos")

if __name__ == "__main__":
    main()