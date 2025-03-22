import cv2
import json
import os
from datetime import datetime
from utils.config_loader import load_config
from models.segmentation_model import SegmentationModel

class VideoSegmentor:
    def __init__(self):
        # Load configuration
        self.config = load_config('../configs/settings.yaml')
        
        # Load segmentation model
        self.model = SegmentationModel(self.config)

    def process_frame(self, frame):
        """
        Perform image segmentation on the frame and return the results with a timestamp and time of day.
        """
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Perform image segmentation on the frame
        segmentation_data = self.model.segment_frame(frame)
        segmentation_data['timestamp'] = timestamp
        segmentation_data['time_of_day'] = self.detect_time_of_day(frame)
        
        return segmentation_data

    def detect_time_of_day(self, frame):
        """
        Detect if the current frame is in the morning, night, or dim-lighted setting.
        """
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the average brightness of the frame
        brightness = gray.mean()
        
        # Determine time of day based on brightness
        if brightness > 150:
            return "morning"
        elif 50 <= brightness <= 150:
            return "dim_light"
        else:
            return "night"

def process_video(video_path):
    """
    Process the video to segment objects in real-time.
    """
    segmentor = VideoSegmentor()
    
    cap = cv2.VideoCapture(video_path)
    output_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Segment objects in the frame
        segmentation_data = segmentor.process_frame(frame)
        
        # Append the results to the output data
        output_data.append(segmentation_data)
    
    # Release the video capture object
    cap.release()
    
    # Save the output data to a JSON file
    output_path = os.path.join('output', 'data.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    video_path = "input_videos/input_video.mp4"  # Replace with the actual video path
    process_video(video_path)