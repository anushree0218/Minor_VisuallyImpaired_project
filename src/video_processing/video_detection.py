import cv2
import json
import os
from datetime import datetime
from utils.config_loader import load_config
from models.detection_model import load_detection_model

# Load configuration
config = load_config('settings.yaml')

# Load detection model
detector = load_detection_model(model_type=config['detection_model_type'], model_path=config['detection_model_path'])

def detect_objects(frame, timestamp):
    """
    Detect objects in a given frame and return the results with a timestamp.
    """
    # Perform object detection using the loaded model
    results = detector.detect(frame)
    
    # Format the results with timestamp
    detection_data = {
        "timestamp": timestamp,
        "objects": results
    }
    return detection_data

def detect_time_of_day(frame):
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
    Process the video to detect objects and time of day in real-time.
    """
    cap = cv2.VideoCapture(video_path)
    output_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Detect objects in the frame
        detection_data = detect_objects(frame, timestamp)
        
        # Detect time of day
        time_of_day = detect_time_of_day(frame)
        detection_data["time_of_day"] = time_of_day
        
        # Append the results to the output data
        output_data.append(detection_data)
    
    # Release the video capture object
    cap.release()
    
    # Save the output data to a JSON file
    output_path = os.path.join('output', 'data.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    video_path = "input_videos/input_video.mp4"  # Replace with the actual video path
    process_video(video_path)