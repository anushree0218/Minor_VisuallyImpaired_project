import cv2
import time
from datetime import datetime
from utils.config_loader import load_config
from utils.download_video import capture_video
from video_processing.video_detection import VideoDetection
from video_processing.video_segmentation import VideoSegmentation
from llm_integration.llm import LLMIntegration
from tts.txt_to_speech import TextToSpeech
from utils.data_combiner import combine_data, save_to_json

def main():
    config = load_config()
    cap = capture_video(config['camera']['source'])
    detection = VideoDetection()
    segmentation = VideoSegmentation()
    llm = LLMIntegration(api_key=config['llm']['api_key'], model=config['llm']['model'])
    tts = TextToSpeech(voice=config['tts']['voice'], speed=config['tts']['speed'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = datetime.now().isoformat()
        detection_results = detection.detect_objects_in_video(frame)
        segmentation_results = segmentation.segment_video_frame(frame)
        combined_data = combine_data(detection_results, segmentation_results, timestamp)

        save_to_json(combined_data, config['output']['json_output_path'])

        with open("src/llm_integration/prompt.txt", "r") as f:
            prompt = f.read() + str(combined_data)

        description = llm.generate_description(prompt)
        tts.speak(description)

        time.sleep(1)  # Adjust the delay as needed

    cap.release()

if __name__ == "__main__":
    main()