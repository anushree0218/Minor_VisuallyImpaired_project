# Visual Assistance Project

This project processes video to detect objects and provide environmental context for visually impaired users.

## Features

- Object detection using YOLOv8
- Environment analysis (light intensity, time of day)
- GUI visualization of detected objects
- JSON and text output of processed data

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Standard Processing

Run the main script to process a video and generate JSON and text outputs:

```
python src/main.py
```

This will process the default test video or create one if it doesn't exist.

### Visualization Mode

Run with the `--gui` flag to process the video with visualization:

```
python src/main.py --gui
```

This will:
- Process the video with object detection
- Save frames with detection visualization to the output/frames directory
- Create a video file with detection visualization at output/detection_video.mp4
- Show detection information in the console

The visualization includes:
- Bounding boxes around detected objects
- Object class names and confidence scores
- A counter showing the number of objects detected in each frame

## Output

The program generates:
- JSON files with detailed detection and segmentation data
- Text summaries of the environment and detected objects

Output files are saved in the `output/description` directory.# Minor_VisuallyImpaired_project