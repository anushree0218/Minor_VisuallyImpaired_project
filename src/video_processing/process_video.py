from models.detection_model import DetectionModel
from models.segmentation_model import SegmentationModel
from video_detection import process_video_detection
from video_segmentation import process_video_segmentation

# Initialize models
detection_model = DetectionModel()
segmentation_model = SegmentationModel({
    "use_gpu": True,
    "min_mask_area": 1000
})

# Process video with detection
process_video_detection(
    video_path="src/utils/data/input_videos/downloaded_content.mov",
    output_path="src/utils/data/output_videos",
    detection_model=detection_model
)

# Process video with segmentation
process_video_segmentation(
    video_path="src/utils/data/input_videos/downloaded_content.mov",
    output_path="src/utils/data/output_videos",
    segmentation_model=segmentation_model
)