import cv2
import numpy as np
from models.segmentation_model import SegmentationModel

def process_video_segmentation(video_path, output_path, segmentation_model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Segment frame with multi-objective optimization
        segmentation_data = segmentation_model.segment_frame(frame)

        # Overlay segmentation masks
        for mask_data in segmentation_data['detailed_masks']:
            mask = np.array(mask_data['mask'])
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (frame_width, frame_height))
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        # Write frame to output video
        out.write(frame)

        # Display frame (optional)
        cv2.imshow('Video Segmentation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()