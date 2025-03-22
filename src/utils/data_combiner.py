import json
from datetime import datetime

def combine_data(detection_results, segmentation_results, timestamp):
    combined_data = {
        "timestamp": timestamp,
        "objects": detection_results,
        "segments": segmentation_results
    }
    return combined_data

def save_to_json(data, file_path):
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')