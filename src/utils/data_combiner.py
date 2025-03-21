import json
import os
from datetime import datetime
from typing import Dict

class DataCombiner:
    def __init__(self, output_dir: str = "output/description"):
        self.output_dir = output_dir
        self.json_path = os.path.join(output_dir, "json")
        self.txt_path = os.path.join(output_dir, "text")
        self._create_dirs()
        
        self.combined_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "frames_processed": 0
            },
            "frames": []
        }

    def _create_dirs(self):
        os.makedirs(self.json_path, exist_ok=True)
        os.makedirs(self.txt_path, exist_ok=True)

    def add_frame_data(self, detection_data: Dict, segmentation_data: Dict):
        """Combine detection and segmentation data for a single frame"""
        frame_data = {
            "objects": detection_data["objects"],
            "environment": {
                **detection_data["environment"],
                **segmentation_data["environment"]
            },
            "partial_objects": segmentation_data["partial_objects"],
            "light_direction": segmentation_data["light_direction"]
        }
        
        self.combined_data["frames"].append(frame_data)
        self.combined_data["metadata"]["frames_processed"] += 1

    def save_output(self):
        """Save combined data to JSON and text files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_filename = f"environment_context_{timestamp}.json"
        with open(os.path.join(self.json_path, json_filename), 'w') as f:
            json.dump(self.combined_data, f, indent=2)
        
        # Save TXT
        txt_filename = f"environment_summary_{timestamp}.txt"
        self._save_text_output(os.path.join(self.txt_path, txt_filename))
        
        return json_filename, txt_filename

    def _save_text_output(self, filepath: str):
        """Generate human-readable text summary"""
        summary = [
            f"Environment Context Analysis Report",
            f"Generated at: {self.combined_data['metadata']['created_at']}",
            f"Total Frames Processed: {self.combined_data['metadata']['frames_processed']}",
            "\nKey Findings:"
        ]

        # Aggregate data across frames
        object_counter = {}
        environment_features = {
            'light_intensity': {'low': 0, 'medium': 0, 'high': 0},
            'time_of_day': {'day': 0, 'night': 0},
            'light_direction': {'left': 0, 'center': 0, 'right': 0}
        }

        for frame in self.combined_data["frames"]:
            # Count objects
            for obj in frame["objects"]:
                name = obj["name"]
                object_counter[name] = object_counter.get(name, 0) + 1
                
            # Track environment features
            env = frame["environment"]
            environment_features['light_intensity'][env['light_intensity']] += 1
            environment_features['time_of_day'][env['time_of_day']] += 1
            environment_features['light_direction'][frame['light_direction']] += 1

        # Add object counts
        summary.append("\nDetected Objects:")
        for obj, count in sorted(object_counter.items()):
            summary.append(f"- {obj.capitalize()}: {count} instances")

        # Add environment analysis
        summary.append("\nEnvironment Analysis:")
        summary.append(f"Most Common Light Intensity: {max(environment_features['light_intensity'], key=environment_features['light_intensity'].get)}")
        summary.append(f"Time of Day: {max(environment_features['time_of_day'], key=environment_features['time_of_day'].get)}")
        summary.append(f"Primary Light Direction: {max(environment_features['light_direction'], key=environment_features['light_direction'].get)}")

        # Add partial objects
        partial_counts = sum(len(frame['partial_objects']) for frame in self.combined_data["frames"])
        summary.append(f"\nPartial Objects Detected: {partial_counts}")

        with open(filepath, 'w') as f:
            f.write('\n'.join(summary))