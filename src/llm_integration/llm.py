# llm_integration/llm.py
import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class EnvironmentDescriber:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
            
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {'Content-Type': 'application/json'}

    def generate_description(self, json_path):
        data = self._parse_json(json_path)
        prompt = self._create_prompt(data)
        return self._call_gemini_api(prompt)

    def _call_gemini_api(self, prompt):
        try:
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            return self._parse_response(response.json())
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except KeyError as e:
            raise Exception(f"Unexpected API response format: {str(e)}")

    def _parse_response(self, response_json):
        try:
            return response_json['candidates'][0]['content']['parts'][0]['text']
        except KeyError:
            raise Exception("No valid content in API response")

    def _create_prompt(self, data):
        return f"""Act as a visual assistant for blind people. Analyze this environment data and create a detailed, vivid description in essay format. Follow these guidelines:
1. Start with overall scene summary
2. Describe lighting conditions and light direction
3. Mention time of day/night
4. List and describe all detected objects and their positions
5. Explain colors and textures
6. Note any potential obstacles or safety concerns
7. Keep descriptions tactile and sound-oriented
8. Use simple, clear language
9. Limit to 3-4 paragraphs

Environment Data:
{data}"""

    def _parse_json(self, json_path):
        # Keep the same JSON parsing logic from previous implementation
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        summary = {
            'total_frames': data['metadata']['frames_processed'],
            'light_conditions': {},
            'objects': {},
            'environment_features': {}
        }

        light_counts = {'low': 0, 'medium': 0, 'high': 0}
        time_counts = {'day': 0, 'night': 0}
        direction_counts = {'left': 0, 'center': 0, 'right': 0}
        
        for frame in data['frames']:
            light_counts[frame['environment']['light_intensity']] += 1
            time_counts[frame['environment']['time_of_day']] += 1
            direction_counts[frame['light_direction']] += 1
            
            for obj in frame['objects']:
                name = obj['name'].lower()
                summary['objects'][name] = summary['objects'].get(name, 0) + 1
        
        summary['light_conditions'] = {
            'intensity': max(light_counts, key=light_counts.get),
            'time_of_day': max(time_counts, key=time_counts.get),
            'common_direction': max(direction_counts, key=direction_counts.get),
            'direction_distribution': direction_counts
        }
        
        sample_frame = data['frames'][0]['environment']
        summary['environment_features'] = {
            'dominant_colors': sample_frame['dominant_colors'],
            'texture_complexity': sum(f['environment']['texture_complexity'] for f in data['frames'])/len(data['frames'])
        }
        
        return json.dumps(summary, indent=2)