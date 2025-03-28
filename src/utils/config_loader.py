import yaml
import os
from typing import Dict

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    # Convert relative path to absolute path from project root
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config