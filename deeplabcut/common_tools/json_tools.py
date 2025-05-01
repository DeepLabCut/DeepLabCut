import json
import os
from pathlib import Path

def convert_json_indent_folder_level(source_folder_path, target_folder_path):
    """
    Read JSON files and save them with proper indentation.
    
    Args:
        source_folder_path (str): Path to the folder containing original JSON files
        target_folder_path (str): Path to save the formatted JSON files
    """
    # Create target folder if it doesn't exist
    Path(target_folder_path).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files from source folder
    json_files = [f for f in os.listdir(source_folder_path) if f.endswith('.json')]
    print(f"Processing {len(json_files)} JSON files...")
    
    for json_file in json_files:
        source_path = os.path.join(source_folder_path, json_file)
        target_path = os.path.join(target_folder_path, json_file)

        # Read source JSON file and save with proper indentation
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=4)