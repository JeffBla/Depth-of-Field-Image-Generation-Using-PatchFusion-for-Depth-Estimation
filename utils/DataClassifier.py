import os
import shutil
import json
from pathlib import Path

def move_images_based_on_metadata(raw_folder:Path, blur_folder:Path, clear_folder:Path, metadata):
    # Create directories if they don't exist
    os.makedirs(blur_folder, exist_ok=True)
    os.makedirs(clear_folder, exist_ok=True)

    # Move files based on metadata
    for id, imgs in metadata.items():
        # Create id number with 4 digits
        id = f"{id:04}"        

        clear_source_path = raw_folder / imgs['clear']
        blur_source_path = raw_folder / imgs['blur']
        
        clear_dest_path = clear_folder / (str(id)+'.JPG')
        blur_dest_path = blur_folder / (str(id)+'.JPG')
        
        if os.path.exists(clear_source_path) and os.path.exists(blur_source_path):
            shutil.move(clear_source_path, clear_dest_path)
            shutil.move(blur_source_path, blur_dest_path)
            print(f"Moved {imgs['clear']} to 'clear' folder.")
            print(f"Moved {imgs['blur']} to 'blur' folder.")
        else:
            print(f"{imgs['clear']} and {imgs['blur']} does not exist in the raw folder.")

def parse_metadata_txt(metadata_file):
    metadata = {}
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines[1:]):  # Skip the header
            clear, blur = line.strip().split()
            metadata[id] = {'clear':f"IMG_{clear}.JPG", 'blur': f"IMG_{blur}.JPG"}
    return metadata

if __name__ == "__main__":
    raw_folder = Path('./data/raw')
    blur_folder = raw_folder / 'blur'
    clear_folder = raw_folder / 'clear'
    metadata_file = raw_folder / 'metadata.txt'
    meatadata_json = raw_folder / 'metadata.json'
    
    metadata = parse_metadata_txt(metadata_file)

    move_images_based_on_metadata(raw_folder, blur_folder, clear_folder, metadata)
    # Save the parsed metadata to a JSON file
    with open(meatadata_json, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)