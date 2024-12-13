import os
import shutil
import json
from pathlib import Path

def move_images_based_on_metadata(target_folder:Path, raw_folder:Path, blur_folder:Path, clear_folder:Path, metadata):
    # Create directories if they don't exist
    os.makedirs(blur_folder, exist_ok=True)
    os.makedirs(clear_folder, exist_ok=True)

    # Catch the last id number
    try:
        last_id = sorted(os.listdir(blur_folder))[-1]
        if last_id != None:
            last_id = int(last_id.split('.')[0])
    except:
        last_id = -1
        print("There is no picture in the blur folder before.")

    # Move files based on metadata
    for id, imgs in metadata.items():
        # Create id number with 4 digits
        id = f"{id+last_id+1:04}"        

        clear_source_path = target_folder / imgs['clear']
        blur_source_path = target_folder / imgs['blur']
        
        clear_dest_path = clear_folder / (str(id)+'.JPG')
        blur_dest_path = blur_folder / (str(id)+'.JPG')
        
        if os.path.exists(clear_source_path) and os.path.exists(blur_source_path):
            shutil.move(clear_source_path, clear_dest_path)
            shutil.move(blur_source_path, blur_dest_path)
            print(f"Moved {imgs['clear']} to 'clear' folder.")
            print(f"Moved {imgs['blur']} to 'blur' folder.")
        else:
            print(f"{imgs['clear']} and {imgs['blur']} does not exist in the raw folder.")

def parse_metadata_txt(metadata_file, isCSV):
    metadata = {}
    with open(metadata_file, 'r') as f:
        lines = f.readlines()
        for id, line in enumerate(lines[1:]):  # Skip the header
            if isCSV:
                blur,clear = line.strip().split(',')
            else:
                blur, clear = line.strip().split(' ')
            metadata[id] = {'clear':f"IMG_{int(clear):04}.JPG", 'blur': f"IMG_{int(blur):04}.JPG"}
    return metadata

if __name__ == "__main__":
    raw_folder = Path('./data/test')
    blur_folder = raw_folder / 'truth'
    clear_folder = raw_folder / 'clear'

    target_folder = Path('./data/test/raw')
    metadata_file = target_folder / 'Classification_test.csv'
    isCSV = True
    
    metadata = parse_metadata_txt(metadata_file, isCSV)

    move_images_based_on_metadata(target_folder, raw_folder, blur_folder, clear_folder, metadata)
