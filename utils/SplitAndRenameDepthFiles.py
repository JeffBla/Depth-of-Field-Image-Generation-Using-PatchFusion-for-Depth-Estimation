import os
import shutil
from pathlib import Path
import re

def get_number_from_filename(filename):
    """Extract the number from filename like '0000.JPG.png' or '0000.JPG_uint16.png'"""
    match = re.search(r'(\d+)', str(filename))
    if match:
        return match.group(1)
    return None

def split_and_rename_depth_files():
    # Define paths
    source_dir = Path('data/proc/depth')
    preview_dir = Path('data/proc/depth_preview')
    depth_dir = Path('data/proc/depth')
    
    # Create preview directory if it doesn't exist
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files from source directory
    files = list(source_dir.glob('*.png'))
    
    # Process files
    for file in files:
        # Get the number from original filename
        number = get_number_from_filename(file)
        if number is None:
            print(f'Warning: Could not extract number from {file.name}, skipping...')
            continue
            
        # Create new filename with format '0000.png'
        new_name = f'{number.zfill(4)}.png'
        
        if '_uint16' in file.name:
            # Move and rename uint16 files within depth directory
            dest_path = depth_dir / new_name
            if file != dest_path:  # Only move if the path is different
                shutil.move(str(file), str(dest_path))
                print(f'Renamed {file.name} to {new_name} in depth directory')
        else:
            # Move and rename non-uint16 files to preview directory
            dest_path = preview_dir / new_name
            shutil.move(str(file), str(dest_path))
            print(f'Moved and renamed {file.name} to {new_name} in preview directory')
    
if __name__ == '__main__':
    split_and_rename_depth_files()