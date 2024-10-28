import cv2
import os
import hydra
from pathlib import Path

def process_images_in_folder(input_folder, output_folder):
    clear_input_path = input_folder / 'clear'
    blur_input_path = input_folder / 'blur'
    clear_output_path = output_folder / 'clear'
    blur_output_path = output_folder / 'blur'

    os.makedirs(clear_output_path, exist_ok=True)
    os.makedirs(blur_output_path, exist_ok=True)

    for image_name in os.listdir(clear_input_path):
        input_path = clear_input_path / image_name
        output_path = clear_output_path / image_name
        process_images(input_path, output_path)

    for image_name in os.listdir(blur_input_path):
        input_path = blur_input_path / image_name
        output_path = blur_output_path / image_name
        process_images(input_path, output_path)

def process_images(input_path, output_path):

    # Load the image
    image = cv2.imread(input_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image not found at {input_path}")
    
    # Resize the image to 1024x1024
    resized_image = cv2.resize(image, (1024, 1024))
    
    # Save the processed image
    cv2.imwrite(output_path, resized_image)
    print(f"Processed image saved at {output_path}")

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    print(os.getcwd())
    imgProcessConf = cfg.utils.ImgProcess
    # Contain clear and blur images
    process_images_in_folder(Path(imgProcessConf.input_path), Path(imgProcessConf.output_path))

if __name__ == "__main__":
    main()