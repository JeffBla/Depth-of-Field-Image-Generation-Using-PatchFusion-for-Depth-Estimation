from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
from PIL import Image
import numpy as np
import pandas as pd

def load_images_to_dict(folder_path):
    image_dict = {}
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            image = np.array(Image.open(file_path))
            image_dict[file_name] = image
        except Exception as e:
            print(f"無法讀取檔案 {file_name}: {e}")
    return image_dict

def main():
    truth_folder = "./data/test/truth"
    generated_folder = "./data/test/generated"
    software_folder = "./data/test/software"

    truth_dict = load_images_to_dict(truth_folder)
    generated_dict = load_images_to_dict(generated_folder)
    software_dict = load_images_to_dict(software_folder)

    results = []
    for key in truth_dict.keys():
        clean_image = truth_dict[key]
        generated_image = generated_dict[key]
        software_image = software_dict[key]

        model_psnr_value = PSNR(clean_image, generated_image, data_range=255)
        model_ssim_value = SSIM(clean_image, generated_image, multichannel = True, channel_axis = 2, data_range=255)

        software_psnr_value = PSNR(clean_image, software_image, data_range=255)
        software_ssim_value = SSIM(clean_image, software_image, multichannel = True, channel_axis = 2, data_range=255)

        results.append({"key": key, "model_psnr": model_psnr_value, "software_psnr": software_psnr_value, "model_ssim": model_ssim_value, "software_ssim": software_ssim_value})

    df = pd.DataFrame(results)
    output_path = "./data/test/image_metrics.xlsx"
    df.to_excel(output_path, index=False)
    
if __name__ == "__main__":
    main()
    