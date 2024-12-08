import torch
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def psnr(clean_images, generated_images):
    psnr_value = 0
    for clean, generated in zip(clean_images, generated_images):
        psnr_value += PSNR(clean, generated, data_range=255)
    psnr_value /= len(clean_images)
    return psnr_value

def ssim(clean_images, generated_images):
    ssim_value = 0
    for clean, generated in zip(clean_images, generated_images):
        ssim_value += SSIM(clean, generated, multichannel = True, channel_axis = 2, data_range=255)
    ssim_value /= len(clean_images)
    return ssim_value

if __name__ == "__main__":
    n = 10
    clean_images = [torch.rand(3, 1024, 1024) * 255 for _ in range(n)]
    generated_images = [torch.rand(3, 1024, 1024) * 255 for _ in range(n)]
    
    clean_images = [image.permute(1, 2, 0).numpy().astype('float32') for image in clean_images]
    generated_images = [image.permute(1, 2, 0).numpy().astype('float32') for image in generated_images]
    
    psnr_value = psnr(clean_images, generated_images)
    print(f"PSNR 值為: {psnr_value:.2f} dB")

    ssim_value = ssim(clean_images, generated_images)
    print(f"SSIM 值為: {ssim_value:.2f}")