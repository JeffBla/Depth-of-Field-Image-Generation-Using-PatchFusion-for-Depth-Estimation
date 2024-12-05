import cv2
import numpy as np
import torch
import wandb
from concurrent.futures import ThreadPoolExecutor

from src.dof_model import DoFG, VQVAEPix2PixSystem

class DoFGEnhanced(DoFG):
    def __init__(self, generatorConf):
        super().__init__(generatorConf)

    def enhance_pixel(self, image, kernel_size, sigma, enhanced_channel, b, i, j):
        k_size = int(kernel_size[b, i, j])
        if k_size % 2 == 0:
            k_size += 1  # Ensure the kernel size is odd
        # Ensure the sigma is non-negative
        sig = abs(sigma[b, i, j])
        half_k = k_size // 2
        # Ensure the indices are within bounds
        i_min, i_max = max(0, i - half_k), min(image.shape[1], i + half_k + 1)
        j_min, j_max = max(0, j - half_k), min(image.shape[2], j + half_k + 1)
        smoothed_pixel = cv2.GaussianBlur(image[b, i_min:i_max, j_min:j_max], (k_size, k_size), sig)
        return b, i, j, smoothed_pixel[half_k, half_k] + enhanced_channel[b, i, j]

    def enhance_image(self, image, generated):
        """
        Enhance the image using Gaussian smooth filter and the generated image.
        
        :param image: Input image
        :param generated: Generated image containing kernel size, sigma, and enhanced channel
        :return: Enhanced image
        """
        # Convert the image to a numpy array if it's a tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            # The image is in BCHW format, convert it to BHWC format
            image = np.transpose(image, (0, 2, 3, 1))

        # Convert the generated image to a numpy array if it's a tensor
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()

        # Extract kernel size, sigma, and enhanced channel from the generated image
        kernel_size = generated[:, 0]
        sigma = generated[:, 1]
        enhanced_channel = generated[:, 2:]

        # Create an empty array for the enhanced image
        enhanced_image = np.zeros_like(image)

        # Apply the Gaussian smooth filter to each pixel in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.enhance_pixel, image, kernel_size, sigma, enhanced_channel, b, i, j)
                for b in range(image.shape[0])
                for i in range(image.shape[1])
                for j in range(image.shape[2])
            ]
            for future in futures:
                b, i, j, enhanced_pixel = future.result()
                enhanced_image[b, i, j] = enhanced_pixel

        # Convert the enhanced image back to a tensor if needed
        if isinstance(enhanced_image, np.ndarray):
            # Convert the image back to BCHW format
            enhanced_image = np.transpose(enhanced_image, (0, 3, 1, 2))
            enhanced_image = torch.tensor(enhanced_image)

        return enhanced_image

    def training_step(self, batch, batch_idx, discriminator):
        real_A, real_B = batch
        
        # Forward pass
        generated, z_e, z_q, perplexity = self(real_A)
        
        # Enhance the generated image
        enhanced_generated = self.enhance_image(real_A, generated)
        
        # VQ-VAE losses
        vq_loss = self.mse_loss(z_q, z_e.detach())
        commitment_loss = self.mse_loss(z_q.detach(), z_e)

        # Adversarial loss
        l1_loss = self.l1_loss(enhanced_generated, real_B)
        fake_pred = discriminator(torch.cat([real_A, enhanced_generated], dim=1))
        real_target = torch.ones_like(fake_pred)
        gan_loss = self.criterion(fake_pred, real_target)
        
        # Calculate total loss
        loss = (self.lambda_l1 * l1_loss + 
                self.lambda_vq * (vq_loss + self.lambda_commit * commitment_loss) +
                self.lambda_gan * gan_loss)
    
        # Log losses
        wandb.log({'train/l1_loss': l1_loss, 
                    'train/vq_loss': vq_loss,
                    'train/commitment_loss': commitment_loss,
                    'train/perplexity': perplexity,
                    'train/total_g_loss': loss})
        
        return loss

    def predict_step(self, batch, batch_idx):
        real_A, _ = batch  
        generated, _, _, _ = self(real_A)
        enhanced_generated = self.enhance_image(real_A, generated)
        return enhanced_generated

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch
        
        # Forward pass and get outputs
        generated, z_e, z_q, perplexity = self(real_A)
        
        # Enhance the generated image
        enhanced_generated = self.enhance_image(real_A, generated)
        
        # Calculate test metrics
        val_l1_loss = self.l1_loss(enhanced_generated, real_B)
        val_vq_loss = self.mse_loss(z_q, z_e.detach())
        val_commitment_loss = self.mse_loss(z_q.detach(), z_e)
        
        # Log test metrics
        wandb.log({'val/l1_loss': val_l1_loss,
                    'val/vq_loss': val_vq_loss,
                    'val/commitment_loss': val_commitment_loss,
                    'val/perplexity': perplexity})
        
        # Save sample test images
        if batch_idx % 100 == 0:
            wandb.log({'val/img': [wandb.Image(real_A, "RGB", "real_A"),
                                   wandb.Image(real_B, "RGB", "real_B"),
                                   wandb.Image(enhanced_generated, "RGB", "generated")]})

        return {
            'val_l1_loss': val_l1_loss,
            'val_vq_loss': val_vq_loss,
            'val_commitment_loss': val_commitment_loss,
            'val_perplexity': perplexity,
            'generated_images': enhanced_generated
        }

class VQVAEPix2PixSystemEnhanced(VQVAEPix2PixSystem):
    def __init__(self, modelConf):
        super().__init__(modelConf)
        self.generator = DoFGEnhanced(modelConf.generator)

# Example usage
if __name__ == "__main__":
    model = VQVAEPix2PixSystemEnhanced()
    # Load an image and enhance it
    # image = load_image('path_to_image')
    # enhanced_image = model.enhance_image(image)
    # save_image('path_to_save_enhanced_image', enhanced_image)