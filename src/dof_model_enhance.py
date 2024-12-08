import torch
from torchvision.transforms import functional as vis_F
import wandb

from src.dof_model import DoFG, VQVAEPix2PixSystem
from src.utils.loss_logger import log_imgs, log_losses

class DoFGEnhanced(DoFG):
    def __init__(self, generatorConf):
        super().__init__(generatorConf)

    def enhance_pixel(self, clean_img, kernel_size, sigma, enhanced_channel, b, i, j):
        k_size = int(kernel_size[b,i, j])
        if k_size % 2 == 0:
            k_size += 1  # Ensure the kernel size is odd
        # Ensure the sigma is non-negative
        sig = abs(sigma[b, i, j])
        half_k = k_size // 2
        # Ensure the indices are within bounds
        i_min, i_max = max(0, i - half_k), min(clean_img.shape[2], i + half_k + 1)
        j_min, j_max = max(0, j - half_k), min(clean_img.shape[3], j + half_k + 1)
        smoothed_pixel = vis_F.gaussian_blur(clean_img[b, :, i_min:i_max, j_min:j_max], (k_size, k_size), sig.item())
        return smoothed_pixel.squeeze() + enhanced_channel[b,:, i, j]

    def enhance_image(self, image, generated):
        """
        Enhance the image using Gaussian smooth filter and the generated image.
        
        :param image: Input image
        :param generated: Generated image containing kernel size, sigma, and enhanced channel
        :return: Enhanced image
        """
        # Extract kernel size, sigma, and enhanced channel from the generated image
        kernel_size = generated[:, 0]
        sigma = generated[:, 1]
        enhanced_channel = generated[:, 2:]

        # Create an empty array for the enhanced image
        enhanced_image = torch.zeros_like(image)

        # Apply the Gaussian smooth filter to each pixel in parallel
        for b in range(image.shape[0]):
            for i in range(image.shape[2]):
                for j in range(image.shape[3]):
                    enhanced_pixel = self.enhance_pixel(image, kernel_size, sigma, enhanced_channel, b, i, j)
                    enhanced_image[b, :, i, j] = enhanced_pixel

        return enhanced_image

    def training_step(self, batch, batch_idx, discriminator):
        real_A, real_B = batch
        
        # Forward pass
        generated, z_e, z_q, perplexity = self(real_A)
        
        # Enhance the generated image
        enhanced_generated = self.enhance_image(real_A[:-1], generated)
        
     # VQ-VAE losses
        vq_loss, commitment_loss = self.calculate_vqvae_losses(z_e, z_q)

        # Adversarial loss
        l1_loss, gan_loss = self.calculate_adversarial_loss(real_A, enhanced_generated, real_B, discriminator)
           
        # Calculate total loss
        loss = (self.lambda_l1 * l1_loss + 
                self.lambda_vq * (vq_loss + self.lambda_commit * commitment_loss) +
                self.lambda_gan * gan_loss)
    
        # Log losses
        log_losses({
            'l1_loss': l1_loss, 
            'vq_loss': vq_loss,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'total_g_loss': loss
        }, step_type="train")
        
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
        clean_img = real_A[:,:-1]
        enhanced_generated = self.enhance_image(clean_img, generated)
        
        # Calculate test metrics
        val_l1_loss = self.l1_loss(enhanced_generated, real_B)
        val_vq_loss, val_commitment_loss = self.calculate_vqvae_losses(z_e, z_q)

        
        log_losses({
            'l1_loss': val_l1_loss,
            'vq_loss': val_vq_loss,
            'commitment_loss': val_commitment_loss,
            'perplexity': perplexity
        }, step_type="val")
        
        # Save sample images
        if batch_idx % 100 == 0:
            log_imgs({
                'real_A':real_A,
                'real_B':real_B,
                'generated':enhanced_generated
            }, step_type='val')

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