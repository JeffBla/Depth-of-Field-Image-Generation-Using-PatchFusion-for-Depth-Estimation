import torch
from torchvision.transforms.functional import gaussian_blur

from src.dof_model import DoFG, DoFD, VQVAEPix2PixSystem
from src.utils.loss_logger import log_imgs, log_losses

class DoFGEnhanced(DoFG):
    def __init__(self, generatorConf):
        super().__init__(generatorConf)

    def enhance_image(self, image, generated):
        """
        Enhance the image using Gaussian smooth filter and the generated image, optimized for CUDA.

        :param image: Input image tensor of shape [B, C, H, W] (on CUDA)
        :param generated: Tensor of shape [B, 3+C, H, W] containing kernel size, sigma, and enhancement (on CUDA)
        :return: Enhanced image of shape [B, C, H, W] (on CUDA)
        """
        # Extract kernel size, sigma, and enhanced channel
        kernel_size = generated[:, 0]  # [B, H, W]
        sigma = generated[:, 1]  # [B, H, W]
        enhanced_channel = generated[:, 2:]  # [B, C, H, W]

        # Ensure kernel size is odd & nature number
        kernel_size = torch.ceil(kernel_size).int().abs()  # Ensure kernel_size is an naturl number
        kernel_size = kernel_size + (kernel_size % 2 == 0).int()  # Make it odd

        # Initialize the enhanced image tensor
        enhanced_image = torch.zeros_like(image)

        # Handle each kernel size in a batch-wise manner
        unique_kernel_sizes = torch.unique(kernel_size)
        for k_size in unique_kernel_sizes:
            k_size = k_size.item()  # Get scalar kernel size

            # Mask for the current kernel size
            mask = kernel_size == k_size
            if not mask.any():
                continue

            # Apply Gaussian blur to the entire image
            sig = sigma[mask].mean().item()
            blurred_image = gaussian_blur(image, (k_size, k_size), sig)

            # Indices of pixels matching the kernel size
            batch_indices, height_indices, width_indices = mask.nonzero(as_tuple=True)

            # Aggregate enhanced images
            enhanced_image[batch_indices, :, height_indices, width_indices] = blurred_image[batch_indices, :, height_indices, width_indices] +enhanced_channel[batch_indices, :, height_indices, width_indices]
            
        return enhanced_image

    def forward(self, x):
        generated, z_e, z_q, perplexity = super().forward(x)
        clean_img = x[:, :-1]
        enhanced = self.enhance_image(clean_img, generated)
        return enhanced, z_e, z_q, perplexity

    def training_step(self, batch, batch_idx, discriminator):
        real_A, real_B = batch
        
        # Forward pass
        generated, z_e, z_q, perplexity = self(real_A)
        
        # VQ-VAE losses
        vq_loss, commitment_loss = self.calculate_vqvae_losses(z_e, z_q)

        # Adversarial loss
        l1_loss, gan_loss = self.calculate_adversarial_loss(real_A, generated, real_B, discriminator)
           
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
        return generated

    def validation_step(self, batch, batch_idx):
        real_A, real_B = batch
        
        # Forward pass and get outputs
        generated, z_e, z_q, perplexity = self(real_A)
        
        # Calculate test metrics
        val_l1_loss = self.l1_loss(generated, real_B)
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
                'generated':generated
            }, step_type='val')

        return {
            'val_l1_loss': val_l1_loss,
            'val_vq_loss': val_vq_loss,
            'val_commitment_loss': val_commitment_loss,
            'val_perplexity': perplexity,
            'generated_images': generated
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