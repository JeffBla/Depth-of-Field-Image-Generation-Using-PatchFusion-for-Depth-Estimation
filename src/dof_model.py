import pytorch_lightning as pl
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.models.vqvae import VQVAE
from src.models.pix2pix import Generator, Discriminator
from src.utils.loss_logger import log_imgs, log_losses

class DoFG(pl.LightningModule):
    def __init__(self, generatorConf: DictConfig):
        super().__init__()
        self.in_ch = generatorConf.in_channels
        self.out_ch = generatorConf.out_channels
        self.additional_decode_layer = generatorConf.additional_decode_layer
        self.additional_encode_layer = generatorConf.vqvae.additional_encode_layer
        self.codebook_size = generatorConf.vqvae.codebook_size
        self.codebook_len = generatorConf.vqvae.codebook_len
        self.lr = generatorConf.lr
        self.beta1 = generatorConf.beta1
        self.beta2 = generatorConf.beta2
        self.lambda_l1 = generatorConf.lambda_l1
        self.lambda_vq = generatorConf.lambda_vq
        self.lambda_gan = generatorConf.lambda_gan
        self.lambda_commit = generatorConf.lambda_commit

        self.save_hyperparameters()
        
        # Models
        self.vqvae = VQVAE(self.in_ch, self.out_ch , self.codebook_size, self.codebook_len, self.additional_encode_layer)
        self.generator = Generator(self.codebook_len, self.out_ch, self.additional_decode_layer)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        z_e = self.vqvae.encoder(x)
        z_e, z_q, decoder_input, perplexity = self.vqvae.vq(z_e)
        generated = self.generator(decoder_input)
        return generated, z_e, z_q, perplexity

    def training_step(self, batch, batch_idx, discriminator):
        '''
        @param batch: A tuple containing the input and target images:
            - real_A: Clean image & Depth image
            - real_B: Blur image
        '''
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
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.vqvae.parameters()) + list(self.generator.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        return optimizer
    
    def calculate_vqvae_losses(self, z_e: torch.Tensor, z_q: torch.Tensor) -> tuple:
        """
        Calculate VQ-VAE losses.
        
        Args:
            z_e (torch.Tensor): Encoder output.
            z_q (torch.Tensor): Quantized output.
        
        Returns:
            tuple: VQ loss and commitment loss.
        """
        vq_loss = self.mse_loss(z_q, z_e.detach())
        commitment_loss = self.mse_loss(z_q.detach(), z_e)
        return vq_loss, commitment_loss

    def calculate_adversarial_loss(self, real_A: torch.Tensor, generated: torch.Tensor, real_B: torch.Tensor, discriminator: nn.Module) -> tuple:
        """
        Calculate adversarial loss.
        
        Args:
            real_A (torch.Tensor): Input image.
            generated (torch.Tensor): Generated image.
            real_B (torch.Tensor): Target image.
            discriminator (nn.Module): Discriminator model.
        
        Returns:
            tuple: L1 loss and GAN loss.
        """
        l1_loss_value = self.l1_loss(generated, real_B)
        fake_pred = discriminator(torch.cat([real_A, generated], dim=1))
        real_target = torch.ones_like(fake_pred)
        gan_loss = self.mse_loss(fake_pred, real_target)
        return l1_loss_value, gan_loss

class DoFD(pl.LightningModule):
    def __init__(self, discriminatorConf: DictConfig):
        super().__init__()        
        self.discriminator = Discriminator(discriminatorConf.input_nc, discriminatorConf.ndf, discriminatorConf.n_layers)
        self.lr = discriminatorConf.lr
        self.beta1 = discriminatorConf.beta1
        self.beta2 = discriminatorConf.beta2
        
        self.save_hyperparameters()

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.discriminator(x)

    def training_step(self, batch, batch_idx, generator):
        '''
        @param batch: A tuple containing the input and target images:
            - real_A: Clean image & Depth image
            - real_B: Blur image
        '''
        real_A, real_B = batch
        
        # Get generated image from generator
        generated = generator(real_A)[0]

        # Real loss
        real_pred = self(torch.cat([real_A, real_B], dim=1))
        real_label = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_label)
        
        # Fake loss
        fake_pred = self(torch.cat([real_A, generated.detach()], dim=1))
        fake_target = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_target)
        
        # Total loss
        total_loss = (real_loss + fake_loss) * 0.5
        
        # Log losses
        wandb.log({'train/d_real_loss': real_loss,
                    'train/d_fake_loss': fake_loss, 
                    'train/d_total_loss': total_loss})
        
        return total_loss

    def validation_step(self, batch, batch_idx, generator):
        real_A, real_B = batch
        generated = generator(real_A)[0]
        
        # Calculate discriminator scores
        # Real loss
        real_pred = self(torch.cat([real_A, real_B], dim=1))
        real_label = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_label)
        
        # Fake loss
        fake_pred = self(torch.cat([real_A, generated.detach()], dim=1))
        fake_target = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_target)
        
        wandb.log({'val/real_loss': real_loss, 'val/fake_loss': fake_loss})
        
        return {
            'real_score': real_loss,
            'fake_score': fake_loss
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        return optimizer

# Training setup
class VQVAEPix2PixSystem(pl.LightningModule):
    def __init__(self, modelConf: DictConfig):
        super().__init__()
        self.generator = DoFG(modelConf.generator)
        self.discriminator = DoFD(modelConf.discriminator)
        self.automatic_optimization = False  # Disable automatic optimization

    def configure_optimizers(self):
        g_opt = self.generator.configure_optimizers()
        d_opt = self.discriminator.configure_optimizers()
        return [g_opt, d_opt]

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        
        # Train discriminator
        d_opt.zero_grad()
        d_loss = self.discriminator.training_step(batch, batch_idx, self.generator)
        self.manual_backward(d_loss)
        d_opt.step()

        # Train generator
        g_opt.zero_grad()
        g_loss = self.generator.training_step(batch, batch_idx, self.discriminator)
        self.manual_backward(g_loss)
        g_opt.step()

        loss_metrics =  {"total_g_loss": g_loss, "total_d_loss": d_loss}
        # mannual Log losses
        self._log_dict(loss_metrics, step_type="train")

        return loss_metrics

    def predict_step(self, batch, batch_idx):
        return self.generator.predict_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        # Get generator metrics
        generator_metrics = self.generator.validation_step(batch, batch_idx)
        
        # Get discriminator metrics
        discriminator_metrics = self.discriminator.validation_step(batch, batch_idx, self.generator)
        
        # Combine metrics
        metrics = {
            **generator_metrics,
            **discriminator_metrics
        }
        
        return metrics
    
    def _log_dict(self, metrics, step_type: str = "train"):
        """Helper method to manually log metrics"""
        on_step = step_type == "train"
        on_epoch = True
        
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach()
            self.log(
                f"{name}", 
                value, 
                on_step=on_step, 
                on_epoch=on_epoch, 
                prog_bar=True
            )