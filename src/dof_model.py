import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.vqvae import VQVAE
from src.models.pix2pix import Generator, Discriminator

class DoFG(pl.LightningModule):
    def __init__(self, in_ch, out_ch, codebook_size, codebook_len, 
                 lr=2e-4, beta1=0.5, beta2=0.999,
                 lambda_l1=10.0, lambda_vq=1.0, lambda_gan=1.0, lambda_commit=0.25):
        super().__init__()
        self.save_hyperparameters()
        
        # Models
        self.vqvae = VQVAE(in_ch, out_ch, codebook_size, codebook_len)
        self.generator = Generator(codebook_len, out_ch)
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_l1 = lambda_l1
        self.lambda_vq = lambda_vq
        self.lambda_gan = lambda_gan
        self.lambda_commit = lambda_commit

    def forward(self, x):
        z_e = self.vqvae.encoder(x)
        z_e, z_q, decoder_input, perplexity = self.vqvae.vq(z_e)
        generated = self.generator(decoder_input)
        return generated, z_e, z_q, perplexity

    def training_step(self, batch, batch_idx):
        '''
        @param batch: A tuple containing the input and target images:
            - real_A: Clean image & Depth image
            - real_B: Blur image
        '''
        real_A, real_B = batch
        
        # Forward pass
        generated, z_e, z_q, perplexity = self(real_A)
        
        # VQ-VAE losses
        vq_loss = self.mse_loss(z_q, z_e.detach())
        commitment_loss = self.mse_loss(z_q.detach(), z_e)

        # Adversarial loss
        l1_loss = self.l1_loss(generated, real_B)
        fake_pred = self.trainer.get_model('discriminator')(real_A, generated)
        real_target = torch.ones_like(fake_pred)
        gan_loss = self.criterion(fake_pred, real_target)
        
        # Calculate total loss
        loss = (self.lambda_l1 * l1_loss + 
                self.lambda_vq * (vq_loss + self.lambda_commit * commitment_loss) +
                self.lambda_gan * gan_loss)
    
        # Log losses
        self.log('train/l1_loss', l1_loss)
        self.log('train/vq_loss', vq_loss)
        self.log('train/commitment_loss', commitment_loss)
        self.log('train/perplexity', perplexity)
        self.log('train/total_loss', loss)
        
        return loss

    def predict_step(self, batch, batch_idx):
        real_A, _ = batch  
        generated, _, _, _ = self(real_A)
        return generated

    def test_step(self, batch, batch_idx):
        real_A, real_B = batch
        
        # Forward pass and get outputs
        generated, z_e, z_q, perplexity = self(real_A)
        
        # Calculate test metrics
        test_l1_loss = self.l1_loss(generated, real_B)
        test_vq_loss = self.mse_loss(z_q, z_e.detach())
        test_commitment_loss = self.mse_loss(z_q.detach(), z_e)
        
        # Log test metrics
        self.log('test/l1_loss', test_l1_loss)
        self.log('test/vq_loss', test_vq_loss)
        self.log('test/commitment_loss', test_commitment_loss)
        self.log('test/perplexity', perplexity)
        
        # Save sample test images
        if batch_idx % 100 == 0:
            self.logger.experiment.add_images('test/real_A', real_A, self.current_epoch)
            self.logger.experiment.add_images('test/real_B', real_B, self.current_epoch)
            self.logger.experiment.add_images('test/generated', generated, self.current_epoch)
        
        return {
            'test_l1_loss': test_l1_loss,
            'test_vq_loss': test_vq_loss,
            'test_commitment_loss': test_commitment_loss,
            'test_perplexity': perplexity,
            'generated_images': generated
        }
    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.vqvae.parameters()) + list(self.generator.parameters()),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        return optimizer

class DoFD(pl.LightningModule):
    def __init__(self, lr=2e-4, beta1=0.5, beta2=0.999):
        super().__init__()
        self.save_hyperparameters()
        
        self.discriminator = Discriminator(7, 64, 3)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x1, x2):
        return self.discriminator(x1, x2)

    def training_step(self, batch, batch_idx):
        '''
        @param batch: A tuple containing the input and target images:
            - real_A: Clean image & Depth image
            - real_B: Blur image
        '''
        real_A, real_B = batch
        
        # Get generated image from generator
        generated = self.trainer.get_model('generator')(real_A)[0]

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
        self.log('train/d_real_loss', real_loss)
        self.log('train/d_fake_loss', fake_loss)
        self.log('train/d_total_loss', total_loss)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        real_A, real_B = batch
        generated = self.trainer.get_model('generator')(real_A)[0]
        
        # Calculate discriminator scores
        # Real loss
        real_pred = self(real_A, real_B)
        real_label = torch.ones_like(real_pred)
        real_loss = self.criterion(real_pred, real_label)
        
        # Fake loss
        fake_pred = self(real_A, generated.detach())
        fake_target = torch.zeros_like(fake_pred)
        fake_loss = self.criterion(fake_pred, fake_target)
        
        self.log('test/real_loss', real_loss)
        self.log('test/fake_loss', fake_loss)
        
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
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def configure_optimizers(self):
        g_opt = self.generator.configure_optimizers()
        d_opt = self.discriminator.configure_optimizers()
        return [g_opt, d_opt]

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            return self.generator.training_step(batch, batch_idx)
        else:
            return self.discriminator.training_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.generator.predict_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        # Get generator metrics
        generator_metrics = self.generator.test_step(batch, batch_idx)
        
        # Get discriminator metrics
        discriminator_metrics = self.discriminator.test_step(batch, batch_idx)
        
        # Combine metrics
        metrics = {
            **generator_metrics,
            **discriminator_metrics
        }
        
        return metrics