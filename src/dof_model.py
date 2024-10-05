import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.patchfusion import PatchFusion
from src.models.vae import VAE
from src.models.gan import Generator, Discriminator


class DoFModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.patchfusion = PatchFusion()
        self.vae = VAE(input_dim=hparams.input_dim,
                       hidden_dim=hparams.hidden_dim,
                       latent_dim=hparams.latent_dim
                       )  # Assuming 3 color channels + 1 depth channel
        self.generator = Generator(hparams.generator)
        self.discriminator = Discriminator(hparams.discriminator)

        self.l1_loss = nn.L1Loss()

    def forward(self, x):
        depth_map = self.patchfusion(x)
        concat_input = torch.cat([x, depth_map], dim=1)
        compressed = self.vae.encode(concat_input)
        generated = self.generator(compressed)
        return generated, depth_map

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch

        # Generate fake images
        fake_images, depth_maps = self(real_images)

        # Train discriminator
        if optimizer_idx == 0:
            real_validity = self.discriminator(real_images, depth_maps)
            fake_validity = self.discriminator(fake_images.detach(),
                                               depth_maps)
            d_loss = (torch.mean(
                (real_validity - 1)**2) + torch.mean(fake_validity**2)) / 2
            self.log('train_d_loss', d_loss)
            return d_loss

        # Train generator
        if optimizer_idx == 1:
            fake_validity = self.discriminator(fake_images, depth_maps)
            g_loss = torch.mean((fake_validity - 1)**2)
            l1_loss = self.l1_loss(fake_images, real_images)
            total_g_loss = g_loss + self.hparams.lambda_l1 * l1_loss
            self.log('train_g_loss', total_g_loss)
            return total_g_loss

    def configure_optimizers(self):
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        opt_g = optim.Adam(list(self.patchfusion.parameters()) +
                           list(self.vae.parameters()) +
                           list(self.generator.parameters()),
                           lr=self.hparams.lr)
        return [opt_d, opt_g], []

    def on_train_epoch_end(self):
        self.logger.experiment.log({
            "epoch":
            self.current_epoch,
            "train_d_loss":
            self.trainer.callback_metrics["train_d_loss"],
            "train_g_loss":
            self.trainer.callback_metrics["train_g_loss"],
        })
