import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VAE(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=1e-3):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(), nn.Flatten())

        # Latent space
        self.fc_mu = nn.Linear(128 * (input_dim // 8) * (input_dim // 8),
                               latent_dim)
        self.fc_var = nn.Linear(128 * (input_dim // 8) * (input_dim // 8),
                                latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(
            latent_dim, 128 * (input_dim // 8) * (input_dim // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128,
                               64,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,
                               32,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,
                               input_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1), nn.Sigmoid())

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, self.hparams.input_dim // 8,
                   self.hparams.input_dim // 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        loss = recon_loss + kl_loss

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total loss
        loss = recon_loss + kl_loss

        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.hparams.learning_rate)
