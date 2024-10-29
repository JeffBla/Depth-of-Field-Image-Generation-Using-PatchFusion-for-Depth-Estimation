import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.patchfusion import PatchFusion
from src.models.vae import VAE
from src.models.gan import Generator, Discriminator


class DoFG(nn.Module):
    def __init__(self, in_ch, out_ch, codebook_size, codebook_len):
        super().__init__()
        self.patchfusion = PatchFusion()
        self.vqvae = VQVAE(in_ch, out_ch, codebook_size, codebook_len)
        self.generator = Generator(codebook_len, out_ch)

    def forward(self, x):
        depth_map = self.patchfusion(x)
        concat_input = torch.cat([x, depth_map], dim=1)
        
        z_e = self.vqvae.encoder(concat_input)
        z_e, z_q, decoder_input, perplexity = self.vqvae.vq(z_e)
        generated = self.generator(decoder_input)
        return generated, z_e, z_q, perplexity
    
class DoFD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = Discriminator()

    def forward(self, x1, x2):
        loss = self.discriminator(x1, x2)
        return loss