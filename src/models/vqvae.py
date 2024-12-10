import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_ch, codebook_len):
        super().__init__()
        #1024*1024,4ch -> 512*512,16ch -> 256*256,32ch -> 256*256,64ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, codebook_len//4, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(codebook_len//4, codebook_len//2, 4, 2, 1),
            nn.ReLU()
        )
        self.pre_vq = nn.Conv2d(codebook_len//2, codebook_len, 3, 1, 1)
    
    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pre_vq(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #256*256,64ch -> 512*512,32ch -> 1024*1024,16ch -> 1024*1024,3ch
        self.post_vq = nn.ConvTranspose2d(in_ch, in_ch//2, 4, 2, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_ch//2, in_ch//4, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_ch//4, out_ch, 3, 1, 1)
        )
        
    def forward(self, inputs):
        x = self.post_vq(inputs)
        x = self.decoder(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, codebook_len):
        super(VectorQuantizer, self).__init__()
        self.codebook_len = codebook_len
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(self.codebook_size, self.codebook_len)
        self.embedding.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)
        
    def forward(self, z_e):
        input_shape = z_e.shape
        flat_input = z_e.view(-1, self.codebook_len)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                      + torch.sum(self.embedding.weight**2, dim=1)
                      - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z_e.device)
        encodings.scatter_(1, encoding_indices, 1)
        z_q = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, perplexity

class VQVAE(nn.Module):
    def __init__(self, in_ch, out_ch, codebook_size, codebook_len):
        super().__init__()
        self.encoder = Encoder(in_ch, codebook_len)
        self.vector_quantizer = VectorQuantizer(codebook_size, codebook_len)
        self.decoder = Decoder(codebook_len, out_ch)
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_e, z_q, decoder_input, perplexity = self.vq(z_e)
        x_recon = self.decoder(decoder_input)
        return x_recon, z_e, z_q, perplexity

    def vq(self, z_e):
        z_e = self.vq_preprocess(z_e)
        z_q, perplexity = self.vector_quantizer(z_e)
        decoder_input = self.vq_postprocess(z_e, z_q)
        return z_e, z_q, decoder_input, perplexity
        
    def vq_preprocess(self, z_e):
        return z_e.permute(0, 2, 3, 1).contiguous()
    
    def vq_postprocess(self, z_e, z_q):
        decoder_input = z_e + (z_q - z_e).detach()
        return decoder_input.permute(0, 3, 1, 2).contiguous()