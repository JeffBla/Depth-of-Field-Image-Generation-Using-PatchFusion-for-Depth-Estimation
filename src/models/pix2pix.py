import torch
import torch.nn as nn
import torch.nn.functional as F

class downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, apply_batchnorm=True):
        super().__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch) if apply_batchnorm else nn.Identity()
        )

    def forward(self, x):
        return self.down(x)

class upsample_with_additional_layer(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, additional_layer: int, drop_out=False):
        super().__init__()
        up = [nn.ReLU(True), nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1), nn.BatchNorm2d(out_ch)]
        up += [nn.Dropout(0.5) if drop_out else nn.Identity()]
        for i in range(0, additional_layer):
            up += [nn.ReLU(True), nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch)]
        self.up = nn.Sequential(*up)

    def forward(self, x):
        return self.up(x)

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, additional_decode_layer):
        super().__init__()

        additional_layer = [0, 0, 0, 0]
        for i in range(0, 4):
            additional_layer[i] = additional_decode_layer // 4 + (1 if additional_decode_layer % 4 > i else 0)
        
        self.down_1 = nn.Conv2d(in_ch, 64, 4, 2, 1) #256*256,ch64 -> 128*128,64ch 
        self.down_2 = downsample(64, 128) #128*128,64ch -> 64*64,128ch
        self.down_3 = downsample(128, 256) #64*64,128ch -> 32*32,256ch
        self.down_4 = downsample(256, 512) #32*32,256ch -> 16*16,512ch
        self.down_5 = downsample(512, 512) #16*16,512ch -> 8*8,512ch
        self.down_6 = downsample(512, 512) #8*8,512ch -> 4*4,512ch
        self.down_7 = downsample(512, 512) #4*4,512ch -> 2*2,512ch
        self.down_8 = downsample(512, 512, False) #2*2,512ch -> 1*1,512ch
        
        self.up_1 = upsample_with_additional_layer(512, 512, 0) #1*1,512ch -> 2*2,512ch
        self.up_2 = upsample_with_additional_layer(1024, 512, 0, True) #2*2,1024ch -> 4*4,512ch
        self.up_3 = upsample_with_additional_layer(1024, 512, 0, True) #4*4,1024ch -> 8*8,512ch
        self.up_4 = upsample_with_additional_layer(1024, 512, additional_layer[0]) #8*8,1024ch -> 16*16,512ch
        self.up_5 = upsample_with_additional_layer(1024, 256, additional_layer[1]) #16*16,1024ch -> 32*32,256ch
        self.up_6 = upsample_with_additional_layer(512, 128, additional_layer[2]) #32*32,512ch -> 64*64,128ch
        self.up_7 = upsample_with_additional_layer(256, 64, additional_layer[3]) #64*64,256ch -> 128*128,64ch

        self.last_Conv = nn.Sequential(
            upsample_with_additional_layer(128, 64, 0), #128*128,128ch -> 256*256,64ch
            upsample_with_additional_layer(64, 32, 0), #256*256,64ch -> 512*512,32ch
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1), #512*512,32ch -> 1024*1024,3ch
            nn.Tanh()
        )

        self.init_weight()
    
    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)
        down_5 = self.down_5(down_4)
        down_6 = self.down_6(down_5)
        down_7 = self.down_7(down_6)
        down_8 = self.down_8(down_7)

        up_1 = self.up_1(down_8)
        up_2 = self.up_2(torch.cat([up_1, down_7], dim=1))
        up_3 = self.up_3(torch.cat([up_2, down_6], dim=1))
        up_4 = self.up_4(torch.cat([up_3, down_5], dim=1))
        up_5 = self.up_5(torch.cat([up_4, down_4], dim=1))
        up_6 = self.up_6(torch.cat([up_5, down_3], dim=1))
        up_7 = self.up_7(torch.cat([up_6, down_2], dim=1))
        out = self.last_Conv(torch.cat([up_7, down_1], dim=1))
        return out

class CBLR(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()

        self.cblr = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True)
        )
    
    def forward(self, x):
        return self.cblr(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc=7, ndf=64, n_layers=3):
        super().__init__()
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [CBLR(ndf * nf_mult_prev, ndf * nf_mult, stride=2)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [CBLR(ndf * nf_mult_prev, ndf * nf_mult, stride=2)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)