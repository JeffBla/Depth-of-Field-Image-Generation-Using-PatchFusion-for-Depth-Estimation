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

class upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, drop_out=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.5) if drop_out else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.down_1 = nn.Conv2d(in_ch, 64, 4, 2, 1) #256*256,ch64 -> 128*128,64ch 
        self.down_2 = downsample(64, 128) #128*128,64ch -> 64*64,128ch
        self.down_3 = downsample(128, 256) #64*64,128ch -> 32*32,256ch
        self.down_4 = downsample(256, 512) #32*32,256ch -> 16*16,512ch
        self.down_5 = downsample(512, 512) #16*16,512ch -> 8*8,512ch
        self.down_6 = downsample(512, 512) #8*8,512ch -> 4*4,512ch
        self.down_7 = downsample(512, 512) #4*4,512ch -> 2*2,512ch
        self.down_8 = downsample(512, 512, False) #2*2,512ch -> 1*1,512ch
        
        self.up_1 = upsample(512, 512) #1*1,512ch -> 2*2,512ch
        self.up_2 = upsample(1024, 512, True) #2*2,1024ch -> 4*4,512ch
        self.up_3 = upsample(1024, 512, True) #4*4,1024ch -> 8*8,512ch
        self.up_4 = upsample(1024, 512) #8*8,1024ch -> 16*16,512ch
        self.up_5 = upsample(1024, 256) #16*16,1024ch -> 32*32,256ch
        self.up_6 = upsample(512, 128) #32*32,512ch -> 64*64,128ch
        self.up_7 = upsample(256, 64) #64*64,256ch -> 128*128,64ch

        self.last_Conv = nn.Sequential(
            upsample(128, 64), #128*128,128ch -> 256*256,64ch
            upsample(64, 32), #256*256,64ch -> 512*512,32ch
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
            nn.Conv2d(in_ch, out_ch, 4, stride, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.cblr(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminator = nn.Sequential(
            CBLR(6, 64, 2), #1024*1024,6ch -> 512*512,64ch
            CBLR(64, 128, 2), #512*512,64ch -> 256*256,128ch
            CBLR(128, 256, 2), #256*256,128ch -> 128*128,256ch
            CBLR(256, 512, 2), #128*128,256ch -> 64*64,512ch
            CBLR(512, 512, 1), #64*64,512ch -> 63*63,512ch
            nn.Conv2d(512, 1, 4, 1, 1) #63*63,512ch -> 62*62,1ch
        )

    def forward(self, x1, x2):
        in_x = torch.cat([x1, x2], dim=1)
        return self.discriminator(in_x)