import torch
from torch import nn

class PercentLayer(nn.Module):
    def __init__(self, CTG, WATER, AIR):
        super().__init__()
        self.CTG = CTG
        self.WATER = WATER
        self.AIR = AIR
    """3 input. 1 output"""
    def forward(self, x):

        x = torch.softmax(x, 1)

        ct = self.CTG * x[:, 0] + self.WATER * x[:, 1] + self.AIR * x[:, 2]
        ct = ct.unsqueeze(1)

        rescale = (ct - self.AIR)  / (3000 - self.AIR)
        rescale = 2 * rescale - 1
        
        # tanh = nn.Tanh()
        # output = tanh(x)  # add a non-linear function
        return rescale, x
