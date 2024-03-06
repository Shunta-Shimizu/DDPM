import torch 
import torch.nn as nn
from modules import DoubleConv, DownSamplingBlock, UpSamplingBlock

class Simple_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(Simple_Unet, self).__init__()
        self.double_conv = DoubleConv(in_channels, 64)
        self.down_sampling1 = DownSamplingBlock(64, 128)
        self.down_sampling2 = DownSamplingBlock(128, 256)
        self.down_sampling3 = DownSamplingBlock(256, 512)
        self.down_sampling4 = DownSamplingBlock(512, 1024)

        self.up_sampling1 = UpSamplingBlock(1024, 512)
        self.up_sampling2 = UpSamplingBlock(512, 256)
        self.up_sampling3 = UpSamplingBlock(256, 128)
        self.up_sampling4 = UpSamplingBlock(128, 64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

        self.time_dim = time_dim

    def positional_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device="cuda").float() / channels)
        )
        channels //= 2
        pos_enc_a = torch.sin(t.repeat(1, channels) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.positional_encoding(t, self.time_dim)

        x0 = self.double_conv(x)
        x1 = self.down_sampling1(x0, t)
        x2 = self.down_sampling2(x1, t)
        x3 = self.down_sampling3(x2, t)
        x4 = self.down_sampling4(x3, t)

        x5 = self.up_sampling1(x4, x3, t)
        x6 = self.up_sampling2(x5, x2, t)
        x7 = self.up_sampling3(x6, x1, t)
        x8 = self.up_sampling4(x7, x0, t)
        x_out = self.out_conv(x8)

        return x_out