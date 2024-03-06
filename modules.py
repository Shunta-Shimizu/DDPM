import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.double_conv(x)
        return x

class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
        self.emb_layer = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=out_channels),
            nn.SiLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels)
        )
    
    def forward(self, x, t):
        x = self.down_conv(x)
        # emb = self.emb_layer(t)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2, bias=True)
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)
        self.emb_layer = nn.Sequential(
            nn.Linear(in_features=emb_dim, out_features=out_channels),
            nn.SiLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels)
        )
    
    def forward(self, x, xi, t):
        x = self.up_conv(x)

        diffY = xi.size()[2] - x.size()[2]
        diffX = xi.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, xi], dim=1)
        x = self.double_conv(x)
        # emb = self.emb_layer(t)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

