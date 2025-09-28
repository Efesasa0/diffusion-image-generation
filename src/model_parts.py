import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_residual=False):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.is_same_channels = in_channels == out_channels
        self.is_residual = is_residual

        if is_residual and not self.is_same_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.shortcut = None
    
    def forward(self, x):

        out = self.conv(x)

        if not self.is_residual:
            return out
        
        if self.is_same_channels:
            out += x 
        else:
            out += self.shortcut(x)

        return out / np.sqrt(2) # Normalizing residual flow

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ResidualDoubleConv(out_channels, out_channels),
            ResidualDoubleConv(out_channels, out_channels),
        )
    
    def forward(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
        x = torch.cat((x, skip), 1)
        x = self.conv(x)

        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()

        # Diffusion nets handle residual connections inside DoubleConv
        self.conv = nn.Sequential(
            ResidualDoubleConv(in_channels, out_channels),
            ResidualDoubleConv(out_channels, out_channels),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):

        return self.conv(x)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(EmbedFC, self).__init__()

        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):

        x = x.view(-1, self.input_dim)
        x = self.fc(x)

        return x