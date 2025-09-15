from src.model_parts import ResidualDoubleConv, UpSample, DownSample, EmbedFC
import torch.nn as nn
import torch

class ContextUnet(nn.Module):
    
    def __init__(self, in_channels, features=256, context_features=10, image_size=(16, 16)):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.features = features
        self.context_features = context_features
        self.height, self.width = image_size

        self.init_conv = ResidualDoubleConv(in_channels, features, is_residual=True)

        self.down1 = DownSample(features, features)
        self.down2 = DownSample(features, 2*features)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d((4)),
            nn.GELU(),
        )

        self.timeembed1 = EmbedFC(1, 2*features)
        self.timeembed2 = EmbedFC(1, 1*features)
        self.contextembed1 = EmbedFC(context_features, 2*features)
        self.contextembed2 = EmbedFC(context_features, 1*features)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*features, 2*features, self.height//4, self.height//4),
            nn.GroupNorm(8, 2*features),
            nn.ReLU(),
        )
        self.up1 = UpSample(4*features, features)
        self.up2 = UpSample(2*features, features)

        self.out = nn.Sequential(
            nn.Conv2d(2*features, features, 3, 1, 1),
            nn.GroupNorm(8, features),
            nn.ReLU(),
            nn.Conv2d(features, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.context_features).to(x)
        
        cemb1 = self.contextembed1(c).view(-1, self.features*2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.features*2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.features, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.features, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

