import torch 
import torch.nn as nn
from models.ResNetBlock import ResNetBlock


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size = 1)
        self.deconv = nn.ConvTranspose3d(out_channel, out_channel, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(out_channel * 2, out_channel, kernel_size = 1)

    def forward(self, prev, x):
        x = self.deconv(self.conv1(x))
        y = torch.cat((prev, x), dim = 1)
        return self.conv2(y)

class FinalConv(nn.Module): # Input channels are equal to output channels
    def __init__(self, in_channels, out_channels=32, norm="instance"):
        super(FinalConv, self).__init__()
        if norm == "batch":
            norm_layer = nn.BatchNorm3d(num_features=in_channels)
        elif norm == "group":
            norm_layer = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        elif norm == 'instance':
            norm_layer = nn.InstanceNorm3d(in_channels)

        self.layer = nn.Sequential(
            norm_layer,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            )
    def forward(self, x):
        return self.layer(x)

class Decoder(nn.Module):
    def __init__(self, img_dim, patch_dim, embedding_dim, num_classes = 3):
        super().__init__()
        self.img_dim = img_dim 
        self.patch_dim = patch_dim 
        self.embedding_dim = embedding_dim
    
        self.decoder_upsample_1 = Upsample(128, 64)
        self.decoder_block_1 = ResNetBlock(64)

        self.decoder_upsample_2 = Upsample(64, 32)
        self.decoder_block_2 = ResNetBlock(32)

        self.decoder_upsample_3 = Upsample(32, 16)
        self.decoder_block_3 = ResNetBlock(16)

        self.endconv = FinalConv(16, num_classes)
        # self.normalize = nn.Sigmoid()

    def forward(self, x1, x2, x3, x):
        x = self.decoder_upsample_1(x3, x)
        x = self.decoder_block_1(x) 

        x = self.decoder_upsample_2(x2, x)
        x = self.decoder_block_2(x)

        x = self.decoder_upsample_3(x1, x)
        x = self.decoder_block_3(x)

        y = self.endconv(x)
        return y 