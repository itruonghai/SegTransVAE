import torch 
import torch.nn as nn

#Re-use from encoder block
def normalization(planes, norm = 'instance'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'instance':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError("Does not support this kind of norm.")
    return m 
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, norm = 'instance'):
        super().__init__()
        self.resnetblock = nn.Sequential(
            normalization(in_channels, norm = norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1),
            normalization(in_channels, norm = norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        y = self.resnetblock(x)
        return y + x  
