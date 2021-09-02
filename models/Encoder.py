import torch 
import torch.nn as nn 
import torch.nn.functional as F 

def normalization(planes, norm = 'gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(8, planes)
    elif norm == 'instance':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError("Does not support this kind of norm.")
    return m 


class InitConv(nn.Module):
    def __init__(self, in_channels = 4, out_channels = 16, dropout = 0.2):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.dropout = nn.Dropout3d(dropout)
    def forward(self, x):
        return self.dropout(self.conv(x))

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, norm = 'gn'):
        super().__init__()
        self.bn1 = normalization(in_channels, norm = norm)
        self.relu1 = nn.ReLU(inplace= True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1)

        self.bn2 = normalization(in_channels, norm = norm)
        self.relu2 = nn.ReLU(inplace= True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size = 3, padding = 1)

    def forward(self, x):
        y = self.conv1(self.relu1(self.bn1(x)))
        y = self.conv2(self.relu2(self.bn2(y)))
        return y + x   

class EncoderDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, base_channels, dropout = 0.2):
        super().__init__()

        self.init_conv = InitConv(in_channels, base_channels, dropout = dropout)
        self.encoder_block1 = EncoderBlock(in_channels = base_channels)
        self.encoder_down1 = EncoderDown(base_channels, base_channels * 2)

        self.encoder_block2_1 = EncoderBlock(base_channels * 2)
        self.encoder_block2_2 = EncoderBlock(base_channels * 2)
        self.encoder_down2 = EncoderDown(base_channels * 2, base_channels * 4)

        self.encoder_block3_1 = EncoderBlock(base_channels * 4)
        self.encoder_block3_2 = EncoderBlock(base_channels * 4)
        self.encoder_down3 = EncoderDown(base_channels * 4, base_channels * 8)

        self.encoder_block4_1 = EncoderBlock(base_channels * 8)
        self.encoder_block4_2 = EncoderBlock(base_channels * 8)
        self.encoder_block4_3 = EncoderBlock(base_channels * 8)
        self.encoder_block4_4 = EncoderBlock(base_channels * 8)
        # self.encoder_down3 = EncoderDown(base_channels * 8, base_channels * 16)
    def forward(self, x):
        x = self.init_conv(x) #(1, 16, 128, 128, 128)

        x1 = self.encoder_block1(x) 
        x1_down = self.encoder_down1(x1)  #(1, 32, 64, 64, 64)

        x2 = self.encoder_block2_2(self.encoder_block2_1(x1_down)) 
        x2_down = self.encoder_down2(x2) #(1, 64, 32, 32, 32)

        x3 = self.encoder_block3_2(self.encoder_block3_1(x2_down))
        x3_down = self.encoder_down3(x3) #(1, 128, 16, 16, 16)

        output = self.encoder_block4_4(
                            self.encoder_block4_3(
                            self.encoder_block4_2(
                            self.encoder_block4_1(x3_down))))  #(1, 128, 16, 16, 16)
        return x1, x2, x3, output 

if __name__ == "__main__":   
    x = torch.rand((1, 4, 128, 128, 128))
    vae = Encoder(4, 32)
    _, _, _, y = vae(x)
    print(y.shape)


    
