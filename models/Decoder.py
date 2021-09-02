import torch 
import torch.nn as nn


class Decoder_Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, out_channel, kernel_size = 1)
        self.deconv = nn.ConvTranspose3d(out_channel, out_channel, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv3d(out_channel * 2, out_channel, kernel_size = 1)

    def forward(self, prev, x):
        x = self.deconv(self.conv1(x))
        y = torch.cat((prev, x), dim = 1)
        return self.conv2(y)

class DecoderBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU(inplace = True)
        self.bn2 = nn.BatchNorm3d(in_channel)

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.relu2(self.bn2(self.conv2(x1)))
        return x1 + x

class Decoder(nn.Module):
    def __init__(self, img_dim, patch_dim, embedding_dim):
        super().__init__()
        self.img_dim = img_dim 
        self.patch_dim = patch_dim 
        self.embedding_dim = embedding_dim
    
        self.decoder_upsample_1 = Decoder_Upsample(self.embedding_dim // 4, self.embedding_dim // 8)
        self.decoder_block_1 = DecoderBlock(self.embedding_dim // 8)

        self.decoder_upsample_2 = Decoder_Upsample(self.embedding_dim // 8, self.embedding_dim // 16)
        self.decoder_block_2 = DecoderBlock(self.embedding_dim // 16)

        self.decoder_upsample_3 = Decoder_Upsample(self.embedding_dim // 16, self.embedding_dim // 32)
        self.decoder_block_3 = DecoderBlock(self.embedding_dim // 32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 3, kernel_size = 1)

    def forward(self, x1, x2, x3, x):

        x = self.decoder_upsample_1(x3, x)
        x = self.decoder_block_1(x) 

        x = self.decoder_upsample_2(x2, x)
        x = self.decoder_block_2(x)

        x = self.decoder_upsample_3(x1, x)
        x = self.decoder_block_3(x)

        y = self.endconv(x)
        return y 