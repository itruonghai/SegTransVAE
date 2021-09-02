import torch 
import torch.nn as nn
from models.Transformer import Transformer, FixedPositionalEncoding, LearnedPositionalEncoding 
from models.Encoder import Encoder
from models.Decoder import Decoder
from models.VAE import VAE
# class loss_rectr(nn.Module):
#     def __init__(self):
#         super(loss_function, self).__init__()

#     def forward(self, recon_x, x, mu, sigma):
#         mse = F.mse_loss(recon_x, x)
#         kld = 0.5 * torch.mean(mu ** 2 + sigma ** 2 - torch.log(1e-8 + sigma ** 2) - 1)
#         loss = mse + kld
#         return loss
class TransformerBTS(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, num_classes, 
                embedding_dim, num_heads, num_layers, hidden_dim,
                dropout = 0.0, attention_dropout = 0.0,
                conv_patch_representation = True, positional_encoding = 'learned',
                use_VAE = False):

        super().__init__()
        assert embedding_dim % num_heads == 0 
        assert img_dim % patch_dim == 0 

        self.img_dim = img_dim 
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_dim = patch_dim    
        self.num_channels = num_channels 
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.conv_patch_representation = conv_patch_representation
        self.use_VAE = use_VAE

        self.num_patches = int((img_dim // patch_dim) * (img_dim // patch_dim) * (img_dim // patch_dim))
        self.seq_length = self.num_patches  
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.pe_dropout = nn.Dropout(self.dropout)

        self.transformer = Transformer(
            embedding_dim, num_layers, num_heads, embedding_dim // num_heads,  hidden_dim, dropout
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        
        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(128, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.encoder = Encoder(self.num_channels, 16)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace = True)
        self.FeatureMapping = FeatureMapping(in_channel = self.embedding_dim)
        self.FeatureMapping1 = FeatureMapping1(in_channel = self.embedding_dim // 4 )
        self.decoder = Decoder(self.img_dim, self.patch_dim, self.embedding_dim)
        self.vae = VAE(input_shape = (1, 128, 16, 16, 16) , latent_dim= 128, num_channels= self.num_channels)
    def encode(self, x):
        if self.conv_patch_representation:
            x1, x2, x3, x = self.encoder(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        x = self.transformer(x) 
        x = self.pre_head_ln(x)

        return x1, x2, x3, x


    def decode(self, x1, x2, x3, x):
        #x: (1, 4096, 512) -> (1, 16, 16, 16, 512)
        return self.decoder(x1, x2, x3, x)



    def forward(self, x, is_validation = True):
        x1, x2, x3, x = self.encode(x)
        x = x.view( x.size(0), 
                    self.img_dim // self.patch_dim,
                    self.img_dim // self.patch_dim, 
                    self.img_dim // self.patch_dim, 
                    self.embedding_dim)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        x = self.FeatureMapping(x)
        x = self.FeatureMapping1(x)  
        if self.use_VAE and not is_validation:
            vae_out, mu, sigma = self.vae(x)
        y = self.decode(x1, x2, x3, x)
        if self.use_VAE and not is_validation:
            return y, vae_out, mu, sigma
        else:
            return y
        



class FeatureMapping(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, in_channel // 4, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(in_channel // 4)
        self.relu1 = nn.ReLU(inplace= True)
        self.conv2 = nn.Conv3d(in_channel // 4, in_channel // 4, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(in_channel // 4)
        self.relu2 = nn.ReLU(inplace= True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x   

class FeatureMapping1(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(in_channel)
        self.relu1 = nn.ReLU(inplace= True)
        self.conv2 = nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(in_channel)
        self.relu2 = nn.ReLU(inplace= True)

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.relu2(self.bn2(self.conv1(x1)))
        return x1 + x    