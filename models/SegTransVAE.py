import torch 
import torch.nn as nn
from models.Transformer import Transformer, FixedPositionalEncoding, LearnedPositionalEncoding 
from models.Encoder import Encoder
from models.Decoder import Decoder
from models.VAE import VAE

class SegTransVAE(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, num_classes, 
                embedding_dim, num_heads, num_layers, hidden_dim, in_channels_vae, 
                dropout = 0.0, attention_dropout = 0.0,
                conv_patch_representation = True, positional_encoding = 'learned',
                use_VAE = False):

        super().__init__()
        assert embedding_dim % num_heads == 0 
        assert img_dim[0] % patch_dim == 0 and img_dim[1] % patch_dim == 0 and img_dim[2] % patch_dim == 0 

        self.img_dim = img_dim 
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_dim = patch_dim    
        self.num_channels = num_channels 
        self.in_channels_vae = in_channels_vae
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.conv_patch_representation = conv_patch_representation
        self.use_VAE = use_VAE

        self.num_patches = int((img_dim[0] // patch_dim) * (img_dim[1] // patch_dim) * (img_dim[2] // patch_dim))
        self.seq_length = self.num_patches  
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.embedding_dim, self.seq_length
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
        self.bn = nn.InstanceNorm3d(128)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.FeatureMapping = FeatureMapping(in_channel = self.embedding_dim, out_channel= self.in_channels_vae)
        self.FeatureMapping1 = FeatureMapping1(in_channel = self.in_channels_vae)
        self.decoder = Decoder(self.img_dim, self.patch_dim, self.embedding_dim, num_classes)

        self.vae_input = (1, self.in_channels_vae, img_dim[0] // 8, img_dim[1] // 8, img_dim[2] // 8)
        if use_VAE:
            self.vae = VAE(input_shape = self.vae_input , latent_dim= 256, num_channels= self.num_channels)
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
#         print("In decode...")
#         print(" x1: {} \n x2: {} \n x3: {} \n x: {}".format( x1.shape, x2.shape, x3.shape, x.shape))
#         break
        return self.decoder(x1, x2, x3, x)

    def forward(self, x, is_validation = True):
        x1, x2, x3, x = self.encode(x)
        x = x.view( x.size(0), 
                    self.img_dim[0] // self.patch_dim,
                    self.img_dim[1] // self.patch_dim, 
                    self.img_dim[2] // self.patch_dim, 
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
    def __init__(self, in_channel, out_channel, norm = 'instance'):
        super().__init__()
        if norm == 'bn':
            norm_layer_1 = nn.BatchNorm3d(out_channel)
            norm_layer_2 = nn.BatchNorm3d(out_channel)
        elif norm == 'gn':
            norm_layer_1 = nn.GroupNorm(8, out_channel)
            norm_layer_2 = nn.GroupNorm(8, out_channel)
        elif norm == 'instance':
            norm_layer_1 = nn.InstanceNorm3d(out_channel)
            norm_layer_2 = nn.InstanceNorm3d(out_channel)
        self.feature_mapping = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size = 3, padding = 1),
            norm_layer_1,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size = 3, padding = 1),
            norm_layer_2,
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.feature_mapping(x)   


class FeatureMapping1(nn.Module):
    def __init__(self, in_channel, norm = 'instance'):
        super().__init__()
        if norm == 'bn':
            norm_layer_1 = nn.BatchNorm3d(in_channel)
            norm_layer_2 = nn.BatchNorm3d(in_channel)
        elif norm == 'gn':
            norm_layer_1 = nn.GroupNorm(8, in_channel)
            norm_layer_2 = nn.GroupNorm(8, in_channel)
        elif norm == 'instance':
            norm_layer_1 = nn.InstanceNorm3d(in_channel)
            norm_layer_2 = nn.InstanceNorm3d(in_channel)
        self.feature_mapping1 = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1),
            norm_layer_1,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channel, in_channel, kernel_size = 3, padding = 1),
            norm_layer_2,
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        y = self.feature_mapping1(x)
        return x + y    
    