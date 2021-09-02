import torch 
import torch.nn as nn
from TransBTS.Transformer import Transformer, FixedPositionalEncoding, LearnedPositionalEncoding 
from TransBTS.Encoder import Encoder
from TransBTS.Decoder import Decoder
class TransformerBTS(nn.Module):
    def __init__(self, img_dim, patch_dim, num_channels, num_classes, 
                embedding_dim, num_heads, num_layers, hidden_dim,
                dropout = 0.0, attention_dropout = 0.0,
                conv_patch_representation = True, positional_encoding = 'learned'):

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
        self.decoder = Decoder(self.img_dim, self.patch_dim, self.embedding_dim)

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



    def forward(self, x):
        x1, x2, x3, x = self.encode(x)
        y = self.decode(x1, x2, x3, x)
        return y

