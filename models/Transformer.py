import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, function):
        super().__init__()
        self.norm = nn.LayerNorm(dim) 
        self.function = function 

    def forward(self, x):
        return self.function(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.0):
        super().__init__()
        all_head_size = heads * dim_head 
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads  
        self.scale = dim_head ** -0.5 

        self.softmax = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, all_head_size * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(all_head_size, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #(batch, heads * dim_head) -> (batch, all_head_size)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        dots =  torch.matmul(q, k.transpose(-1, -2)) * self.scale 

        atten = self.softmax(dots)

        out = torch.matmul(atten, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)), 
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    def forward(self, x):
        for attention, feedforward in self.layers:
            x = attention(x) + x 
            x = feedforward(x) + x  
        return x

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=768):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.seq_length = seq_length
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x

    def forward(self, x, position_ids=None):
        position_embeddings = self.position_embeddings
#         print(x.shape, self.position_embeddings.shape)
        return x + position_embeddings
