import torch 
import torch.nn as nn
from ResNetBlock import ResNetBlock

def calculate_total_dimension(a):
    res = 1 
    for x in a:
        res *= x 
    return res
 
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim, num_channels):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = input_shape[1]  #input_shape[0] is batch size
        self.latent_dim = latent_dim
        self.encoder_channels = self.in_channels // 16  

        #Encoder
        self.VAE_reshape = nn.Conv3d(self.in_channels, self.encoder_channels, 
                     kernel_size = 3, stride = 2, padding=1),

        flatten_input_shape =  calculate_total_dimension(input_shape)
        flatten_input_shape_after_vae_reshape = \
            flatten_input_shape * self.encoder_channels // (8 * self.in_channels)

        #Convert from total dimension to latent space
        self.to_latent_space = nn.Linear(
            flatten_input_shape_after_vae_reshape // self.in_channels, 1)

        self.mean = nn.Linear(self.in_channels, self.latent_dim)
        self.logvar = nn.Linear(self.in_channels, self.latent_dim)
        self.epsilon = nn.Parameter(torch.randn(1, latent_dim))

        #Decoder
        self.to_original_dimension = nn.Linear(self.latent_dim, flatten_input_shape_after_vae_reshape)
        self.Reconstruct = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.encoder_channels, self.in_channels,
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'),

            nn.Conv3d(
                self.in_channels, self.in_channels // 2, 
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'), 
            ResNetBlock(self.in_channels // 2), 

            nn.Conv3d(
                self.in_channels // 2, self.in_channels // 4, 
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'), 
            ResNetBlock(self.in_channels // 4), 

            nn.Conv3d(
                self.in_channels // 4, self.in_channels // 8, 
                stride = 1, kernel_size = 1),
            nn.Upsample(scale_factor=2, mode = 'nearest'), 
            ResNetBlock(self.in_channels // 8), 

            nn.GroupNorm(8, self.in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.in_channels // 8, num_channels, 
                kernel_size = 1, stride = 1)
        )


    def forward(self, x):   #x has shape = input_shape
        #Encoder
        x = self.VAE_reshape(x) 
        shape = x.shape

        x = x.view(self.in_channels, -1)
        x = self.to_latent_space(x)
        x = x.view(1, self.in_channels)

        mean = self.mean(x)
        logvar = self.logvar(x)
        sigma = torch.exp(0.5 * logvar)
        sample = mean + self.epsilon * torch.exp(logvar)

        #Decoder
        y = self.to_original_dimension(sample)
        y = y.view(*shape)
        return self.Reconstruct(y), mean, sigma
if __name__ == "__main__":
    x = torch.rand((1, 128, 16, 16, 16))
    vae = VAE(input_shape = x.shape, latent_dim = 128, num_channels = 4)
    y = vae(x)
    print(y.shape)


         

