class loss_vae(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, sigma):
        mse = F.mse_loss(recon_x, x)
        kld = 0.5 * torch.mean(mu ** 2 + sigma ** 2 - torch.log(1e-8 + sigma ** 2) - 1)
        loss = mse + kld
        return loss