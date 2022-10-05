import torch
from tools.nn_base import Network
from typing import Union
from tools.nn_base import Network
from diffusers import DDPMScheduler, DDIMScheduler
import torch.nn.functional as F
from .diffusion_utils import HiddenDDIM, gaussian_kl


class DiagGaussian:
    def __init__(self):
        self.device = 'cuda:0'

    def sample(self, latents, context):
        return latents

    def sample_init_latents(self, batch_size, context):
        return torch.randn((batch_size, *self.latent_shape), device=self.device)

    def __call__(self, latent_input, context):
        #raise NotImplementedError
        mu, log_sigma = torch.chunk(latent_input, 2, dim=-1)
        sigma = log_sigma.exp()
        self.latent_shape = mu.shape[1:]
        return torch.randn_like(mu) * sigma + mu,   {'kl': gaussian_kl(mu, log_sigma)}




class VAE(Network):
    def __init__(self, encoder: Network, decoder: Network, latent: Union[HiddenDDIM, DiagGaussian], cfg=None) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent = latent

    def sample(self, batch_size, context):
        init_latent = self.latent.sample_init_latents(batch_size, context)
        return self.decoder(self.latent.sample(init_latent, context), context)
    
    def forward(self, input, context):
        latent_input = self.encoder(input, context)
        latent, losses = self.latent(latent_input, context)

        decoded = self.decoder(latent, context)
        assert decoded.shape == input.shape
        losses['recon'] = F.mse_loss(decoded, input)
        return decoded, losses