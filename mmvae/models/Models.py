from typing import Tuple, Union, Any
import torch
import torch.nn as nn
import mmvae.models.utils as utils

class Expert(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, discriminator = None):
        super(Expert, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def forward(self, x):
        raise NotImplementedError()

class VAE(nn.Module):
    """
    The VAE class is a single expert/modality implementation. It's a simpler version of the
    MMVAE and functions almost indentically.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, mean: nn.Linear, var: nn.Linear) -> None:
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mean = mean
        self.fc_var = var
        
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        return self.fc_mean(x), self.fc_var(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor) -> Tuple[Union[Any, torch.Tensor], Any, Any, Tuple[Any]]:
        
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x = self.decode(z)
        
        return x, mu, log_var, z 