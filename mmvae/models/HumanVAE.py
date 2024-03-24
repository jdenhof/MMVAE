import torch
import torch.nn as nn
import mmvae.models as M
import mmvae.models.utils as utils

class HumanExpert(M.Expert):
    
    input_dim = 60664
    class Encoder(nn.Module):
        def __init__(self, hidden_dim: int, latent_dim: int):
            super(HumanExpert.Encoder, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(HumanExpert.input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_dim, latent_dim),
                nn.LeakyReLU(),
            )
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x
            
    class Decoder(nn.Module):
        def __init__(self, hidden_dim: int, latent_dim: int):
            super(HumanExpert.Decoder, self).__init__()
            self.fc1 = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(hidden_dim, HumanExpert.input_dim),
                nn.LeakyReLU(),
            )
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            return x 
        
    def __init__(self, hidden_dim: int, latent_dim: int):
        super(HumanExpert, self).__init__(
            HumanExpert.Encoder(hidden_dim, latent_dim), 
            HumanExpert.Decoder(hidden_dim, latent_dim))
        
class HumanVAE(M.VAE):
    class Encoder(nn.Sequential):
        def __init__(self, input_dim: int, hidden_dim: int):
            super(HumanVAE.Encoder, self).__init__(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
            )
    class Decoder(nn.Sequential):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super(HumanVAE.Decoder, self).__init__(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.LeakyReLU()
            )
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(HumanVAE, self).__init__(
            HumanVAE.Encoder(input_dim, hidden_dim),
            HumanVAE.Decoder(input_dim, hidden_dim, latent_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.Linear(hidden_dim, latent_dim, )
        )
                
class Model(nn.Module):

    def __init__(self, expert: HumanExpert, shared_vae: HumanVAE):
        super().__init__()
        
        self.expert = expert
        self.shared_vae = shared_vae

    def forward(self, x: torch.Tensor):
        x = self.expert.encoder(x)
        x, mu, logvar, z = self.shared_vae(x)
        x = self.expert.decoder(x)
        return x, mu, logvar, z
    
def configure_model(trainer) -> Model:
    model = Model(
            HumanExpert(512, 256),
            HumanVAE(256, 128, 32),
        )
    if trainer.hparams['init_weights']:
        model.apply(utils.init_kaiming_normal)
    
    return model.to(trainer.device)

        
