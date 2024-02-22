import torch
import torch.nn as nn
import torch.nn.functional as F
import d_mmvae.models as M

class Model(nn.Module):

    def __init__(self, expert: M.Expert, solo_vae: M.VAE):
        super().__init__()
        
        self.expert = expert
        self.solo_vae = solo_vae

    def forward(self, train_input: torch.Tensor):
        shared_input = self.expert.encoder(train_input)
        
        re_param, mu, var, shared_output = self.solo_vae(shared_input)
        
        expert_output = self.expert.decoder(shared_output)
        return re_param, mu, var, expert_output
    
class SingleEncoder(nn.Module):
    def __init__(self):
        super(SingleEncoder, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

class SingleDecoder(nn.Module):
    def __init__(self):
        super(SingleDecoder, self).__init__()
        self.fc1 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x
    
def configure_model() -> Model:
        return Model(
            M.Expert(
                nn.Sequential(
                        nn.Linear(60664, 8192),
                        nn.ReLU(),
                        nn.Linear(8192, 2048),
                        nn.ReLU(),
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                ),
                nn.Sequential( 
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 8192),
                    nn.ReLU(),
                    nn.Linear(8192, 60664),
                    nn.ReLU(),
                )
            ), 
            M.VAE(
                SingleEncoder(),
                SingleDecoder(),
                nn.Linear(128, 32),
                nn.Linear(128, 32)
            )
        )
     