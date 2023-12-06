import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim) -> None:
        super(Encoder, self).__init__()
        
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dims)])
        # if len(hidden_dims) > 1:
        #     for i in range(len(hidden_dims)-1):
        #         self.encoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.mean = nn.Linear(hidden_dims, latent_dim)
        self.var = nn.Linear(hidden_dims, latent_dim)

    def forward(self, x):
        for layer in self.encoder_layers:
            #print("Encoder", x)
            x = torch.relu(layer(x))
        return self.mean(x), self.var(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim) -> None:
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dims)])
        # if len(hidden_dims) > 1:
        #     for i in range(len(hidden_dims)-1):
        #         self.decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.output = nn.Linear(hidden_dims, output_dim)

    def forward(self, x):
        for layer in self.decoder_layers:
            #print("Decoder", x)
            x = torch.relu(layer(x))
        return torch.relu(self.output(x))

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder) -> None:
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterize(self, mean, var):
        #print("Mean", mean, "Var", var)
        eps = torch.randn_like(var)#.to(DEVICE)
        return mean + var*eps

    def forward(self, x):
        mean, logvar = self.Encoder(x)
        z = self.reparameterize(mean, torch.exp(0.5 * logvar))
        x_hat = self.Decoder(z)
        return x_hat, mean, logvar

class Model(VAE):
    def __init__(self):
        super().__init__(Encoder(60664, 512, 128), Decoder(128, 512, 60664))

