import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim

import pandas as pd
import cmmvae
from cmmvae.data.local import SpeciesManager

# --- Gradient Reversal Layer ---
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)

# --- Conditional Layer ---
class ConditionalLayer(nn.Module):
    """
    Routes each sample’s latent vector z through the branch corresponding
    to its metadata label.
    """
    def __init__(self, latent_dim, out_dim, condition_values):
        """
        Args:
            latent_dim (int): Dimension of latent vector z.
            out_dim (int): Output dimension for the branch.
            condition_values (list): List of all possible values for this metadata.
        """
        super(ConditionalLayer, self).__init__()
        # Create a branch (here a simple Linear layer) for each condition value.
        self.out_dim = out_dim
        self.branches = nn.ModuleDict({
            str(val): nn.Linear(latent_dim, out_dim) for val in condition_values
        })

    def forward(self, z, cond_labels):
        """
        Args:
            z (Tensor): [batch, latent_dim] latent representations.
            cond_labels (list or iterable): length=batch list of labels (as strings)
                                            corresponding to the condition.
        Returns:
            Tensor of shape [batch, out_dim] where each row is computed from the branch
            corresponding to the sample’s label.
        """
        device = z.device
        batch_size = z.size(0)
        out = torch.zeros(batch_size, self.out_dim, device=device)
        # Loop over each branch in this conditional layer.
        for cond in self.branches:
            # Find indices where the metadata label equals this branch key.
            indices = [i for i, label in enumerate(cond_labels) if str(label) == cond]
            if indices:
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                z_subset = z.index_select(0, idx)
                out_subset = self.branches[cond](z_subset)
                out[idx] = out_subset
        return out

# --- VAE with Metadata-Conditioned Branches ---
class VAE(nn.Module):
    def __init__(self, input_dim, encoder_dims, latent_dim, conditional_configs, decoder_dims):
        """
        Args:
            input_dim (int): Number of input features (e.g. 60,000 RNA-seq features).
            encoder_dims (list): Sizes for encoder hidden layers.
            latent_dim (int): Dimension of the latent space.
            conditional_configs (dict): Dictionary where keys are metadata field names and
                values are dicts with:
                    - 'values': list of possible values for that field.
                    - 'out_dim': the output dimension for that field’s branch.
            decoder_dims (list): Sizes for decoder hidden layers.
        """
        super(VAE, self).__init__()
        # Build encoder network.
        encoder_layers = []
        prev_dim = input_dim
        for h in encoder_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        self.latent_dim = latent_dim

        # Build conditional layers (one per metadata field).
        self.conditional_layers = nn.ModuleDict({
            key: ConditionalLayer(latent_dim, config['out_dim'], config['values'])
            for key, config in conditional_configs.items()
        })

        # The decoder takes the concatenated outputs of all conditional layers.
        total_cond_dim = sum([config['out_dim'] for config in conditional_configs.values()])
        decoder_layers = []
        prev_dim = total_cond_dim
        for h in decoder_dims:
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # Reconstruction layer.
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, metadata):
        """
        Args:
            x (Tensor): [batch, input_dim] input data.
            metadata (dict): Keys are metadata field names; values are lists of labels
                             (one per sample) for that field.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # For each metadata field, route z through the corresponding conditional branch.
        cond_outputs = []
        for key, cond_layer in self.conditional_layers.items():
            cond_labels = metadata[key]  # e.g. for 'organ', a list like ['heart', 'lung', ...]
            cond_out = cond_layer(z, cond_labels)
            cond_outputs.append(cond_out)
        z_cat = torch.cat(cond_outputs, dim=1)
        x_recon = self.decoder(z_cat)
        return x_recon, mu, logvar, z

# --- Discriminator for Adversarial Feedback ---
class Discriminator(nn.Module):
    def __init__(self, latent_dim, num_classes):
        """
        A simple classifier that predicts a metadata label from latent z.
        Args:
            latent_dim (int): Dimension of latent space z.
            num_classes (int): Number of classes for this metadata field.
        """
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z):
        return self.net(z)

# --- Training Loop ---
def train(directory: str = ""):
    loader = SpeciesManager(
        name="human",
        directory_path="/mnt/projects/debruinz_project/july2024_census_data/subset/",
        train_npz_masks="human_counts_.*.npz",
        train_metadata_masks="human_metadata_.*.npz",
    ).create_train_dataloader()

    with open("unique_condtions.json", "r") as f:
        unique_conditions = json.load(f)

    # Data and network dimensions.
    input_dim = 60530  # RNA-seq data: 60,000 features.
    encoder_dims = [2048, 1024]
    latent_dim = 128
    decoder_dims = [1024, 2048]

    def get_condtional_out_dim(column: str):
        size = len(unique_conditions[column])
        return
    conditional_out_dim = lambda size: 64 if size > 64 else 32 if size > 32 else 16
    conditional_configs = {
        col: {
            'values': unique_conditions[col],
            'out_dim': conditional_out_dim(unique_conditions[col])
        } for col in unique_conditions
    }

    # For adversarial feedback, create a discriminator for each metadata field.
    discriminators = nn.ModuleDict({
        key: Discriminator(latent_dim, len(config['values']))
        for key, config in conditional_configs.items()
    })

    # Loss weighting hyperparameters.
    beta = 1.0     # Weight for the KL divergence.
    alpha = 1.0    # Weight for adversarial loss.
    adv_lambda = 1.0  # Gradient reversal multiplier.

    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

    # Initialize VAE and its optimizer.
    vae = VAE(input_dim, encoder_dims, latent_dim, conditional_configs, decoder_dims).to(device)
    vae_optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Create optimizers for the discriminators.
    disc_optimizers = { key: optim.Adam(disc.parameters(), lr=learning_rate)
                         for key, disc in discriminators.items() }

    mse_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    batch_idx = 0
    num_batches = None
    for epoch in range(num_epochs):
        for (tensor, metadata, name) in loader:
            batch_idx += 1
            batch_idx += (epoch+1)*()
            x = tensor
            x = x.to(device)

            # --- Update Discriminators ---
            # Freeze the VAE when training the discriminators.
            vae.eval()
            with torch.no_grad():
                _, _, _, z = vae(x, metadata)
            for key, disc in discriminators.items():
                disc_optimizers[key].zero_grad()
                logits = disc(z)
                # Convert metadata labels to indices.
                config = conditional_configs[key]
                value_to_index = {val: i for i, val in enumerate(config['values'])}
                labels = torch.tensor([value_to_index[label] for label in metadata[key]], device=device)
                loss_disc = ce_loss_fn(logits, labels)
                loss_disc.backward()
                disc_optimizers[key].step()

            # --- Update VAE (including adversarial loss) ---
            vae.train()
            vae_optimizer.zero_grad()
            x_recon, mu, logvar, z = vae(x, metadata)
            recon_loss = mse_loss_fn(x_recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            adv_loss_total = 0.0
            for key, disc in discriminators.items():
                # Apply gradient reversal so that the VAE “fools” the discriminator.
                z_rev = grad_reverse(z, lambda_=adv_lambda)
                logits = disc(z_rev)
                config = conditional_configs[key]
                value_to_index = {val: i for i, val in enumerate(config['values'])}
                labels = torch.tensor([value_to_index[label] for label in metadata[key]], device=device)
                adv_loss_total += ce_loss_fn(logits, labels)

            total_loss = recon_loss + beta * kl_loss + alpha * adv_loss_total
            total_loss.backward()
            vae_optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{num_batches or 'unknown'}], "
                      f"Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, Adv: {adv_loss_total.item():.4f}")
        if epoch == 0:
            num_batches = batch_idx

if __name__ == "__main__":
    train()