from typing import Optional, overload

import torch
from torch.distributions import Normal
import pandas as pd

from cmmvae.modules.vae import VAE
from cmmvae.modules.base import FCBlockConfig, ConditionalLayers, ConcatBlockConfig


class CLVAE(VAE):
    """
    Conditional Latent Variational Autoencoder class.

    This class extends the basic VAE to incorporate conditional layers,
    allowing for conditioning the latent space on additional metadata.

    Args:
        encoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the encoder's fully connected block.
        decoder_config (cmmvae.modules.base.FCBlockConfig):
            Configuration for the decoder's fully connected block.
        conditional_config (Optional[cmmvae.modules.base.FCBlockConfig]):
            Configuration for the conditional layers.
        conditional_paths (Dict[str, str]):
            Mapping of conditional paths for the conditional layers.
        selection_order (Optional[List[str]]):
            Order in which to apply selection of conditionals.
        encoder_kwargs (dict): Additional keyword arguments for the encoder.
    """

    def __init__(
        self,
        encoder_config: FCBlockConfig,
        decoder_config: FCBlockConfig,
        conditional_config: Optional[FCBlockConfig] = None,
        conditionals_directory: Optional[str] = None,
        conditionals: Optional[list[str]] = None,
        selection_order: Optional[list[str]] = None,
        concat_config: Optional[ConcatBlockConfig] = None,
        **encoder_kwargs
    ):
        conditionals_module = None
        if conditional_config and conditionals and conditionals_directory:
            conditionals_module = ConditionalLayers(
                directory=conditionals_directory,
                conditionals=conditionals,
                fc_block_config=conditional_config,
                selection_order=selection_order,
            )
        else:
            import warnings
            warnings.warn("No conditionals found for CLVAE")

        if conditionals_module and selection_order and selection_order[0] == "parallel":
            if not concat_config or conditional_config is None:
                raise RuntimeError(
                    "Please define concat_config when selection_order = parallel"
                )
            concat_dim = (
                len(conditionals_module.selection_order) * conditional_config.layers[-1]
            )

            decoder_config.layers.insert(0, concat_dim)
            for attr in ['activation_fn', 'dropout_rate', 'return_hidden', 'use_layer_norm', 'use_batch_norm']:
                setattr(decoder_config, attr, getattr(concat_config, attr) + getattr(decoder_config, attr))

        super().__init__(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            **encoder_kwargs,
        )
        self.conditionals = conditionals_module

    def forward(self, x: torch.Tensor, metadata: pd.DataFrame, target_metadata: pd.DataFrame, **kwargs):
        qz, z, hidden_representations = self.encode(x, **kwargs)
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        z = self.after_reparameterize(z, target_metadata, **kwargs)
        xhat = self.decode(z, **kwargs)
        return qz, pz, z, xhat, hidden_representations

    def after_reparameterize(
        self, z: torch.Tensor, metadata: pd.DataFrame, **kwargs
    ) -> torch.Tensor:
        """
        Modify the latent variable after reparameterization
            by applying conditional layers.

        If conditional layers are defined, they will be applied to
            the latent variable `z` using the provided `metadata`.

        Args:
            z (torch.Tensor): Latent variable of shape (batch_size, n_latent).
            metadata (pd.DataFrame): Metadata associated with the input data.

        Returns:
            torch.Tensor:
                Processed latent variable after applying conditionals, if any.
        """
        if self.conditionals:
            return self.conditionals(z, metadata, **kwargs)
        # Return the unmodified latent variable
        # if no conditionals are present
        return z
