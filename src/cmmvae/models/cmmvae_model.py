from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer  # type: ignore

from cmmvae.models import BaseModel
from cmmvae.modules import CMMVAE
from cmmvae.constants import REGISTRY_KEYS as RK
from cmmvae.modules.base.components import GradientReversalFunction
from cmmvae.config import AutogradConfig


class CMMVAEModel(BaseModel):
    """
    Conditional Multi-Modal Variational Autoencoder (CMMVAE) model for handling expert-specific data.

    This class is designed for training VAEs with multiple experts and adversarial components.

    Args:
        module (Any): Conditional Multi-Modal VAE module.
        batch_size (int, optional): Batch size for logging purposes only. Defaults to 128.
        record_gradients (bool, optional): Whether to record gradients of the model. Defaults to False.
        save_gradients_interval (int): Interval of steps to save gradients. Defaults to 25.
        gradient_record_cap (int, optional): Cap on the number of gradients to record to prevent clogging TensorBoard. Defaults to 20.
        kl_annealing_fn (KLAnnealingFn, optional): Annealing function used for kl_weight. Defaults to `KLAnnealingFn(1.0)`
        predict_dir (str): Directory to save predictions. If not absolute path then saved within Tensorboard log_dir. Defaults to "".
        predict_save_interval (int): Interval to save embeddings and metadata to prevent OOM Error. Defaults to 600.
        initial_save_index (int): The starting point for predictions index when saving (ie z_embeddings_0.npz for -1). Defaults to -1.
        use_he_init_weights (bool): Initialize weights using He initialization. Defaults to True.

    Attributes:
        module (`CMMVAE`): The CMMVAE module for processing and generating data.
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
        adversarial_criterion (nn.CrossEntropyLoss): Loss function for adversarial training.
        kl_annealing_fn (cmmvae.modules.base.KLAnnealingFn): KLAnnealingFn for weighting KL Divergence. Defaults to KLAnnealingFn(1.0).
    """

    def __init__(
        self,
        module: CMMVAE,
        adv_weight=None,
        adversarial_method="",
        autograd_config: Optional[AutogradConfig] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.module = module
        self.automatic_optimization = (
            False  # Disable automatic optimization for manual control
        )
        # Criterion for adversarial loss
        if adversarial_method == "GRF":
            adv_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        else:
            adv_criterion = nn.BCELoss(reduction="mean")

        self.adversarial_criterion = adv_criterion
        self.init_weights()
        self.adv_weight = adv_weight if adv_weight else 1.0
        self.adversarial_method = adversarial_method
        self.autograd_config = autograd_config or AutogradConfig()

    def grf(
        self,
        hidden_representations: list[torch.Tensor],
        expert_id: str,
        detach: bool = False,
    ):
        assert self.module.adversarials
        batch_size = hidden_representations[0].shape[0]
        label_value = 1.0 if expert_id == "human" else 0.0
        label = torch.full((batch_size,), label_value, device=self.device)

        adv_losses = []
        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, self.module.adversarials)
        ):
            if detach:
                hidden_rep = hidden_rep.detach()
            else:
                # Apply Gradient Reversal Function when updating the main network
                hidden_rep = GradientReversalFunction.apply(hidden_rep, 1)

            adv_output = adv(hidden_rep)

            # Calculate adversarial loss
            disc_loss = self.adversarial_criterion(adv_output.view(-1), label)
            adv_losses.append(disc_loss)

        self.auto_log(
            {f"layer_{i}": l for i, l in enumerate(adv_losses)},
            tags=[RK.ADV_LOSS, self.stage_name, expert_id],
            key_pos="last",
        )

        return torch.sum(torch.stack(adv_losses))

    def gradient_reversal_domain_classifier(
        self,
        hidden_representations,
        expert_id,
        adversarial_optimizers,
    ):
        # Compute adversarial loss for adversarial networks
        adv_loss = self.grf(hidden_representations, expert_id, detach=True)
        # Backpropagate adversarial loss to adversarial networks
        self.manual_backward(adv_loss)
        # Clip and update adversarial networks
        for optim in adversarial_optimizers.values():
            if self.autograd_config.adversarial_gradient_clip:
                self.clip_gradients(
                    optim, *self.autograd_config.adversarial_gradient_clip
                )
            optim.step()

        # Now compute adversarial loss for main network (with gradient reversal)
        adv_loss_main = self.grf(hidden_representations, expert_id, detach=False)
        # Add adversarial loss to total loss (with weight)
        return adv_loss_main

    def _adversarial_feedback(
            self,
            adversarial_group,
            hidden_representations: list[torch.Tensor],
            label: torch.Tensor,
            loss_dict: dict,
            generator: bool,
    ):
        """
        Calculate the loss for an adversarial group 
        """
        assert (
            len(hidden_representations)
            == len(adversarial_group.adversarials)
        )

        label = label.to(self.device)
        if generator:
            label = 1 - label

        losses = []
        for i, (hidden_rep, adv) in enumerate(
            zip(hidden_representations, adversarial_group.adversarials)
        ):
            # Get adversarial predictions
            if not generator:
                hidden_rep = hidden_rep.detach()

            adv = adv.to(self.device)

            adv_output = adv(hidden_rep)
            sum_loss = torch.nn.functional.binary_cross_entropy(adv_output, label.float(), reduction = "sum")
            # sum_loss = torch.nn.functional.cross_entropy(adv_output, label.float(), reduction = "sum")
            # current_discriminator_loss = self.adversarial_criterion(
            #     adv_output, label.float()
            # )

            losses.append(sum_loss)
            if not generator:
                self.manual_backward(sum_loss)
                self.step_adv_optimizers()
                mean_loss = torch.nn.functional.binary_cross_entropy(adv_output,label.float(), reduction = "mean")
                # mean_loss = torch.nn.functional.cross_entropy(adv_output, label.float(), reduction = "mean")
                loss_dict[RK.ADV_LOSS + adversarial_group.conditional + str(i)] = mean_loss
        
        return torch.stack(losses).sum()


    def adversarial_feedback(
        self,
        hidden_representations,
        labels,
        main_loss_dict
    ):
        
        assert self.module.adversarial_groups

        # for adv_group in self.module.adversarial_groups:
        #     self.zero_adv_optimizers(adv_group.conditional)

        adv_group_losses = []
        for adv_group in self.module.adversarial_groups:

            _ = self._adversarial_feedback(adv_group, hidden_representations,
                                              labels[adv_group.conditional], main_loss_dict, generator=False)
            # self.manual_backward(disc_losses)

            gen_loss = self._adversarial_feedback(adv_group, hidden_representations, 
                                              labels[adv_group.conditional], main_loss_dict, generator=True)

            adv_group_losses.append(gen_loss)
        # return 0
        return torch.stack(adv_group_losses).mean()

    def training_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ) -> None:
        x, metadata, expert_id, labels = batch

        # Get optimizers
        optims = self.get_optimizers()
        expert_optimizer = optims["experts"][expert_id]
        vae_optimizer = optims["vae"]

        ## TODO: don't push this to the repo
        adversarial_optimizers = None
        ##

        # Zero all gradients
        vae_optimizer.zero_grad()
        expert_optimizer.zero_grad()
        for adv_group in self.module.adversarial_groups:
            self.zero_adv_optimizers(adv_group.conditional)

        # Perform forward pass
        qz, pz, z, xhats, hidden_representations = self.module(
            x=x, metadata=metadata, expert_id=expert_id
        )

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        main_loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )
        total_loss = main_loss_dict[RK.LOSS]


        adv_loss = None
        if self.module.adversarial_groups and self.current_epoch >= 0:
            if self.adversarial_method == "GRF":
                adv_loss = self.gradient_reversal_domain_classifier(
                    hidden_representations + [z], expert_id, adversarial_optimizers
                )
            else:
                adv_loss = self.adversarial_feedback(
                    hidden_representations, labels, main_loss_dict
                )

        if adv_loss:
            main_loss_dict[RK.ADV_LOSS] = adv_loss
            total_loss = total_loss + (adv_loss * self.adv_weight)


        # Backpropagate main loss
        self.manual_backward(total_loss)
        # total_loss.backward()

        main_loss_dict[RK.LOSS] = total_loss

        self.log_gradient_norms(
            {"vae": vae_optimizer, f"expert_{expert_id}": expert_optimizer},
            tag_prefix="grad_norms/main_network",
        )

        if adversarial_optimizers:
            self.log_gradient_norms(
                adversarial_optimizers, tag_prefix="grad_norms/adversarials"
            )

        # Clip gradients for stability
        if self.autograd_config.vae_gradient_clip:
            self.clip_gradients(vae_optimizer, *self.autograd_config.vae_gradient_clip)

        if self.autograd_config.expert_gradient_clip:
            self.clip_gradients(
                expert_optimizer, *self.autograd_config.expert_gradient_clip
            )

        # Update the weights
        vae_optimizer.step()
        expert_optimizer.step()

        # for name,param in self.module.adversarial_groups[0].adversarials.named_parameters():
        #     print(f"{name}: {param.grad}")

        # if self.module.adversarial_groups:
        #     self.step_adv_optimizers()
        self.kl_annealing_fn.step()

        

        # Log the loss
        self.auto_log(main_loss_dict, tags=[self.stage_name, expert_id])

    def validation_step(self, batch: tuple[torch.Tensor, pd.DataFrame, str]):
        """
        Perform a single validation step.

        This step evaluates the model on a validation batch, logging losses.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
        """
        x, metadata, expert_id, _ = batch
        # expert_label = self.module.experts.labels[expert_id]

        # Perform forward pass and compute the loss
        qz, pz, z, xhats, hidden_representations = self.module(x, metadata, expert_id)

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        # Calculate reconstruction loss
        loss_dict = self.module.vae.elbo(
            qz, pz, x, xhats[expert_id], self.kl_annealing_fn.kl_weight
        )

        self.auto_log(loss_dict, tags=[self.stage_name, expert_id])

        if self.trainer.validating:
            self.log("val_loss", loss_dict[RK.LOSS], logger=False, on_epoch=True)

    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def predict_step(
        self, batch: tuple[torch.Tensor, pd.DataFrame, str], batch_idx: int
    ):
        """
        Perform a prediction step.

        This step extracts latent embeddings and saves them for analysis.

        Args:
            batch (tuple): Batch of data containing inputs, metadata, and expert ID.
            batch_idx (int): Index of the batch.
        """
        x, metadata, species, _ = batch
        embeddings = self.module.get_latent_embeddings(x, metadata, species)
        return embeddings
        # self.save_predictions(embeddings, batch_idx)

    def get_optimizers(self, zero_all: bool = False):
        """
        Retrieve optimizers for the model components.

        This function resets gradients if specified and returns a structured dictionary of optimizers.

        Args:
            zero_all (bool, optional): Flag to reset gradients of all optimizers. Defaults to False.

        Returns:
            dict: Dictionary containing optimizers for experts, VAE, and adversarials.
        """
        optimizers = self.optimizers()

        if zero_all:
            for optim in optimizers:  # type: ignore
                optim.zero_grad()

        def replace_indices_with_optimizers(mapping, optimizer_list):
            if isinstance(mapping, dict):
                return {
                    key: replace_indices_with_optimizers(value, optimizer_list)
                    for key, value in mapping.items()
                }
            else:
                return optimizer_list[mapping]

        # Create a dictionary with indices replaced with optimizer instances
        optimizer_dict = replace_indices_with_optimizers(self.optimizer_map, optimizers)

        return optimizer_dict

    def configure_optimizers(self, optim_cls="Adam") -> list[Optimizer]:  # type: ignore
        """
        Configure optimizers for different components of the model.

        Returns:
            list: List of configured optimizers for experts, VAE, and adversarials.
        """
        optim_cls = Adam if optim_cls == "Adam" else AdamW
        optim_dict = {}
        optim_dict["experts"] = {
            expert_id: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
            for expert_id, module in self.module.experts.items()
        }
        optim_dict["vae"] = optim_cls(
            self.module.vae.parameters(), lr=5e-3, weight_decay=1e-6
        )

        if self.module.adversarial_groups:
            for adversarial_group in self.module.adversarial_groups:
                optim_dict["adversarials" + adversarial_group.conditional ] = {
                    i: optim_cls(module.parameters(), lr=5e-3, weight_decay=1e-6)
                    for i, module in enumerate(adversarial_group.adversarials)
                }

        optimizers = []
        self.optimizer_map = convert_to_flat_list_and_map(optim_dict, optimizers)

        return optimizers

    def step_adv_optimizers(self):
        optimizers = self.get_optimizers()
        for group in self.module.adversarial_groups:
            for _, opt in optimizers["adversarials" + group.conditional].items():
                # if self.autograd_config.adversarial_gradient_clip:
                self.clip_gradients(opt, gradient_clip_val=1.00, gradient_clip_algorithm="value")
                opt.step()

    def zero_adv_optimizers(self, condition):
        optimizers = self.get_optimizers()
        for _, opt in optimizers["adversarials" + condition].items():
            opt.zero_grad()

def convert_to_flat_list_and_map(d: dict, flat_list: Optional[list] = None) -> dict:
    """
    Convert all values in the dictionary to a flat list and return the list and a mapping dictionary.
    Args:
        d (dict): The dictionary to convert.
        flat_list (list, optional): The list to append values to. Defaults to None.

    Returns:
        dict: Mapping dictionary linking keys to indices in the flat list.
    """
    if flat_list is None:
        flat_list = []

    map_dict = {}

    for key, value in d.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            map_dict[key] = convert_to_flat_list_and_map(value, flat_list)
        else:
            # Add value to flat list and set its index in the mapping
            flat_list.append(value)
            map_dict[key] = len(flat_list) - 1

    return map_dict
