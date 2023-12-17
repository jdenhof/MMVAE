from Dataset import ThreadedChunkDataset
from utils import debug_attribute, attribute_initilized_error, isinstance_error, print, assert_fields_exists
import torch
import time
import os

# dev only
from utils import DEBUG

_static_req = (
    'dataset',
    '_device',
    '_optimizer',
    '_save_every', 
    '_snapshot_path',
)

_req = (
    *_static_req,
    'epoch',
)

class CellxGeneTrainer:

    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        dataset: ThreadedChunkDataset,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str
    ) -> None:
        
        self._initialize_fields(model, dataset, optimizer, save_every, snapshot_path, device)

        self.model = model.to(device)

        # TODO: GUUID TO NOT OVERRIDE BASE 
        if not DEBUG and os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        
        assert_fields_exists(self, *_req)
        self.__initialized = True

    def __setattr__(self, attr: str, val) -> None:
        if self.__initialized and attr in _static_req:
            attribute_initilized_error(self, attr)
        
        if DEBUG and attr in _req:
            debug_attribute(self, attr, val)
        super().__setattr__(attr, val)

    def train(self, max_epochs: int):
        if not self.dataset.ready:
            raise RuntimeError("Dataset is not ready for training!")
        
        for epoch in range(self.epoch, max_epochs):
            print(f"Starting epoch: {epoch}", 'epoch_start')
            epoch_time = time.time()
            self._run_epoch(epoch)
            print(f"Epoch ran in: {time.time() - epoch_time}")
            if epoch % self._save_every == 0:
                self._save_snapshot(epoch)

    def _run_epoch(self, epoch :int):
        print(f"[{self._device}] Epoch {epoch}")
        self.total_loss = 0
        for source in iter(self.dataset):
            # source = source.to(self.device) - No Longer needed as created on device
            self._run_batch(source)
        print(f"Average Loss: {self.total_loss / (epoch + 1):4f}")

    def _run_batch(self, source: torch.Tensor):
        self._optimizer.zero_grad()
        output, mean, logvar = self.model(source)
        start_time = time.time()
        recon_loss = torch.nn.MSELoss()(output, source.to_dense())
        self.mse_loss_time += time.time() - start_time
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kld
        loss.backward()
        self.total_loss += loss
        self._optimizer.step()
    
    def _load_snapshot(self, snapshot_path: str):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epoch}")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self._snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self._snapshot_path}")

    def _initialize_fields(self, model: torch.nn.Module, dataset: ThreadedChunkDataset, optimizer: torch.optim.Optimizer, save_every: int, snapshot_path: str, device: str):

        isinstance_error(model, torch.nn.Module)
        isinstance_error(dataset, ThreadedChunkDataset)
        isinstance_error(optimizer, torch.optim.Optimizer)
        isinstance_error(snapshot_path, str)
        isinstance_error(save_every, int) 
        if save_every <= 0:
            raise ValueError("The 'save_every' must be a positive integer.")
        
        self.dataset, self.optimizer, self._save_every, self._snapshot_path, self.device = dataset, optimizer, save_every, snapshot_path, device
        self.epoch = 0

    __initialized = False
    