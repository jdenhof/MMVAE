from torch.nn.parallel import DistributedDataParallel as DDP
from utils import assert_field_exists
from torch.utils.data import DataLoader
import torch
import time
import os

_req = [
    '_epochs_run',
    '_local_rank',
    '_global_rank',
    '_loader',
    '_optimizer',
    '_save_every', 
    '_snapshot_path',
]

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str
    ) -> None:
        
        self._initialize(model, loader, optimizer, save_every, snapshot_path)

        def _set_rank(rank: int):
            return rank if rank is not None else -1
        
        self._local_rank = _set_rank(int(os.environ["LOCAL_RANK"]))
        self._global_rank = _set_rank(int(os.environ["RANK"]))

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(model.to(self._local_rank), device_ids=[self._local_rank])

        [assert_field_exists(self, field) for field in _req]

    def train(self, max_epochs: int):
        for epoch in range(self._epochs_run, max_epochs):
            epoch_time = time.time()
            self._run_epoch(epoch)
            print(f"Epoch ran in: {time.time() - epoch_time}")
            if self.local_rank == 0 and epoch % self._save_every == 0:
                self._save_snapshot(epoch)

    def _run_epoch(self, epoch):
        print(f"[GPU{self._global_rank}] Epoch {epoch}")
        for source in self._loader:
            source = source.to(self._local_rank)
            self._run_batch(source)

    def _run_batch(self, source):
        self._optimizer.zero_grad()
        output, mean, logvar = self.model(source)
        recon_loss = torch.nn.CrossEntropyLoss()(output, source.to_dense())
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kld
        loss.backward()
        self._optimizer.step()
    
    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self._epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self._epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self._snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _initialize(self, model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, save_every: int, snapshot_path: str):
        # Check if 'model' is an instance of torch.nn.Module
        if not isinstance(model, torch.nn.Module):
            raise TypeError("The 'model' must be an instance of torch.nn.Module.")

        # Check if 'loader' is an instance of DataLoader
        if not isinstance(loader, DataLoader):
            raise TypeError("The 'loader' must be an instance of DataLoader.")

        # Check if 'optimizer' is an instance of torch.optim.Optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("The 'optimizer' must be an instance of torch.optim.Optimizer.")

        # Check if 'save_every' is an integer and greater than 0
        if not isinstance(save_every, int) or save_every <= 0:
            raise ValueError("The 'save_every' must be a positive integer.")

        # Check if 'snapshot_path' is a string
        if not isinstance(snapshot_path, str):
            raise TypeError("The 'snapshot_path' must be a string.")
        
        self._loader, self._optimizer, self._save_every, self._snapshot_path = loader, optimizer, save_every, snapshot_path
        self._epochs_run = 0

    _epochs_run: int
    _local_rank: int
    _global_rank: int
    _loader: DataLoader
    _optimizer: torch.optim.Optimizer
    _save_every: int
    _snapshot_path: int
    