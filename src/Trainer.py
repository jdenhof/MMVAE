from DataLoader import CellCensusDataLoader
from utils import debug_attribute, attribute_initilized_error, isinstance_error, assert_fields_exists
import torch
import time
import os

# dev only
from utils import DEBUG

_init_req = (
    'model',
    'dataloader',
    'optimizer',
    'rank',
    '_save_every', 
    '_snapshot_path',
)

_watch = (
    *_init_req,
    'epoch',
)

class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: CellCensusDataLoader,
        rank: str,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str
    ) -> None:
        
        self._initialize_fields(model, dataloader, rank, optimizer, save_every, snapshot_path)

        # TODO: GUUID TO NOT OVERRIDE BASE FOR SNAPSHOT PATH
        if not DEBUG and os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        
        assert_fields_exists(self, *_init_req)
        self.__initialized = True

    def __setattr__(self, attr: str, val) -> None:
        if self.__initialized and attr in _init_req:
            attribute_initilized_error(self, attr)
        
        if DEBUG and attr in _watch:
            debug_attribute(self, attr, val)
        super().__setattr__(attr, val)

    def train(self, max_epochs: int):
        if getattr(self, 'epoch', None) is None:
            self.epoch = 0
        for epoch in range(self.epoch, max_epochs):
            self.dataloader.set_epoch(epoch, max_epochs)
            print(f"Starting epoch: {epoch}", 'epoch_start')
            epoch_time = time.time()
            self._run_epoch(epoch)
            print(f"Epoch ran in: {time.time() - epoch_time}")
            if epoch % self._save_every == 0:
                self._save_snapshot(epoch)

    def _run_epoch(self, epoch :int):
        print(f"[GPU{self.rank}] Epoch {epoch}")
        self.total_loss = 0
        for source in self.dataloader:
            source = source.to(self.rank)
            self._run_batch(source)
        print(f"Average Loss: {self.total_loss / (epoch + 1):4f}")

    def _run_batch(self, source: torch.Tensor):
        self.optimizer.zero_grad()
        output, mean, logvar = self.model(source)
        recon_loss = torch.nn.MSELoss()(output, source.to_dense())
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kld
        loss.backward()
        self.total_loss += loss
        self.optimizer.step()
    
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

    def _initialize_fields(self, model: torch.nn.Module, dataloader: CellCensusDataLoader, rank: int, optimizer: torch.optim.Optimizer, save_every: int, snapshot_path: str):

        isinstance_error(model, torch.nn.Module)
        isinstance_error(dataloader, CellCensusDataLoader)
        isinstance_error(rank, int)
        isinstance_error(optimizer, torch.optim.Optimizer)
        isinstance_error(snapshot_path, str)
        isinstance_error(save_every, int) 

        if save_every <= 0:
            raise ValueError("The 'save_every' must be a positive integer.")
        
        self.model = model
        self.dataloader = dataloader
        self.rank = rank
        self.optimizer = optimizer
        self._save_every = save_every
        self._snapshot_path = snapshot_path

    __initialized = False
    