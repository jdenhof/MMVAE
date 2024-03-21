from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from typing import Generator, Any
import mmvae.data.pipes as p
import torch
import random
import os

class MultiModalLoader:
    """
    Stochastic sampler from dataloaders
    
    Args:
     - exhaust_all: exhaust all dataloaders to completion (default=True)
    """

    def __init__(self, *modals: DataLoader2, exhaust_all=True):
        self.exhaust_all = exhaust_all
        if len(modals) == 0:
            raise ValueError("A dataloader must be defined!")
        self.modals = modals
        self.__len = None

    def __len__(self):
        if self.__len == None:
            raise RuntimeError("Length of MultiModalLoader cannot be determined until one entire forward pass")
        return self.__len

    def __iter__(self) -> Generator[tuple[torch.Tensor, str, Any], Any, None]:
        loaders = [iter(modal) for modal in self.modals]
        self.__len = None
        while loaders:
            loader_idx = random.randrange(len(loaders))
            try:
                yield next(loaders[loader_idx])
                if self.__len == None:
                    self.__len = 0
                self.__len += 1
            except StopIteration as e:
                if not self.exhaust_all:
                    return
                del loaders[loader_idx]

def worker_init_fn(DataPipe, WorkerInfo):
    print(DataPipe, WorkerInfo, flush=True)
    os.environ['WORKER_ID'] = str(WorkerInfo.worker_id)
    return DataPipe
class ChunkedCellCensusDataLoader(DataLoader2):
    """
        Dataloader wrapper for CellCensusPipeline

        Args:
         - *args: inputs to be yielded for every iteration
         - directory_path: string path to chunk location
         - masks: unix style regex matching for each string in array
         - batch_size: size of output tensor first dimension
         - num_workers: number of worker process to initialize

         Attention:
          - num_workers must be greater or equal to the total chunks to load
        """
    def __init__(self, *args, directory_path: str = None, masks: list[str] = None, batch_size: int = None, num_workers: int = None): # type: ignore
        super(ChunkedCellCensusDataLoader, self).__init__(
            datapipe=p.CellCensusPipeLine(*args, directory_path=directory_path, masks=masks, batch_size=batch_size), # type: ignore
            datapipe_adapter_fn=None,
            reading_service=MultiProcessingReadingService(num_workers=num_workers, worker_init_fn=worker_init_fn, worker_prefetch_cnt=1)
        )
        
def configure_multichunk_dataloaders(
    batch_size,
    directory_path="/active/debruinz_project/human_data/python_data",
    train_masks=[f'human_chunk_{key}.npz' for key in range(1, 7)],
    test_masks=[f'human_chunk_{key}.npz' for key in range(80, 86)],
    test_batch_size=None,
) -> tuple[ChunkedCellCensusDataLoader, ChunkedCellCensusDataLoader]:
    """
    Returns tuple of (train_dataset, test_dataset).
    """
    if test_batch_size == None:
        test_batch_size = batch_size
        
    if batch_size <= 0 or test_batch_size <= 0:
        raise RuntimeError("Train and test batchsizes must be greater than 0!")
        
    return (
        ChunkedCellCensusDataLoader(
            directory_path=directory_path, 
            masks=train_masks, 
            batch_size=batch_size, 
            num_workers=3
        ),
        ChunkedCellCensusDataLoader(
            directory_path=directory_path,
            masks=test_masks, 
            batch_size=test_batch_size, 
            num_workers=3
        ),
    )
    
        
import mmvae.data.utils as utils
def configure_singlechunk_dataloaders(
    data_file_path: str,
    metadata_file_path: str,
    train_ratio: float,
    batch_size: int,
    device: torch.device,
    test_batch_size: int = None
):
    """
    Splits a csr_matrix provided by data_file_path with equal length metadata_file_path by train_ratio 
        which is a floating point between 0-1 (propertion of training data to test data). 
    
    If device is not None -> the entire dataset will be loaded on device at once.
    """
    if not test_batch_size:
        test_batch_size = batch_size
        
    (train_data, train_metadata), (validation_data, validation_metadata) = utils.split_data_and_metadata(
        data_file_path,
        metadata_file_path,
        train_ratio)
    
    from mmvae.data.datasets.CellCensusDataSet import CellCensusDataset, collate_fn
    if device:
        train_data = train_data.to(device)
        validation_data = validation_data.to(device)
        
    train_dataset = CellCensusDataset(train_data, train_metadata)
    test_dataset = CellCensusDataset(validation_data, validation_metadata)
    
    from torch.utils.data import DataLoader
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=test_batch_size,
            collate_fn=collate_fn,
        )
    )