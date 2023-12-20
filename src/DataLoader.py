from torch.utils.data import DataLoader
from Dataset import CellCensusDataset
from ChunkLoader import ChunkLoader

class CellCensusDataLoader(DataLoader):

    def __init__(
            self, 
            batch_size: int, 
            num_workers: int, 
            num_chunk_workers: int, 
            buffer_size: int,
            chunk_paths: list[str]
        ) -> None:
        self._chunk_loader = ChunkLoader(num_chunk_workers, buffer_size, chunk_paths)
        super(CellCensusDataLoader, self).__init__(
            dataset=CellCensusDataset(self._chunk_loader, batch_size),
            num_workers=num_workers,
            batch_size=None,
            shuffle=False,
            pin_memory=False, # TODO: Fix at https://github.com/pytorch/pytorch/issues/115330#issuecomment-1853919592 to set True
        )

    def __iter__(self):
        if getattr(self, 'epoch', None) is None: 
            raise RuntimeError("Epoch not set in dataloader CellCensusDataLoader.set_epoch(epoch: int, max_epochs: int = None)")
        return super().__iter__()
        

    def set_epoch(self, epoch: int, max_epochs: int = None):

        if getattr(self, 'epoch', None) is None: # If not started
            self._chunk_loader.set_epoch(epoch)  # Dispatches first epoch to chunk loader
        
        if epoch + 1 < max_epochs: # Dispatches next epoch to chunk loader if less than max_epochs
            self._chunk_loader.set_epoch(epoch + 1)
        self.epoch = epoch

    def __enter__(self):
        self._chunk_loader._distributer.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Exiting Chunk Loader {exc_type is StopIteration}")
        try:
            self._chunk_loader.quit()
        except:
            self._chunk_loader._distributer.terminate()
        finally:
            self._chunk_loader._factory.manager.shutdown()
        return exc_type is StopIteration