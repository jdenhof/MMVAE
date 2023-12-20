from torch.utils.data import IterableDataset
from utils import attribute_initilized_error
from MultiProcessChunkLoader import MultiProcessChunkLoader
from scipy.sparse import csr_matrix
import queue
import torch

# dev only
from utils import DEBUG

class CellCensusDataset(IterableDataset):

    _initialized = False
    
    def __init__(self, chunk_loader: MultiProcessChunkLoader, batch_size: int):
        super(CellCensusDataset, self).__init__()
        self._chunk_loader = chunk_loader
        self.batch_size = batch_size

    @property
    def ready(self):
        return getattr(self, '_chunk_loader', None) is not None

    def __iter__(self):
        if not self.ready:
            raise RuntimeError("Dataset is not ready to be iterated - Chunk loader not initialized")
        self._current_chunk = None
        return self

    def get_chunk_with_retrys(self, max_retrys = 5):
        retrys = 0
        while max_retrys > retrys:
            try:
                chunk = self._chunk_loader.get(timeout=5 * retrys)
            except queue.Empty:
                retrys += 1
                continue

            if chunk is None:
                raise StopIteration()
            
            return chunk
        raise ValueError("Max retrys exceeded!")
    
    def __next__(self):

        if self._current_chunk is None:
            self._current_chunk = self.get_chunk_with_retrys()
            self._batch_index = 0
    
        chunk_size = self._current_chunk.shape[1]
        
        if self._batch_index + self.batch_size >= chunk_size:
            self._current_chunk = None
            self.__next__()
            
        if self._batch_index > chunk_size:
            raise ValueError("Batch start index exceeded chunk size!")
        
        if self._batch_index + self.batch_size >= chunk_size:
            raise ValueError("Batch end index exceeded chunk size!")
        
        batch = self._current_chunk[self._batch_index:self._batch_index + self.batch_size, :]
        assert(isinstance(batch, csr_matrix))
        if not batch.shape[0] == self.batch_size:
            raise ValueError(f"Batch of shape {batch.shape} does not equal batch size {self.batch_size} ({self._batch_index}, {self._batch_index + self.batch_size}, {chunk_size})!")
        self._batch_index += self.batch_size
        return torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape, check_invariants=DEBUG)
    
    def __setattr__(self, attr: str, val) -> None:

        if self._initialized and attr in ('batch_size',):
            attribute_initilized_error(self, attr)

        super().__setattr__(attr, val)

    _chunk_loader: MultiProcessChunkLoader = None
    _current_chunk: csr_matrix = None