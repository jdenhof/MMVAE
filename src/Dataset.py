from queue import Empty
from torch.utils.data import IterableDataset
from utils import assert_fields_exists, attribute_initilized_error, print
from UltraChunkLoader import ChunkLoader
from scipy.sparse import csr_matrix as CsrMatrix
import torch

# dev only
from utils import DEBUG

_static_req = (
    'batch_size',
    'device',
    'sample_size',
)

class ThreadedChunkDataset(IterableDataset):

    _initialized = False
    
    def __init__(self, batch_size: int, device: str, sample_size: int, loader: ChunkLoader = None):
        super(ThreadedChunkDataset, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.loader = loader

        assert_fields_exists(self, *_static_req)
        self._initialized = True

    @property
    def ready(self):
        return getattr(self, 'loader', None) is not None

    def __iter__(self):
        self._current_chunk = None
        return self

    def get_chunk_with_retrys(self, max_retrys = 5):
        retrys = 0
        while max_retrys > retrys:
            try:
                chunk = self.loader.get(timeout=5 * retrys)
            except Empty:
                retrys += 1
                print("Retrying get...")
                continue

            if chunk is None:
                print("Raising stop iteration", 'stop_iteration')
                raise StopIteration()
            
            print(f"Chunk popped from queue in {retrys} tr{'y' if retrys == 1 else 'ies'}", 'chunk_popped')
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
        
        if self._batch_index + self.batch_size > chunk_size:
            raise ValueError("Batch end index exceeded chunk size!")

        batch = self._current_chunk[self._batch_index:self._batch_index + self.batch_size, :]
        assert(isinstance(batch, CsrMatrix))
        if not batch.shape[0] == self.batch_size:
            raise ValueError(f"Batch of shape {batch.shape} does not equal batch size {self.batch_size} ({self._batch_index}, {self._batch_index + self.batch_size}, {chunk_size})!")
        self._batch_index += self.batch_size
        return torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape, check_invariants=DEBUG, device=self.device)
        
    def __len__(self):
        """ Note: 
        TODO: Could change due to no longer using DDP
            - Length must be known at __init__ because chunk_size cannot be determined until first call of __iter__ which happens after __len__ called
        """
        return self.sample_size
    
    def __setattr__(self, attr: str, val) -> None:

        if self._initialized and attr in ('batch_size',):
            attribute_initilized_error(self, attr)

        super().__setattr__(attr, val)

    loader: ChunkLoader = None
    _current_chunk: CsrMatrix = None