import threading
import queue
import torch
from torch.utils.data import Dataset
from TrainerDefaults import CHUNK_SIZE
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
import os

class CellxGeneDataset(Dataset):
    """
    Custom Dataset that initializes a queue to load chunks in parallel to gathering data.

    Args:
        - batch_size (int): Size of the batch.
        - chunk_directory (str): Path to the directory containing data chunks.
        - phase (str): 'train' or 'test' - determines which files get picked up in the chunk_directory.

    Notes:
        - __getitem__: This method returns a batched tensor.
        - Chunk Initialization: Chunks are initialized by scanning the chunk_directory supplied and pulling in any file that contains .npz and matches the phase ('train' or 'test') required. (ie. train_chunk1.npz, chunk_test_1.npz, ...)
    """

    def __init__(self, batch_size, chunk_directory: str, buffer_size: int, phase="train"):
        
        if buffer_size is None or buffer_size < 1:
            raise TypeError("Chunk buffer size must be greater than 1!")
    
        self._initialize_chunk_paths(chunk_directory, phase)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = queue.Queue(maxsize=buffer_size)

        self.loading_thread = threading.Thread(target=self._load_chunks)
        self.loading_thread.daemon = True
        self.loading_thread.start()

        self.batch_idx = 0
        self.chunks_loaded = 0
        self.chunk_size = None
        self.current_chunk = None
        self.rank = None

    def _initialize_chunk_paths(self, directory: str, phase: str):
        if not os.path.exists(directory):
            raise FileNotFoundError(directory)
        
        if phase is not 'test' or phase is not 'train':
            raise TypeError(f"Phase {phase} must be 'test' or 'train'!")
        
        self.chunk_paths = [path for path in os.listdir(directory) if '.npz' in path and phase in path]

    @property
    def total_batches(self) -> int:
        return self.batch_idx // self.batch_size
    
    def set_rank(self, rank: int):
        if self.rank is not None:
            raise AssertionError("Dataset rank can only be set once!")
        self.rank = rank

    def _load_chunks(self) -> None:
        np.random.shuffle(self.chunk_paths)
        for path in self.chunk_paths:
            # NOTE: the buffer should never be full because of block=True on buffer.put which does't
            # add to the buffer until slot available
            if self.buffer.full():
                print("Buffer is full")
                break
            
            chunk = self._load_chunk(path)
            self.buffer.put(chunk, block=True) # waits for available slot
        self.buffer.put(None) # Indicates end of buffer input

    def _load_chunk(self, path) -> csr_matrix:
        # Logic to load and shuffle the chunk
        chunk = sp.load_npz(path)
        shuffled_index = np.random.permutation(chunk.shape[0])
        return chunk[shuffled_index, :]

    def __getitem__(self, _):
        
        if (self.current_chunk == None):
            self.current_chunk = self.buffer.get(timeout=60) # Peek at the first chunk in the buffer
            
        self.chunk_size = self.current_chunk.shape[1]
        batch = self.current_chunk[self.batch_idx:self.batch_idx + self.batch_size, :]
        self.batch_idx += self.batch_size
        
        # Optionally, move to the next chunk if the current one is exhausted
        if self._is_chunk_exhausted():
            self.current_chunk = None
            self.chunks_loaded += 1 # Increment chunk index
        
        return torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape)
    
    def _is_chunk_exhausted(self):
        return self.batch_idx % self.chunk_size + self.batch_size > self.chunk_size
    
    def __len__(self):
        # must be static to ensure distributed data is not duplicated
        # if (self.chunk_size == None):
        #     self.chunk_size = 285341
        # return len(self.chunk_paths) * (self.chunk_size // self.batch_size)
        return CHUNK_SIZE
