from torch.utils.data import IterableDataset, get_worker_info
from utils import assert_field_exists
from scipy.sparse import csr_matrix
from scipy import sparse as sp
from typing import Optional
import numpy as np
import threading
import torch
import queue
import os

# dev only
from main import DEBUG

def worker_init_fn(worker_id: int): 
    worker_info = get_worker_info()
    assert(worker_id == worker_info.id) # sanity purposes only
    worker_info.dataset._worker_init_fn(worker_info.id, worker_info.num_workers)
    worker_info.dataset.start_loading_chunks()

_req = [ # required 
    'batch_size', # 
    '_seed', # Used to maintain the same seed for all workers 
    '_batch_index', # The start index within a chunk for the next batch iterated by batch_size
    '_buffer_size', # Number of chunks to read into memory at once
    '_chunk_paths', # Location of chunks in file system
    '_chunks_loaded', # Number of chunks that have been loaded (only used for metrics)
    '_chunks_exhausted', # Number of chunks that have already been used
]

class ThreadedChunkDataset(IterableDataset):
    
    batch_size: int = None
    
    def __init__(self, batch_size: int = None, chunk_directory: str = None, buffer_size: int = None, sample_size: int = None, seed: int = None, phase:str ="train"):
        super().__init__()

        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than 0")
        
        self.batch_size = batch_size
        self._sample_size = sample_size
        self._buffer_size = buffer_size
        self._seed = seed

        self._buffer = queue.Queue[csr_matrix](maxsize=buffer_size)
        self._initialize_chunk_paths(chunk_directory, phase)

        [assert_field_exists(self, field) for field in _req ]

    @property
    def _is_worker_initialized(self):
        return self._worker_id is not None and self._distributed_chunked_paths is not None

    def _initialize_chunk_paths(self, directory, phase):
        if not os.path.exists(directory):
            raise FileNotFoundError(directory)

        if phase not in ['test', 'train']:
            raise ValueError("Phase must be 'test' or 'train'")
        
        if DEBUG: # overrides phase so chunks only need to end in .npz
            phase = ''

        # TODO: Add error handeling
        self._chunk_paths = [os.path.join(directory, path) for path in os.listdir(directory) if 'chunk' in path and '.npz' in path and phase in path]
        if len(self._chunk_paths) == 0:
            raise FileNotFoundError(f"No chunks found at {directory}\nTain Example file:\nchunk_1_train.npz\nTest Example File:\ntest_chunk_1.npz")
        
    def start_loading_chunks(self):
        assert(self._is_worker_initialized)
        loading_thread = threading.Thread(target=self._load_chunks)
        loading_thread.daemon = True
        loading_thread.start()
        # TODO: Gracefully end thread

    def _load_chunks(self):
        epoch = 0
        while True:
            if not self._is_worker_initialized:
                continue
            np.random.seed(self._seed + epoch)
            np.random.shuffle(self._distributed_chunked_paths)
            for path in self._distributed_chunked_paths:
                chunk = self._load_chunk(path)
                self._buffer.put(chunk, block=True)
                self._chunks_loaded += 1
            epoch += 1

    def _load_chunk(self, path) -> csr_matrix:
        chunk = sp.load_npz(path)
        shuffled_indices = np.random.permutation(chunk.shape[0])
        return chunk[shuffled_indices, :]
    
    def _get_chunk(self) -> csr_matrix:
        try:
            self._current_chunk = self._buffer.get(timeout=60)
            self._batch_index = 0
            self._chunk_size = self._current_chunk.shape[0]
        except StopIteration:
            self._current_chunk = None

    def _worker_init_fn(self, worker_id, num_workers):
        assert(self._worker_id is None)
        self._worker_id = worker_id
        assert_field_exists(self, '_chunk_paths')
        assert(len(self._chunk_paths) > 0)
        chunks_per_worker = int(np.ceil(len(self._chunk_paths) / float(num_workers)))
        assert(chunks_per_worker > 0)
        if num_workers > 1:
            partition_size = worker_id * chunks_per_worker
            distributed_chunked_paths = self._chunk_paths[partition_size: partition_size + chunks_per_worker]
        else:
            distributed_chunked_paths = self._chunk_paths
        self._distributed_chunked_paths = distributed_chunked_paths
        print(f"[Worker{worker_id}] Chunks: {len(self._distributed_chunked_paths)}")

    def __iter__(self):

        assert(self._is_worker_initialized)

        while True:
            if self._current_chunk is None:
                self._get_chunk()
                if self._current_chunk is None:
                    break

            end_idx = self._batch_index + self.batch_size
            # TODO: Could cause errors if num samples cut off no longer aligns with sample_size
            if self._batch_index + end_idx + self.batch_size >= self._chunk_size:
                self._chunks_exhausted += 1
                self._current_chunk = None

            batch = self._current_chunk[self._batch_index:end_idx, :]
            assert(isinstance(batch, csr_matrix))
            assert(batch.shape[0] == self.batch_size)
            yield torch.sparse_csr_tensor(batch.indptr, batch.indices, batch.data, batch.shape)
            self._batch_index += self.batch_size

    def __len__(self):
        """ Note: 
            - Length must be known at __init__ because chunk_size cannot be determined until first call of __iter__ which happens after __len__ called
        """
        # if self.chunk_paths is None or self.chunk_size is None:
        #     raise AttributeError("Chunk paths and chunk size cannot be None")
        # return self.chunk_size * len(self.chunk_paths)
        return self._sample_size
    
    _buffer: queue.Queue[csr_matrix]
    _seed: int = None
    _sample_size: int = None
    _batch_index: int = 0 
    _buffer_size: int = None
    _chunk_size: int = None
    _chunk_paths: Optional[str] = None
    _chunks_loaded: int = 0
    _chunks_exhausted: int = 0
    _worker_id: Optional[int] = None
    _current_chunk: Optional[csr_matrix] = None
    _distributed_chunked_paths: Optional[str] = None