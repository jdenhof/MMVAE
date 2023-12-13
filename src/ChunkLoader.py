import multiprocessing
from queue import Empty
from typing import Any
from utils import debug_attribute, attribute_initilized_error, assert_field_exists, range_error, print
import numpy as np
import random
from scipy import sparse as sp
import os

# dev only
from utils import DEBUG

class Phase:
    train = 'train'
    test = 'test'
    phases = ('train', 'test')

    def validate(phase):
        if phase not in Phase.phases:
            raise ValueError(f"Phase {phase} not in {Phase.phases}")
        return phase
    
_static_req = (
    'directory',
    'phase',
    'chunk_paths',
    'num_workers',
    'buffer_size',
    'seed'
)

class ChunkLoader:

    max_workers: int = 16
    max_input_size: int = 100
    max_buffer_size: int = 20
    default_timeout: int = 20
    __initialized = False

    def __init__(
        self, 
        epochs: int,
        directory: str, 
        phase: str, 
        num_workers : int, 
        buffer_size: int, 
        seed: int,
    ) -> None:
        self.phase = Phase.validate(phase)
    
        if not os.path.exists(directory):
            raise FileNotFoundError(directory)
        self.directory = directory
        
        self.epochs = epochs
        
        self.seed = seed # TODO: Error handle
        self.num_workers = range_error(self, num_workers, ChunkLoader.max_workers)
        self.buffer_size = range_error(self, buffer_size, ChunkLoader.max_buffer_size)
        self._initialize_chunk_paths()

        self.output_queue = multiprocessing.Queue(maxsize=self.buffer_size)
        self.input_queues = [multiprocessing.Queue(maxsize=self.max_input_size) for _ in range(self.num_workers)]

        [assert_field_exists(self, field) for field in _static_req]
        self.__initialized = True

    @property
    def chunks(self) -> int:
        attribute_initilized_error(self, 'chunk_paths')
        return len(self.chunk_paths)

    def get(self):
        if getattr(self, '_get_count', None) is None:
            self._get_count = 0
        if self.__running == False:
            raise RuntimeError(f"{self.__class__.__name__} has not been started: Perhaps you forgot to call start()")
        assert_field_exists(self, 'output_queue')
        if self._get_count == len(self.chunk_paths):
            self._get_count = 0
            raise Empty()
        self._get_count += 1
        chunk = self.output_queue.get(timeout=self.default_timeout)
        if chunk is not None:
            shuffled_indices = np.random.permutation(chunk.shape[0])
            return chunk[shuffled_indices, :]
        return chunk

    def _start(self):
        
        self.workers = [multiprocessing.Process(target=_WorkerTask._worker_task, args=(i, iq, self.output_queue)) for i, iq in enumerate(self.input_queues)]

        for worker in self.workers:
            worker.start()
        self.__running = True

    def set_epoch(self, epoch: int, distribute = True, reset = True):
        self.epoch = epoch
        if distribute:
            self._distribute_work()

    def _distribute_work(self):
        # Distribute paths evenly among input queues
        for i, path in enumerate(self.chunk_paths):
            self.input_queues[i % self.num_workers].put(path)

    def __setattr__(self, __name: str, __value: Any) -> None:
        if self.__initialized and __name in _static_req:
            attribute_initilized_error(self, __name)
        super().__setattr__(__name, __value)

    def _initialize_chunk_paths(self):
        
        phase = '' if DEBUG else self.phase # overrides phase so chunks only need to end in .npz
        chunk_paths = [os.path.join(self.directory, path) for path in os.listdir(self.directory) if '.npz' in path and phase in path]

        if chunk_paths is None or len(chunk_paths) == 0:
            raise FileNotFoundError(f"No chunks found at {self.directory}\nTain Example file:\nchunk_1_train.npz\nTest Example File:\ntest_chunk_1.npz")
        
        self.chunk_paths = chunk_paths
    
    def close(self):
        assert_field_exists(self, 'input_queues')
        assert_field_exists(self, 'workers')
        # Signal workers to terminate
        for iq in self.input_queues:
            iq.put(None)
        for w in self.workers:
            w.join()

    def terminate(self):
        print("FATAL ERROR!! AVOID CALLING TERMINATE")
        assert_field_exists(self, 'workers')
        for w in self.workers:
            w.terminate()

    __running = False


class ChunkPathsGenerator:

    def __init__(self, loader: ChunkLoader):
        assert_field_exists(loader, 'epochs')
        self.epochs = loader.epochs
        assert_field_exists(loader, 'chunk_paths')
        self.paths = loader.chunk_paths

    def __iter__(self):
        for _ in range(self.epochs):
            random.shuffle(self.paths)
            for path in self.paths:
                yield path 

class _WorkerTask:

    @staticmethod
    def _worker_task(id: int, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
        while True:
            try:
                path = input_queue.get()
                if path is None:
                    print(f"Worker {id} is done working")
                    break

                chunk = _WorkerTask._load_chunk(path)
                output_queue.put(chunk)
                print(f"Worker {id} put chunk on queue", f"worker_{id}_put_chunk")
            except:
                raise ChildProcessError(f"Worker {id} interrupted and exiting")

    @staticmethod
    def _load_chunk(path) -> sp.csr_matrix:
        chunk: sp.csr_matrix = sp.load_npz(path)
        # shuffled_indices = np.random.permutation(chunk.shape[0]) - NOTE: Removing all cpu bound operations in worker task
        return chunk # [shuffled_indices, :]
    