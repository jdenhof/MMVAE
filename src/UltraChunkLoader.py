
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from queue import Queue
from multiprocessing.synchronize import Event
from typing import Any
import scipy.sparse as sp
import collections
from utils import DEBUG, assert_fields_exists, isinstance_error, print, attribute_initilized_error
import os

class _ChunkMananger(SyncManager):

    def __init__(self, num_workers: int, buffer_size: int):
        super().__init__()
        self.start()
        self.num_workers = num_workers
        if buffer_size > 20:
            raise ValueError(f"Buffer size max set to 20 - [:{buffer_size}]")
        self.chunk_queue: Queue[sp.csr_matrix] = self.Queue(maxsize=buffer_size)
        self.completion_queue: Queue[int] = self.Queue() # must be infinite to prevent blocking workers

Msg = collections.namedtuple('Msg', ['event', 'args'])
class BaseProcess(mp.Process):
    """A process backed by an internal queue for simple one-way message passing.
    """
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def send(self, event, *args):
        """Puts the event and args as a `Msg` on the queue
        """
        if event == 'quit':
            print("Signaled process to quit")
            msg = None
        else:
            msg = Msg(event, args)
        self.queue.put(msg)

    def dispatch(self, msg):
        event, args = msg

        handler = getattr(self, "do_%s" % event, None)
        if not handler:
            raise NotImplementedError("Process has no handler for [%s]" % event)

        handler(*args)

    def run(self):
        while True:
            msg = self.queue.get()
            if msg is None:
                break
            self.dispatch(msg)

class _ManagedBaseProcess(BaseProcess):
    def __init__(self, manager: _ChunkMananger):
        super().__init__(manager.Queue())

class _Worker(_ManagedBaseProcess):

    def __init__(self, manager: _ChunkMananger, id: int):
        super().__init__(manager)
        self.id = id
        self.chunk_queue = manager.chunk_queue
        self.completion_queue = manager.completion_queue

    def do_load_chunk(self, path: str):
        chunk = _Worker._load_chunk(path)
        self.chunk_queue.put(chunk)
        self.completion_queue.put(self.id)
        print(f"Worker {self.id} put chunk", f'worker_{self.id}_put')

    @staticmethod
    def _load_chunk(path):
        return sp.load_npz(path)

class _Distributer(_ManagedBaseProcess):

    def __init__(self, manager: _ChunkMananger, paths: list[str]):
        super().__init__(manager)
        self.num_workers = manager.num_workers
        self.completion_queue = manager.completion_queue
        self.chunk_queue = manager.chunk_queue
        self.paths = paths
        self.workers = [_Worker(manager, i) for i in range(manager.num_workers)]

    def start(self):
        for worker in self.workers:
            worker.start()
        super().start()

    def join(self):
        for worker in self.workers:
            worker.send('quit')
        for worker in self.workers:
            worker.join()
        self.send('quit')
        super().join()

    def close(self):
        for worker in self.workers:
            worker.close()
        super().close()

    def terminate(self):
        for worker in self.workers:
            worker.terminate()
        super().terminate()

    def do_distribute(self, epoch):
        epoch = epoch if epoch is not None else 'not defined'
        task_count = len(self.paths)
        print(f"Distributing chunks for epoch {epoch}", 'stager_distribute')
        for i, path in enumerate(self.paths):
            self.workers[i % self.num_workers].send('load_chunk', path)

        while task_count > 0:
            self.completion_queue.get()
            task_count -= 1

        print(f"Workers finished loading all chunks for epoch {epoch}", 'stager_complete')
        self.chunk_queue.put(None)

class _ChunkLoader:

    def __init__(self, manager: _ChunkMananger, directory: str, phase: str):
        paths = _ChunkLoader._initialize_chunk_paths(directory, phase)
        self._distributer = _Distributer(manager, paths)

    def start(self, start_epoch: int, max_epochs: int):
        """Loads all epochs into epoch_queue"""
        for i in range(start_epoch, max_epochs):
            if not self._distributer.is_alive():
                raise RuntimeError("The chunk loader has mysteriously quit!")
            self._distributer.send('distribute', i)
        print("Done setting staging epochs")

    def get(self, block=True, timeout: float = None):
        return self._distributer.chunk_queue.get(block=block, timeout=timeout)
    
    @staticmethod
    def _initialize_chunk_paths(directory: str, phase: int):
        phase = '' if DEBUG else phase # overrides phase so chunks only need to end in .npz
        chunk_paths = [os.path.join(directory, path) for path in os.listdir(directory) if '.npz' in path and phase in path]
        if chunk_paths is None or len(chunk_paths) == 0:
            raise FileNotFoundError(f"No chunks found at {directory}\nTain Example file:\nchunk_1_train.npz\nTest Example File:\ntest_chunk_1.npz")
        return chunk_paths
    
class ChunkLoader(_ChunkLoader):

    def __init__(self, epoch: int, max_epochs: int, num_workers: int, buffer_size: int, directory: str, phase: str):
        self.epoch = epoch
        self.max_epochs = max_epochs
        self._manager = _ChunkMananger(num_workers, buffer_size)
        super().__init__(self._manager, directory, phase)

    def __enter__(self):
        self._distributer.start()
        self.start(self.epoch, self.max_epochs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print(f"Exiting Chunk Loader {exc_type is StopIteration}")
        try:
            if exc_type is StopIteration:
                raise NotImplementedError() # Cannot figure out how to cleanly exit
                self._distributer.join()
                self._distributer.close()
            else:
                raise NotImplementedError()
        except:
            self._distributer.terminate()
        finally:
            self._manager.shutdown()
        return exc_type is StopIteration
