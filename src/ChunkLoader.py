
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import queue
from multiprocessing.synchronize import Event
import scipy.sparse as sp
import collections
from utils import DEBUG
import os


class _ChunkManager(BaseManager):
    pass

# Register Queue and Event with the manager
_ChunkManager.register('get_queue', mp.Queue)
_ChunkManager.register('get_event', mp.Event)

class _SharedResourceFactory:

    def __init__(self, buffer_size: int):
        super().__init__()
        self.manager = _ChunkManager()
        self.manager.start()
        if buffer_size > 20:
            raise ValueError(f"Buffer size max set to 20 - [:{buffer_size}]")
        self.chunk_queue: mp.Queue = self.manager.get_queue(maxsize=buffer_size)
        self.completion_queue: mp.Queue = self.manager.get_queue() # must be infinite to prevent blocking workers
        self.quit_event: Event = self.manager.get_event()

Msg = collections.namedtuple('Msg', ['event', 'args'])
class BaseProcess(mp.Process):
    """A process backed by an internal queue for simple one-way message passing.
    """
    def __init__(self, queue: mp.Queue, quit_event: Event):
        super().__init__()
        self.queue = queue
        self.quit_event = quit_event

    def send(self, event, *args):
        """Puts the event and args as a `Msg` on the queue
        """
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
            if self.quit_event.is_set():
                break
            try:
                msg = self.queue.get(timeout=1)
            except queue.Empty:
                continue
            if msg is None:
                break
            self.dispatch(msg)

class _ManagedBaseProcess(BaseProcess):
    def __init__(self, resource_factory: _SharedResourceFactory):
        super().__init__(resource_factory.manager.get_queue(), quit_event=resource_factory.quit_event)

class _Worker(_ManagedBaseProcess):

    def __init__(self, resource_factory: _SharedResourceFactory, id: int):
        super().__init__(resource_factory)
        self.id = id
        self.chunk_queue = resource_factory.chunk_queue
        self.completion_queue = resource_factory.completion_queue

    def do_load_chunk(self, path: str):
        chunk = _Worker._load_chunk(path)
        put_chunk = False
        while not self.quit_event.is_set() and not put_chunk:
            try:
                self.chunk_queue.put(chunk, timeout=1)
                put_chunk = True
            except queue.Full:
                continue

        self.completion_queue.put(self.id)

    @staticmethod
    def _load_chunk(path):
        return sp.load_npz(path)

class _Distributer(_ManagedBaseProcess):

    def __init__(self, num_workers: int, resource_factory: _SharedResourceFactory, paths: list[str]):
        super().__init__(resource_factory)
        self.num_workers = num_workers
        self.completion_queue = resource_factory.completion_queue
        self.chunk_queue = resource_factory.chunk_queue
        self.paths = paths
        self.workers = [_Worker(resource_factory, i) for i in range(num_workers)]

    def start(self):
        for worker in self.workers:
            worker.start()
        super().start()

    def join(self):
        for worker in self.workers:
            worker.join()
        super().join()

    def close(self):
        for worker in self.workers:
            worker.close()
        super().close()

    def terminate(self):
        for worker in self.workers:
            worker.terminate()
        super().terminate()

    def do_distribute(self, epoch: int):

        epoch = epoch if epoch is not None else 'not defined'
        task_count = len(self.paths)

        for i, path in enumerate(self.paths):
            self.workers[i % self.num_workers].send('load_chunk', path)

        while task_count > 0:
            try:
                if self.quit_event.is_set():
                    return
                self.completion_queue.get(timeout=1)
            except queue.Empty:
                continue
            task_count -= 1
        self.chunk_queue.put(None)

class _Cleaner(mp.Process):

    def __init__(self, queue: queue.Queue, done_cleaning: Event):
        self.queue = queue
        self.done_cleaning = done_cleaning
        super().__init__()

    def run(self):
        attempt = 0
        while not self.done_cleaning.is_set():
            try:
                # Try to get an item from the queue
                item = self.queue.get(timeout=1)  # Wait for wait_time seconds for an item
                del item
            except queue.Empty:
                # No item was found in the queue for wait_time seconds
                if attempt % 25 == 0:
                    print("Queue has been empty for a while, cleaner is checking again...")
            finally:
                attempt += 1

class ChunkLoader:

    def __init__(self, num_workers: int, buffer_size: int, paths: list[str]):
        self._factory = _SharedResourceFactory(buffer_size)
        self._distributer = _Distributer(num_workers, self._factory, paths)

    def set_epoch(self, epoch: int):
        if not self._distributer.is_alive():
            raise RuntimeError("The chunk loader has mysteriously quit!")
        self._distributer.send('distribute', epoch)

    def get(self, block=True, timeout: float = None):
        return self._factory.chunk_queue.get(block=block, timeout=timeout)
    
    def quit(self):
        done_cleaning: Event = self._factory.manager.get_event()
        cleaner = _Cleaner(self._factory.chunk_queue, done_cleaning)
        self._factory.quit_event.set()
        cleaner.start()
        self._distributer.join()
        done_cleaning.set()
        cleaner.join()