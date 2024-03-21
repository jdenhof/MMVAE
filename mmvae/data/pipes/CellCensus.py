import numpy as np
import scipy.sparse as sp
from torchdata.datapipes.iter import FileLister, IterDataPipe
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

                
def load_sparse_matrix(file_tuples):
    """Split incoming tuple from FileLister and load scipy .npz"""
    path, file = file_tuples
    return sp.load_npz(file)

def shuffle_sparse_matrix(sparse_matrix: sp.csr_matrix):
    print("Shuffling sparse matrix", flush=True)
    shuffled_indices = np.random.permutation(sparse_matrix.shape[0])
    return sparse_matrix[shuffled_indices]

class ChunkloaderDataPipe(IterDataPipe):
    
    def __init__(self, directory_path: str = None, masks: list[str] = None, batch_size: int = None, verbose=False):
        self.files = FileLister(
            root=directory_path, 
            masks=masks,
            recursive=False,
            non_deterministic=True
        )
        
        if (verbose):
            for file in self.files:
                print(file)
                
    def __iter__(self):
        return iter(
            self.files
            .shuffle()
            .open_files(mode='b')
            .map(load_sparse_matrix)
            .map(shuffle_sparse_matrix)
        )
            
    
def CellCensusPipeLine(*args, directory_path: str = None, masks: list[str] = None, batch_size: int = None) -> IterDataPipe: # type: ignore
    """
    Pipeline built to load Cell Census sparse csr chunks. 

    Important Note: The sharding_filter is applied aftering opening files 
        to ensure no duplication of chunks between worker processes.
    """
    pipe = (ChunkloaderDataPipe(directory_path, masks, batch_size, verbose=True)
        .sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING) # Prevents chunks from being duplicated across workers
        .prefetch(3)
        .sharding_filter()
        .batch_sparse_csr_matrix(batch_size)
        .attach_to_output(*args)
    )
    
    from torchdata.datapipes.utils import to_graph
    g = to_graph(pipe).render(filename='train_graph', directory='/home/denhofja/graphs')

    return pipe
