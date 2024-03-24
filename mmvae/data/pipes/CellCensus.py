import numpy as np
import scipy.sparse as sp
from torchdata.datapipes.iter import FileLister, IterDataPipe
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

class ChunkloaderDataPipe(IterDataPipe):
    
    def __init__(self, directory_path: str, masks: list[str], name: str = None, verbose: bool = None):
        super(ChunkloaderDataPipe, self).__init__()
        self.files = FileLister(
            root=directory_path, 
            masks=masks,
            recursive=False,
            non_deterministic=True
        )
        
        self.verbose = bool(verbose)
        if name is None:
                    name = ""
        self.name = str(name)
        
        if (self.verbose):
            for file in self.files:
                print(name, file)
    
    def load_sparse_matrix(self, file_tuples):
        """Split incoming tuple from FileLister and load scipy .npz"""
        path, file = file_tuples
        if self.verbose:
            print(f"Loading file path: {path}")
        return sp.load_npz(file)
    
    def shuffle_sparse_matrix(self, sparse_matrix: sp.csr_matrix):
        shuffled_indices = np.random.permutation(sparse_matrix.shape[0])
        return sparse_matrix[shuffled_indices]
                
    def __iter__(self):
        return iter(
            self.files
            .shuffle()
            .open_files(mode='b')
            .map(self.load_sparse_matrix)
            .map(self.shuffle_sparse_matrix)
        )
            
    
def CellCensusPipeLine(directory_path: str, masks: list[str], batch_size: int, name="", verbose=False) -> IterDataPipe: # type: ignore
    """
    Pipeline built to load Cell Census sparse csr chunks. 

    Important Note: The sharding_filter is applied aftering opening files 
        to ensure no duplication of chunks between worker processes.
    """
    pipe = (ChunkloaderDataPipe(directory_path, masks, name=name, verbose=verbose)
        .sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING) # Prevents chunks from being duplicated across workers
        .batch_sparse_csr_matrix(batch_size)
        .shuffle()
    )
    return pipe
