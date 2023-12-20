#!/usr/bin/env python
import os
from utils import get_model_instance_from_file, DEBUG
from Arguments import Args, parse_args

def _initialize_chunk_paths(directory: str):
    chunk_paths = [os.path.join(directory, path) for path in os.listdir(directory) if 'chunk' in path and '.npz' in path]
    if chunk_paths is None or len(chunk_paths) == 0:
        raise FileNotFoundError(f"No chunks found at {directory}\nTain Example file:\nchunk_1_train.npz\nTest Example File:\ntest_chunk_1.npz")
    return chunk_paths

def main(args: Args):
    import torch
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"GPU[{rank}] Initialized in a world of size [{world_size}]")
    
    model = get_model_instance_from_file(args.model_path).to(rank)
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[rank], output_device=[rank])

    from DataLoader import CellCensusDataLoader
    with CellCensusDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_chunk_workers=args.num_chunk_workers,
        buffer_size=args.chunk_buffer_size,
        chunk_paths=_initialize_chunk_paths(args.chunks_directory)[:5]
    ) as loader:
        
        optimizer = torch.optim.Adam(model.parameters()) 
        
        from Trainer import Trainer
        trainer = Trainer(model, loader, rank, optimizer, args.save_every, args.snapshot_path)
        trainer.train(args.total_epochs)

    dist.destroy_process_group()
    print("Training Complete!")
   

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    args = parse_args()
    main(args)