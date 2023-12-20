#!/usr/bin/env python
from utils import get_model_instance_from_file, DEBUG
from Args import Args, parse_args

def main(args: Args):
    import torch
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if DEBUG:
        print(f"GPU[{rank}] Initialized in a world of size [{world_size}]")
    
    model = get_model_instance_from_file(args.model_path).to(rank)
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[rank], output_device=[rank])

    import Phase
    from CellCensusDataLoader import CellCensusDataLoader
    with CellCensusDataLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_chunk_workers=args.num_chunk_workers,
        buffer_size=args.chunk_buffer_size,
        directory=args.chunks_directory,
        phase=Phase.TRAIN
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