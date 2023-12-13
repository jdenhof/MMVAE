#!/usr/bin/env python
import os
import sys
from dataclasses import dataclass

DEBUG = True

@dataclass
class _Args:
    model_path: str
    total_epochs: int
    batch_size: int
    num_workers: int
    seed: int
    sample_size: int
    chunk_buffer_size: int
    chunks_directory: str
    save_every: int
    snapshot_path: str

def parse_args():
    import argparse
    import TrainerDefaults

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument(
        'total_epochs', 
        type=int, 
        help='Total epochs to train the model'
    )
    parser.add_argument(
        'batch_size', 
        type=int, 
        help='Input batch size on each device'
    )
    parser.add_argument(
        f"{'--' if DEBUG else ''}model_path",
        type=str, 
        help="Path to model to use for training (Note: class name must be Model)"
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        help='Number of workers per gpu', 
        default=TrainerDefaults.NUM_WORKERS
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        help='Shared seed amoung workers', 
        default=TrainerDefaults.SEED
    )
    parser.add_argument(
        '--sample_size', 
        type=int, 
        help='Size of dataset', 
        default=TrainerDefaults.SAMPLE_SIZE
    )
    parser.add_argument(
        '--chunk_buffer_size', 
        type=int, 
        help='Size of chunk queue per dataset', 
        default=TrainerDefaults.CHUNK_BUFFER_SIZE
    )
    parser.add_argument(
        '--chunks_directory', 
        type=str,
        help='Directory where chunks to be ran are stored', 
        default=TrainerDefaults.CHUNK_DIRECTORY
    )
    parser.add_argument(
        '--save_every', 
        type=int, 
        help='How often to save a snapshot', 
        default=TrainerDefaults.SAVE_EVERY
    )
    parser.add_argument(
        '--snapshot_path', 
        type=str,
        help='Path to save/load training snapshots', 
        default=TrainerDefaults.SNAPSHOT_PATH
    )

    args = parser.parse_args(namespace=_Args)
    if DEBUG:
        args.model_path = TrainerDefaults.MODEL_PATH

    return args

def _get_model_instance_from_file(file_path: str):
    """
        Args: file_path - points to file that contains Model to run
        Notes:
        - file must contain path Model that has zero arguments
        
        Example:
        >>> class Model(VAE):
        ...     def __init__(self):
        ...         super().__init__(Encoder(60664, 512, 128), Decoder(128, 512, 60664))
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("user_defined_module", os.path.join(file_path))
    module = module_from_spec(spec)
    sys.modules["user_defined_module"] = module
    spec.loader.exec_module(module)
    model = getattr(module, 'Model')()

    from torch.nn import Module
    try:
        assert(isinstance(model, Module))
    except:
        raise Exception(f"File: {file_path} - No class Model of torch.nn.Module found")
    
    return model

def main(args: _Args):
    import torch
    from torch.distributed import init_process_group, destroy_process_group
    init_process_group(backend="nccl") # DDP Setup

    if not torch.cuda.is_available():
        devices = "cpu"
    else:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    assert(devices is not None and devices == "cpu" or len(devices) > 0)
    print(f"Main running with device(s): {devices}")

    from Dataset import ThreadedChunkDataset, worker_init_fn
    train_set = ThreadedChunkDataset(
        batch_size=args.batch_size, 
        chunk_directory=args.chunks_directory, 
        buffer_size=args.chunk_buffer_size, 
        sample_size=args.sample_size,
        seed=args.seed)
    
    model = _get_model_instance_from_file(args.model_path)
    
    optimizer = torch.optim.Adam(model.parameters()) # TODO: Use distributed optimizer

    from torch.utils.data import DataLoader
    train_data = DataLoader(
        train_set,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=None,
        shuffle=False,
        pin_memory=True, # pin_memory_device not set so default is cpu 
        prefetch_factor=2,
        persistent_workers=True,
    )

    from Trainer import Trainer
    trainer = Trainer(model, train_data, optimizer, args.save_every, args.snapshot_path)
    trainer.train(args.total_epochs)

    destroy_process_group()

if __name__ == "__main__":
    args = parse_args()
    main(args)