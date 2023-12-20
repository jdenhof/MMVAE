from utils import DEBUG
import argparse
import TrainerDefaults
from dataclasses import dataclass

@dataclass
class Args:
    model_path: str
    total_epochs: int
    batch_size: int
    seed: int
    sample_size: int
    chunk_buffer_size: int
    chunks_directory: str
    save_every: int
    snapshot_path: str
    num_workers: int
    num_chunk_workers: int
    world_size: int

def parse_args():

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
        f"{'--' if DEBUG else ''}world_size",
        type=int,
        help="Number of available devices" # Necessary as torch.cuda.device_count() always returns 2 as we have 2 gpus per gpu node
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
        '--num_workers', 
        type=str,
        help='Number of worker process for data loading with PyTorch DataLoader', 
        default=TrainerDefaults.NUM_WORKERS
    )
    parser.add_argument(
        '--num_chunk_workers', 
        type=str,
        help='Number of process for chunk data loading for ChunkLoader', 
        default=TrainerDefaults.NUM_CHUNK_WORKERS
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

    args = parser.parse_args(namespace=Args)
    if DEBUG:
        args.model_path = TrainerDefaults.MODEL_PATH
        args.world_size = 1

    return args