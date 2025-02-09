from typing import Iterable, Optional
import os
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_images_to_tensorboard(
    tensorboard_dir: str,
    image_paths: Iterable[str],
    tag: Optional[str] = None,
    step: int = 0,
):
    """
    Add images to Tensorboard.

    Args:
        log_dir (str): Path to Tensorboard log directory.
        image_paths (list[str]): Paths to images to load to Tensorboard.
    """
    from torch.utils.tensorboard.writer import SummaryWriter
    writer = SummaryWriter(log_dir=tensorboard_dir)
    for path in image_paths:
        image = get_image_as_tensor(path)
        writer.add_image(tag or get_image_tag(path), image, global_step=step)
        logger.debug(f"Added image to tensorboard: {get_image_tag(path)}\n\t{path}\n")
    writer.close()

def get_image_as_tensor(image_path: str):
    from PIL import Image
    image = Image.open(image_path)
    image = torch.tensor(np.array(image)).permute(2, 0, 1)
    return image

def get_image_tag(path: str) -> str:
    path = os.path.normpath(path)
    components = path.split(os.sep)
    return os.path.join(*components[-2:])