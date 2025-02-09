from typing import Any, Optional, Sequence
import os
import warnings
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
import pandas as pd
import numpy as np
import torch

import cmmvae.utils as utils


class PredictionWriter(BasePredictionWriter):
    def __init__(
        self,
        root_dir: str,
        experiment_name: str = "",
        run_name: str = "",
        hdf5_filename: str = "predictions.h5",
    ):
        super().__init__(write_interval="batch")
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.hdf5_filename = hdf5_filename
        self._curr_size = 0  # Keeps track of total rows written so far

    @property
    def save_dir(self):
        return os.path.join(self.root_dir, self.experiment_name, self.run_name)

    @property
    def hdf5_filepath(self):
        return os.path.join(self.save_dir, self.hdf5_filename)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        if not isinstance(prediction, dict) or not all(
            isinstance(p, (tuple, list))
            and len(p) == 2
            and (
                isinstance(p[0], (torch.Tensor, np.ndarray))
                and isinstance(p[1], pd.DataFrame)
            )
            for p in prediction.values()
        ):
            raise ValueError(
                f"Prediction must be a dictionary of type 'dict[str, tuple[torch.Tensor, pd.DataFrame]]' (got {type(prediction)})"
            )

        for key, (data, metadata) in prediction.items():
            data = data.cpu().numpy() if isinstance(data, torch.Tensor) else data
            data = utils.replace_inf(data)
            utils.h5File.append_batch(self.hdf5_filepath, key, data, metadata)

        self._curr_size += batch[0].shape[
            0
        ]  # Increment by batch size

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        n = 0
        while os.path.exists(self.hdf5_filepath):
            if n == 0:
                warnings.warn(
                    f"PredictionWriter initialized with hdf5_filepath that already exists: {self.hdf5_filepath}"
                )
            n += 1
            self.hdf5_filename = f"{self.hdf5_filename[:1]}{n}"
        os.makedirs(self.save_dir, exist_ok=True)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._curr_size = 0  # Reset after the epoch ends
