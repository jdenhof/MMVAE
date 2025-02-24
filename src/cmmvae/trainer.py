import os
import pickle
import pandas as pd
import torch
import numpy as np
import lightning.pytorch as pl
import scipy.sparse as sp
from cmmvae.data.local import GroupedIndexLookup
from cmmvae.utils._metrics import r2_score
from cmmvae.utils import h5File

from cmmvae.constants import REGISTRY_KEYS as RK


class CMMVAETrainer(pl.Trainer):
    @torch.inference_mode()
    def cross_generate(
        self,
        source_file: str,
        df_file: str,
        columns: list[str],
        iterations: int,
        metric: str = "cosine",
        n_buffer: int = 1000,
        target_file: str = "cross_generate_stats.h5",
    ):
        target_file = os.path.join(self.logger.root_dir, target_file)
        with open(source_file, "rb") as npz_file:
            source = sp.load_npz(npz_file)

        with open(df_file, "r") as metadata_file:
            df = pickle.load(metadata_file)

        lookup = GroupedIndexLookup(df, columns=columns)
        self.model.eval()
        column_scores = {col: 0 for col in columns}
        buffer = {"normal": [[],[]], "cross": [[],[]]}
        for i in range(iterations):
            for column in columns:
                result = lookup.get_random_1_contexts_change()
                index_A, index_B = result.data
                sampleA, metadataA = source[index_A], df[index_B]
                sampleB, metadataB = source[index_B], df[index_B]

                metadataAtoB = metadataB
                metadataBtoA = metadataA

                sampleAtoA = self.model.cross_generate(sampleA, metadataA, metadataA)
                sampleBtoB = self.model.cross_generate(sampleB, metadataB, metadataB)

                buffer["normal"][0].extend((sampleAtoA, sampleBtoB))
                buffer["normal"][1].extend((metadataA, metadataB))

                sampleAtoB = self.model.cross_generate(sampleA, metadataA, metadataB)
                sampleBtoA = self.model.cross_generate(sampleB, metadataB, metadataA)

                buffer["cross"][0].extend((sampleAtoB, sampleBtoA))
                buffer["cross"][1].extend((metadataAtoB, metadataBtoA))

                if metric == "cosine":
                    column_scores[column] += 0.5 * (
                        r2_score(sampleAtoB, sampleBtoB)
                        + r2_score(sampleBtoA, sampleAtoA)
                    )
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            if any(len(d[0]) > n_buffer or len(d[1]) > n_buffer for d in buffer.values()):
                save_buffer(buffer, target_file)

        save_buffer(buffer, target_file)

        for col in column_scores:
            column_scores[col] /= len(column_scores[col])
        return column_scores

def save_buffer(buffer file):
    for key, buff in buffer.items():
        data = torch.Tensor(buff[0]).item()
        mdata = pd.concat(buff[1])
        h5File.save(target_file, k, RK.XHAT, data, mdata)
        buff[0].clear()
        buff[1].clear()