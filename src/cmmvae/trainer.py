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
        key: str,
        source_file: str,
        target_file: str,
        df_file: str,
        columns: list[str],
        iterations: int,
        metric: str = "cosine",
        n_buffer: int = 1000,
    ):
        with open(source_file, "rb") as npz_file:
            source = sp.load_npz(npz_file)
        target = h5File.load_legacy(target_file, key, embeddings=False)
        with open(df_file, "r") as metadata_file:
            df = pickle.load(metadata_file)

        lookup = GroupedIndexLookup(df, columns=columns)
        self.model.eval()
        column_scores = { col: 0 for col in columns }
        dbuffer = []
        mbuffer = []
        for i in range(iterations):
            for column in columns:
                result = lookup.get_random_1_contexts_change()
                index_A, index_B = result.data
                sampleA, metadataA = source[index_A], df[index_B]
                sampleB, metadataB = source[index_B], df[index_B]
                sampleAtoA = target[index_A]
                sampleBtoB = target[index_B]

                metadataAtoB = metadataB
                metadataBtoA = metadataA

                sampleAtoB = self.cross_generate(sampleA, metadataA, metadataB)
                sampleBtoA = self.cross_generate(sampleB, metadataB, metadataA)

                dbuffer.append(sampleAtoB)
                dbuffer.append(sampleBtoA)
                mbuffer.append(metadataAtoB)
                mbuffer.append(metadataBtoA)

                if metric == "cosine":
                    column_scores[column] += 0.5 * (r2_score(sampleAtoB, sampleBtoB) + r2_score(sampleBtoA, sampleAtoA))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            if len(dbuffer) > n_buffer or len(mbuffer) > n_buffer:
                data = torch.Tensor(dbuffer).item()
                mdata = pd.concat(mbuffer)
                h5File.save(target_file, key, "crossgen", data, mdata)

        for col in column_scores:
            column_scores[col] /= len(column_scores[col])
        return column_scores