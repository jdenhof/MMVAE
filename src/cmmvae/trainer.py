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
        bf_data = []
        bf_data_cg = []
        bf_md = []
        bf_md_cg = []
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

                bf_data.append(sampleAtoA)
                bf_data.append(sampleBtoB)
                bf_md.append(metadataA)
                bf_md.append(metadataB)

                sampleAtoB = self.model.cross_generate(sampleA, metadataA, metadataB)
                sampleBtoA = self.model.cross_generate(sampleB, metadataB, metadataA)

                bf_data_cg.append(sampleAtoB)
                bf_data_cg.append(sampleBtoA)
                bf_md_cg.append(metadataAtoB)
                bf_md_cg.append(metadataBtoA)

                if metric == "cosine":
                    column_scores[column] += 0.5 * (
                        r2_score(sampleAtoB, sampleBtoB)
                        + r2_score(sampleBtoA, sampleAtoA)
                    )
                else:
                    raise ValueError(f"Unsupported metric: {metric}")

            if any(len(d) > n_buffer for d in (bf_data, bf_data_cg, bf_md, bf_md_cg)):
                for k, d, md in (
                    ("normal", bf_data, bf_data_cg),
                    ("cross", bf_data_cg, bf_md_cg),
                ):
                    data = torch.Tensor(d).item()
                    mdata = pd.concat(md)
                    h5File.save(target_file, k, RK.XHAT, data, mdata)
                    d.clear()
                    md.clear()

        for k, d, md in (
            (RK.XHAT, bf_data, bf_data_cg),
            (f"{RK.XHAT}_cross", bf_data_cg, bf_md_cg),
        ):
            data = torch.Tensor(d).item()
            mdata = pd.concat(md)
            h5File.save(target_file, key, k, data, mdata)

        for col in column_scores:
            column_scores[col] /= len(column_scores[col])
        return column_scores
