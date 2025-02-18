import argparse
from typing import Literal, Union
import numpy as np
import pandas as pd
import random

from cmmvae.cli import CMMVAECli
from cmmvae.data.local.grouped_index_lookup import GroupedIndexLookup
from cmmvae.utils import h5File
from cmmvae.constants import REGISTRY_KEYS as RK
import logging

logger = logging.getLogger("cmmvae.cross_generation_score")

SIMULARITY_METRIC = Union[Literal["cosine"], Literal["euclidean"]]

def compute(
    data: np.ndarray,
    df: pd.DataFrame,
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean",
    iterations: int = 1000
) -> pd.DataFrame:


def main(
    file_path: str,
    keys: list[str],
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean",
    output_path: str = "cross_generation_scores.csv"
):
    for key in keys:
        prediction = h5File.load(file_path, key, embeddings=False)
        logger.debug("loaded:", prediction)
        results = compute(prediction[RK.DATA], prediction[RK.METADATA], columns, metric=metric)
        results.to_csv(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str,
                        help="File path for hdf5 predictions.")
    parser.add_argument("--keys", nargs='+', type=str,
                        help="Keys for h5file for sampling ('x', 'xhat') or others")
    parser.add_argument("--columns", nargs='+', type=str, help="List of columns of variation.")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="euclidean")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Threshold to limit group size.")
    parser.add_argument("--output_path", type=str, default="seperation_scores.csv")
    args = parser.parse_args()
    cli = CMMVAECli(run=False)
    main(
        file_path=args.file_path,
        keys=args.keys,
        columns=args.columns,
        metric=args.metric,
        output_path=args.output_path,
    )