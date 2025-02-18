from typing import Literal, Union
import argparse
import pickle
from cmmvae.models.cmmvae_model import CMMVAEModel
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cmmvae.cli import CMMVAECli
from cmmvae.data.local.grouped_index_lookup import GroupedIndexLookup
from cmmvae.utils import h5File
from cmmvae.constants import REGISTRY_KEYS as RK
import logging

logger = logging.getLogger("cmmvae.cross_generation_score")

SIMULARITY_METRIC = Union[Literal["cosine"], Literal["euclidean"]]


def compute(
    key: str,
    model: CMMVAEModel,
    source_file: str,
    target_file: str,
    df_file: str,
    columns: list[str],
    iterations: int,
    metric: str = "cosine"
):
    with open(source_file, "rb") as npz_file:
        source = sp.load_npz(npz_file)
    with open(target_file, "rb") as npz_file:
        target = h5File.load(npz_file, key, embeddings=False)
    with open(df_file, "r") as metadata_file:
        df = pickle.load(metadata_file)

    model.cross_generation_score(
        source = source,
        target = target,
        df = df,
        columns = columns,
        iteration = iterations,
        metric = metric,
    )

def main(
    model: CMMVAEModel,
    source_file: str,
    target_file: str,
    df_file: str,
    iterations: int,
    keys: list[str],
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean",
    output_path: str = "cross_generation_scores.csv"
):
    for key in keys:
        results = compute(
            key, model, source_file, target_file, df_file, columns, iterations, metric
        )
        results.to_csv(f"{key}_" + output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str,
                        help="File path for hdf5 predictions.")
    parser.add_argument("--target_file", type=str,
                        help="File path for hdf5 predictions.")
    parser.add_argument("--df_file", type=str,
                        help="File path for the source metadata")
    parser.add_argument("--iterations", type=str,
                        help="Number of iterations to run.")
    parser.add_argument("--keys", nargs='+', type=str,
                        help="Keys for h5file for sampling ('z', 'xhat') or others")
    parser.add_argument("--columns", nargs='+', type=str, help="List of columns of variation.")
    parser.add_argument("--metric", type=str, choices=["cosine", "euclidean"], default="euclidean")
    parser.add_argument("--threshold", type=int, default=1,
                        help="Threshold to limit group size.")
    parser.add_argument("--output_path", type=str, default="seperation_scores.csv")
    args = parser.parse_args()
    cli = CMMVAECli(run=False, args=args)
    main(
        model=cli.model,
        source_file=args.source_file,
        target_file=args.target_file,
        df_file=args.df_file,
        iterations=args.iterations,
        keys=args.keys,
        columns=args.columns,
        metric=args.metric,
        output_path=args.output_path,
    )