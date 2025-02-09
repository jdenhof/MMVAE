import argparse
from typing import Literal, Union
import numpy as np
import pandas as pd
import tqdm

from cmmvae.data.local.grouped_index_lookup import GroupedIndexLookup
from cmmvae.utils import h5File
from cmmvae.constants import REGISTRY_KEYS as RK
import logging

logger = logging.getLogger("cmmvae.seperation_scores")

SIMULARITY_METRIC = Union[Literal["cosine"], Literal["euclidean"]]


def compute_similarity_matrix(data: np.ndarray, metric: SIMULARITY_METRIC ="cosine"):
    """Compute similarity matrix using cosine similarity or Euclidean distance"""
    if metric == "cosine":
        normed_data = data / np.linalg.norm(data, axis=1, keepdims=True)
        similarity_matrix = np.dot(normed_data, normed_data.T)  # Cosine similarity
    elif metric == "euclidean":
        distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)  # Pairwise Euclidean distances
        similarity_matrix = -distances  # Convert distance to similarity (negative distances)
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    return similarity_matrix

def compute_intra_group_similarity(data: np.ndarray, metric: SIMULARITY_METRIC ="cosine"):
    """Compute average intra-group similarity"""
    sim_matrix = compute_similarity_matrix(data, metric)
    num_samples = data.shape[0]
    # Exclude self-similarity (diagonal for cosine, self-distance for Euclidean)
    intra_similarity = (sim_matrix.sum() - np.diag(sim_matrix).sum()) / (num_samples * (num_samples - 1))
    return intra_similarity

def compute_inter_group_similarity(data_A: np.ndarray, data_B: np.ndarray, metric="cosine"):
    """Compute average inter-group similarity between two sets"""
    if metric == "cosine":
        normed_A = data_A / np.linalg.norm(data_A, axis=1, keepdims=True)
        normed_B = data_B / np.linalg.norm(data_B, axis=1, keepdims=True)
        inter_similarity_matrix = np.dot(normed_A, normed_B.T)  # Cosine similarity
    elif metric == "euclidean":
        inter_similarity_matrix = -np.linalg.norm(data_A[:, np.newaxis] - data_B, axis=2)  # Convert distance to similarity
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

    inter_similarity = inter_similarity_matrix.mean()
    return inter_similarity

def separation_score(data_A: np.ndarray, data_B: np.ndarray, metric: SIMULARITY_METRIC = "cosine"):
    """Compute the separation score"""
    intra_A = compute_intra_group_similarity(data_A, metric)
    intra_B = compute_intra_group_similarity(data_B, metric)
    inter_AB = compute_inter_group_similarity(data_A, data_B, metric)
    score = (intra_A + intra_B) / inter_AB
    return {
        "metric": metric,
        "intra_A": intra_A,
        "intra_B": intra_B,
        "inter_AB": inter_AB,
        "separation_score": score
    }

def _compute_scores(
    data: np.ndarray,
    lookup: GroupedIndexLookup,
    metric: SIMULARITY_METRIC,
    progress_bar: bool = True,
):
    metrics = ("intra_A", "intra_B", "inter_AB", "separation_score")
    columns = metrics + ("varying_column", "row_key")
    df = pd.DataFrame(columns=columns)
    groups = list(lookup.get_groups())
    total_groups = len(groups)
    pbar = tqdm.tqdm(total=total_groups, desc="Computing scores", unit="group") if progress_bar else None
    for i, group in enumerate(groups):
        if pbar is not None and i % 100 == 0:
            pbar.set_postfix({
                "Score": df["separation_score"].mean()
            })
        score = separation_score(data[group.data[0]], data[group.data[1]], metric=metric)
        df.loc[len(df)] = [score[m] for m in metrics] + [group.varying_column, group.row_key]
        if pbar is not None:
            pbar.update()
    if pbar is not None:
        pbar.close()
    return df

def compute(
    data: np.ndarray,
    df: pd.DataFrame,
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean"
):
    logger.info(f"Computing GroupedIndexLookup...")
    lookup = GroupedIndexLookup(df, columns=columns)
    logger.info("Computing scores...")
    return _compute_scores(data, lookup, metric=metric)


def main(
    file_path: str,
    keys: list[str],
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean",
    output_path: str = "seperation_scores.csv"
):
    """
    Evaluates Cross-Generation Performance.

    This script evaulates the performance of the cross-generation capabilities
    of the model by comparing the generated samples from known metadata labels.

    Given a metadata combination we seek to see how similar cross-generated
    samples are from known samples of that combination.

    ie.
    Input | Target | Known
       A       B       B
       A       A       A
       A       A       A

    We cross genererate from Input label to Target label and then compare the distance between
    all samples of Known to others to getting a metric of closeness to Known and further from other labels.
    We then do the same vice versa where the Input becomes Target and Target and Known become the Input.
    """
    for key in keys:
        prediction = h5File.load(file_path, key, embeddings=False)
        logger.debug("Loaded:", prediction)
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
    main(
        file_path=args.file_path,
        keys=args.keys,
        columns=args.columns,
        metric=args.metric,
        output_path=args.output_path,
    )
