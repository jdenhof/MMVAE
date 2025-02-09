import argparse
from typing import Literal, Union
import numpy as np
import pandas as pd

from cmmvae.data.local.grouped_index_lookup import GroupedIndexLookup
from cmmvae.utils import h5File
from cmmvae.constants import REGISTRY_KEYS as RK
import logging

logger = logging.getLogger(__name__)

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
):
    logger.debug(f"Computing scores for: {data}")
    scores = {col: {m: 0 for m in ("intra_A", "intra_B", "inter_AB", "separation_score")} for col in lookup.columns}
    for group in lookup.get_groups():
        indicesA, indicesB = group.data
        dataA = data[indicesA]
        dataB = data[indicesB]
        score = separation_score(dataA, dataB, metric=metric)
        logger.debug("Score: ", score)
        for m in scores[group.varying_column]:
            scores[group.varying_column][m] += score[m]
    result = {m: sum(group[m] for group in scores.values()) for m in scores[next(iter(scores))]}
    return {
        "metric": metric,
        "group": scores,
        "total": result
    }

def compute(
    data: np.ndarray,
    df: pd.DataFrame,
    columns: list[str],
    metric: SIMULARITY_METRIC = "euclidean"
):
    print("Computing GroupedIndexLookup...")
    lookup = GroupedIndexLookup(df, columns=columns)
    print("Computing scores...")
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
        print("Loaded:", prediction)
        results = compute(prediction[RK.DATA], prediction[RK.METADATA], columns, metric=metric)
        pd.DataFrame(results["group"]).to_csv(f"{key}_group_{output_path}")
        pd.DataFrame(results["total"]).to_csv(f"{key}_total_{output_path}")


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
