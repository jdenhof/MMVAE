import os
import pandas as pd

import cmmvae
import umap

logger = cmmvae.logging.getLogger(__name__)


def fit(
    X,
    n_neighbors=30,
    min_dist=0.3,
    n_components=2,
    metric="cosine",
    low_memory=False,
    n_jobs=40,
    n_epochs=200,
    **kwargs,
):
    logger.info("Fitting umap embeddings...\n")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        low_memory=low_memory,
        n_jobs=n_jobs,
        n_epochs=n_epochs,
        **kwargs,
    )
    embedding = reducer.fit_transform(X)
    logger.info("Done fitting umap embeddings.\n")
    return embedding

def plot_category(
    embedding,
    metadata,
    category,
    save_path,
    n_largest,
    name,
    method,
    alpha=0.5,
    marker_size=1,
) -> str:
    """
    Plot UMAP embeddings colored by a specific category.

    Args:
        embedding (np.ndarray): The UMAP embeddings.
        metadata (pd.DataFrame): The metadata associated with embeddings.
        category (str): Category to color by.
        save_path (str): Directory to save the plot.
        n_largest (int): Number of most common categories to plot.
        name (str): Name for the file.
        method (str): Method title to add to the graph.
        alpha (float): Opacity of plot points.
        marker_size (int): Size of plot points.

    Returns:
        str: Path to the saved plot image.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    unique_values = metadata[category].value_counts().nlargest(n_largest).index

    # Prepare color map
    cmap = plt.get_cmap("nipy_spectral", len(unique_values))
    color_list = [cmap(i) for i in range(len(unique_values))]

    # Combine embedding and metadata into a DataFrame
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df[category] = metadata[category].values

    # Filter to include only the largest categories
    df = df[df[category].isin(unique_values)]

    # Shuffle the DataFrame to randomize the plotting order
    df = df.sample(frac=1).reset_index(drop=True)

    # Create a dictionary to map categories to colors
    category_to_color = {value: color_list[i] for i, value in enumerate(unique_values)}

    # Map colors to the entire DataFrame
    df["color"] = df[category].map(category_to_color)

    # Plot all points in the shuffled order
    # with specified opacity and marker size
    plt.scatter(x=df["x"], y=df["y"], c=df["color"], s=marker_size, alpha=alpha)

    if method:
        method_str = f" for {method} "
    else:
        method_str = " "

    plt.title(f"UMAP projection{method_str}colored by {category}")

    # Custom legend with a circle for each label
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label.decode("utf-8") if isinstance(label, bytes) else label,
            markerfacecolor=cmap(i),
            markersize=10,
        )
        for i, label in enumerate(unique_values)
    ]

    plt.legend(
        handles=legend_handles,
        title=category,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    image_file_name = f"integrated.{category}.umap.{name}.png"
    image_path = os.path.join(save_path, image_file_name)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()
    return image_path


def main(**kwargs):
    """
    Plot UMAP embeddings and optionally log images to Tensorboard.

    Args:
        directory (str): Directory where embeddings to plot are stored.
        categories (tuple[str]): List of categories to color by.
        keys (tuple[str]): List of embedding keys that prefix save_paths.
        save_dir (str): Path to save UMAP outputs.
        skip_tensorboard (bool): Prevent logging UMAPs to Tensorboard.
    """
    generate_umap(**kwargs)


if __name__ == "__main__":
    main()
