def umap_embeddings(
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
    sys.stderr.write("Fitting umap embeddings...\n")
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
    sys.stderr.write("Done fitting umap embeddings.\n")
    return embedding


def plot_umap(
    directory,
    categories,
    keys,
    n_largest=15,
    method=None,
    save_dir=None,
    **umap_kwargs,
):
    """
    Generate UMAP embeddings and plot them.

    Args:
        directory (str): Directory where embeddings are stored.
        keys (list[str]): List of embedding keys.
        categories (list[str]): List of categories to color by.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP.
        n_components (int): Number of components for UMAP.
        metric (str): Metric for UMAP.
        low_memory (bool): Low memory setting for UMAP.
        n_jobs (int): Number of CPUs available for UMAP.
        n_epochs (int): Number of epochs to run UMAP.
        n_largest (int): Number of most common categories to plot.
        method (str): Method title to add to the graph.
        save_dir (str): Directory to save UMAP plots.
        **umap_kwargs: Extra kwargs passed to `umap.UMAP`.
    """

    if not save_dir:
        save_dir = directory

    image_paths = []
    for key in keys:
        umap_path = os.path.join(directory, f"{key}_umap_embeddings.npz")
        if os.path.exists(umap_path):
            npz_path = umap_path
            meta_path = os.path.join(directory, f"{key}_umap_metadata.pkl")
            embedding, metadata = load_embeddings(npz_path, meta_path)
        else:
            npz_path = os.path.join(directory, f"{key}_embeddings.npz")
            meta_path = os.path.join(directory, f"{key}_metadata.pkl")
            X, metadata = load_embeddings(npz_path, meta_path)

            # Fit and transform the data using UMAP
            embedding = umap_embeddings(X)
            embedding_file_name = f"{key}_umap_embeddings.npz"
            embedding_path = os.path.join(save_dir, embedding_file_name)
            metadata_file_name = f"{key}_umap_metadata.pkl"
            metadata_path = os.path.join(save_dir, metadata_file_name)
            os.makedirs(save_dir, exist_ok=True)
            np.savez(embedding_path, embeddings=embedding)
            metadata.to_pickle(metadata_path)

        for category in categories:
            image_path = plot_category(
                embedding, metadata, category, save_dir, n_largest, key, method
            )
            image_paths.append(image_path)
    return image_paths