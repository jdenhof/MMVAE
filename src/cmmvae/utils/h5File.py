from typing import Union, Optional
import logging
import numpy as np
import pandas as pd
import h5py

from cmmvae.constants import REGISTRY_KEYS as RK
logger = logging.getLogger(__name__)


def _get_or_raise(obj, key, dtype):
    result = obj.get(key)
    if isinstance(result, dtype):
        return result
    else:
        raise KeyError(f"{key} not a dataset in {obj}")


def _get_group(obj, key) -> h5py.Group:
    return _get_or_raise(obj, key, h5py.Group)


def get_group(obj: Union[h5py.File, h5py.Group], key: str) -> Optional[h5py.Group]:
    try:
        return _get_group(obj, key)
    except KeyError as e:
        logger.debug(f"Key could not be found: {e}")


def _get_dataset(obj: Union[h5py.File, h5py.Group], key: str) -> h5py.Dataset:
    return _get_or_raise(obj, key, h5py.Dataset)


def get_dataset(obj: Union[h5py.File, h5py.Group], key: str) -> Optional[h5py.Dataset]:
    try:
        return _get_dataset(obj, key)
    except KeyError as e:
        logger.debug(f"Key could not be found: {e}")


def get_data(group: h5py.Group):
    logger.debug(f"get_data: {group}")
    return get_dataset(group, RK.DATA)


def create_data(group: h5py.Group, *args, **kwargs):
    logger.debug(f"create_data: {group}")
    return group.create_dataset(RK.DATA, *args, **kwargs)


def get_metadata(group: h5py.Group):
    logger.debug(f"get_metadata: {group}")
    return get_group(group, RK.METADATA)


def create_metadata(group: h5py.Group, *args, **kwargs):
    logger.debug(f"create_metadata: {group}")
    return group.create_group(RK.METADATA, *args, **kwargs)


def get_umap_embeddings(group: h5py.Group):
    logger.debug(f"get_umap_embeddings: {group}")
    return get_dataset(group, RK.UMAP_EMBEDDINGS)


def create_umap_embeddings(group: h5py.Group, *args, **kwargs):
    logger.debug(f"create_umap_embeddings: {group}")
    return group.create_dataset(RK.UMAP_EMBEDDINGS, *args, **kwargs)


def as_dataframe(metadata: Optional[h5py.Group]):
    if metadata is not None:
        return pd.DataFrame(
            {col: get_dataset(metadata, col) for col in metadata.keys()}
        )


def load(file_path: str, key: str):
    logger.debug(f"Loading h5py: {key} - {file_path}")
    with h5py.File(file_path) as h5file:
        group = _get_group(h5file, key)
        logger.debug(f"Group: {group}")
        return {
            RK.DATA: get_data(group),
            RK.METADATA: as_dataframe(get_metadata(group)),
            RK.UMAP_EMBEDDINGS: get_umap_embeddings(group)
        }


def write(file_path: str, data: np.ndarray, metadata: pd.DataFrame, key: str):
    with h5py.File(file_path, 'a') as h5file:
        group = get_group(h5file, key) or h5file.create_group(key)
        _append_data(
            ds=get_data(group) or create_data(group),
            data=data)
        _append_metadata(
            group=get_metadata(group) or create_metadata(group),
            metadata=metadata)


def _append_data(ds: h5py.Dataset, data: np.ndarray):
    new_size = ds.shape[0] + data.shape[0]
    ds.resize(new_size, axis=0)
    ds[-data.shape[0] :] = data


def _append_metadata(group: h5py.Group, metadata: pd.DataFrame, strict: bool = True):
    for col in metadata.columns:
        col = str(col)
        column_data = metadata[col].values
        if hasattr(column_data, "tolist"):
            column_data = column_data.tolist()
        elif hasattr(column_data, "to_list"):
            column_data = column_data.to_list() # type: ignore
        elif not isinstance(column_data, list):
            raise TypeError("'column_data' must be of type list")
        if strict and col not in group:
                raise RuntimeError(f"metadata column {col} not in h5File")
        column_ds = get_dataset(group, str(col)) or group.create_dataset(str(col), chunks=True)
        new_size = column_ds.shape[0] + metadata.shape[0]
        column_ds.resize(new_size, axis=0)
        column_ds[-len(column_data) :] = column_data
