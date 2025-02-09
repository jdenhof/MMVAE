from typing import Container, Union, Optional
import logging
import numpy as np
import pandas as pd
from pandas._typing import ArrayLike
import h5py
from cmmvae.utils._logging import log_method_decorator
from cmmvae.constants import REGISTRY_KEYS as RK
import logging
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

def as_dataframe(metadata: h5py.Group):
    return pd.DataFrame(
        {col: _get_dataset(metadata, col) for col in metadata.keys()}
    )

@log_method_decorator(logger)
def load(file_path: str, key: str, data: bool = True, metadata: bool = True, embeddings: bool = True):
    with h5py.File(file_path, swmr=True) as h5file:
        logger.debug(f"h5file handler opened...")
        group = _get_group(h5file, key)
        logger.debug(f"found group for {key}...")
        _data = get_dataset(group, RK.DATA) if data else None
        if _data is not None:
            logger.debug("splicing data...")
            _data = _data[:]
        _metadata = get_group(group, RK.METADATA) if metadata else None
        if _metadata is not None:
            logger.debug("converting metadata to dataframe...")
            _metadata = as_dataframe(_metadata)
        _embeddings = get_dataset(group, RK.UMAP_EMBEDDINGS) if embeddings else None
        if embeddings and _embeddings is not None:
            logger.debug("splicing umap_embeddings...")
            _embeddings = _embeddings[:]
        logger.debug(f"returning...{_data}, {_metadata}, {_embeddings}")
        return {
            RK.DATA: _data,
            RK.METADATA: _metadata,
            RK.UMAP_EMBEDDINGS: _embeddings
        }

def _write(
    file_path: str,
    key: str,
    data: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    mode: str = 'w'
):
    assert all(p is not None for p in (data, metadata, embeddings))
    with h5py.File(file_path, mode, swmr=True) as h5file:
        group = get_group(h5file, key) or h5file.create_group(key)
        if embeddings is not None:
            _append_data(
                ds=get_dataset(group, RK.UMAP_EMBEDDINGS) or group.create_dataset(RK.UMAP_EMBEDDINGS, chunks=True),
                data=embeddings,
                size=embeddings.shape[0]
            )
        if data is not None:
            _append_data(
                ds=get_dataset(group, RK.DATA) or group.create_dataset(RK.DATA, chunks=True),
                data=data,
                size=data.shape[0])
        if metadata is not None:
            _append_metadata(
                group=get_group(group, RK.METADATA) or group.create_group(RK.METADATA),
                metadata=metadata)

def add_embeddings(file_path: str, key: str, embeddings: np.ndarray):
    _write(file_path, key, embeddings=embeddings,mode='a')

def append(file_path: str, key: str, data: np.ndarray, metadata: pd.DataFrame):
    _write(file_path, key, data=data, metadata=metadata, mode='a')

def _append_data(ds: h5py.Dataset, data: Container, size: int):
    new_size = ds.shape[0] + size
    ds.resize(new_size, axis=0)
    ds[-size :] = data

def _append_metadata(group: h5py.Group, metadata: pd.DataFrame, strict: bool = True):
    for col in metadata.columns:
        if strict and col not in group:
            raise RuntimeError(f"metadata column {col} not in h5File")
        column_ds = get_dataset(group, col) or group.create_dataset(col, chunks=True)
        data = _array_like_to_list(metadata[col].values)
        _append_data(column_ds, data, len(data))

def _array_like_to_list(array_like: ArrayLike):
    if hasattr(array_like, "tolist"):
        return array_like.tolist()
    elif hasattr(array_like, "to_list"):
        return array_like.to_list() # type: ignore
    elif not isinstance(array_like, list):
        raise TypeError("'column_data' must be of type list")
    return array_like
