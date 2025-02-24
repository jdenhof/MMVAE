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

def load_legacy(file_path: str, key: str, data: bool = True, metadata: bool = True, embeddings: bool = True):
    with h5py.File(file_path, swmr=True) as h5file:
        logger.debug(f"h5file handler opened...")
        group = _get_group(h5file, key)
        logger.debug(f"found group for {key}...")
        for key, value in group.items():
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

def save(
    file_path: str,
    gkey: str,
    key: str,
    data: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    mode: str = 'a',
    metadata_key: Optional[str] = None,
):
    if all(p is None for p in (data, metadata)):
        raise RuntimeError("Must pass either data or metadata to save that is not None!")

    with h5py.File(file_path, mode, swmr=True) as h5file:
        group = get_group(h5file, gkey) or h5file.create_group(gkey)
        group = get_group(group, key) or gkey.create_group(key)
        if data is not None:
            _append(
                ds=get_dataset(group, RK.DATA) or group.create_dataset(RK.DATA, chunks=True),
                data=data,
                size=data.shape[0])
        if metadata is not None:
            metadata_group=get_group(group, RK.METADATA) or group.create_group(RK.METADATA),
            for col in metadata.columns:
                column_ds = get_dataset(metadata_group, col) or metadata_group.create_dataset(col, chunks=True)
                data = metadata[col].values.tolist()
                _append(column_ds, data, len(data))

def _append(ds: h5py.Dataset, data: Container, size: int):
    new_size = ds.shape[0] + size
    ds.resize(new_size, axis=0)
    ds[-size :] = data