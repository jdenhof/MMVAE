from typing import Optional, Type, Union
import os
import pandas as pd
import numpy as np
import h5py
from cmmvae.constants import REGISTRY_KEYS as RK
import logging

logger = logging.getLogger(__name__)

def is_iterable(obj):
    """
    Check if an object is iterable.

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object is iterable, False otherwise.
    """
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def replace_inf(data: np.ndarray, dtype: Type[np.floating] = np.float32) -> np.ndarray:
    """Replace all values that are inf to float"""
    max_f4 = np.finfo(dtype).max
    min_f4 = np.finfo(dtype).min
    data[np.isposinf(data)] = max_f4
    data[np.isneginf(data)] = min_f4
    data = data.astype(dtype)
    return data


class h5File:

    DATA: str = RK.DATA
    METADATA: str = RK.METADATA
    UMAP_EMBEDDINGS: str = RK.UMAP_EMBEDDINGS

    @staticmethod
    def _get_or_raise(obj, key, dtype):
        result = obj.get(key)
        if isinstance(result, dtype):
            return result
        else:
            raise KeyError(f"{key} not a dataset in {obj}")

    @staticmethod
    def _get_group(obj, key) -> h5py.Group:
        return h5File._get_or_raise(obj, key, h5py.Group)

    @staticmethod
    def get_group(obj: Union[h5py.File, h5py.Group], key: str) -> Optional[h5py.Group]:
        try:
            return h5File._get_group(obj, key)
        except KeyError as e:
            logger.debug(f"Key could not be found: {e}")

    @staticmethod
    def _get_dataset(obj: Union[h5py.File, h5py.Group], key: str) -> h5py.Dataset:
        return h5File._get_or_raise(obj, key, h5py.Dataset)

    @staticmethod
    def get_dataset(obj: Union[h5py.File, h5py.Group], key: str) -> Optional[h5py.Dataset]:
        try:
            return h5File._get_dataset(obj, key)
        except KeyError as e:
            logger.debug(f"Key could not be found: {e}")

    @staticmethod
    def get_data(group: h5py.Group):
        return h5File.get_dataset(group, h5File.DATA)

    @staticmethod
    def create_data(group: h5py.Group):
        return group.create_dataset(h5File.DATA, chunks=True)

    @staticmethod
    def get_metadata(group: h5py.Group):
        return h5File.get_group(group, h5File.METADATA)

    @staticmethod
    def create_metadata(group: h5py.Group):
        return group.create_group(h5File.METADATA)

    @staticmethod
    def get_umap_embeddings(group: h5py.Group):
        return h5File.get_dataset(group, h5File.UMAP_EMBEDDINGS)

    @staticmethod
    def as_dataframe(metadata: Optional[h5py.Group]):
        if metadata is not None:
            return pd.DataFrame(
                {col: h5File.get_dataset(metadata, col) for col in metadata.keys()}
            )

    @staticmethod
    def load(file_path: str, key: str):
        logger.debug(f"Loading h5py: {key} - {file_path}")
        with h5py.File(file_path) as h5file:
            group = h5File._get_group(h5file, key)
            return {
                h5File.DATA: h5File.get_data(group),
                h5File.METADATA: h5File.as_dataframe(h5File.get_metadata(group)),
                h5File.UMAP_EMBEDDINGS: h5File.get_umap_embeddings(group)
            }

    @staticmethod
    def write(file_path: str, data: np.ndarray, metadata: pd.DataFrame, key: str):
        with h5py.File(file_path, 'a') as h5file:
            group = h5File.get_group(h5file, key) or h5file.create_group(key)
            h5File._append_data(
                ds=h5File.get_data(group) or h5File.create_data(group),
                data=data)
            h5File._append_metadata(
                group=h5File.get_metadata(group) or h5File.create_metadata(group),
                metadata=metadata)

    @staticmethod
    def _append_data(ds: h5py.Dataset, data: np.ndarray):
        new_size = ds.shape[0] + data.shape[0]
        ds.resize(new_size, axis=0)
        ds[-data.shape[0] :] = data  # Append new batch

    @staticmethod
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
            column_ds = h5File.get_dataset(group, str(col)) or group.create_dataset(str(col), chunks=True)
            new_size = column_ds.shape[0] + metadata.shape[0]
            column_ds.resize(new_size, axis=0)
            column_ds[-len(column_data) :] = column_data
