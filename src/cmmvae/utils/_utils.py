from typing import Type
import numpy as np

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