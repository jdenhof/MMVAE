from typing import Callable, Iterable, Optional, Iterator, Container
from dataclasses import dataclass
import itertools

import pandas as pd


@dataclass
class GroupedIndexResult:
    data: tuple[pd.Index, pd.Index]
    varying_column: str
    group_key: str
    row_key: str


class GroupedIndexLookup:
    def __init__(self,
        df: pd.DataFrame,
        columns: Optional[Iterable[str]] = None,
        optimize_memory: bool = True,
    ):
        self.df = df
        self.optimize_memory = optimize_memory
        # Columns are stored as tuple to maintain order for keys
        self.columns = tuple(columns or self.df.columns or [])
        for col in self.columns:  # Columns used in filtering
            df[col] = df[col].astype("category")
        assert self.columns, "Columns cannot be empty!"
        self.index_dicts = self._build_index_dicts()
        self.index_dicts = self.filter_by_length(1)

    def _build_index_dicts(self):
        """Build the dictionary mapping column subsets to row indices."""
        index_dicts = {}
        for subset in itertools.combinations(self.columns, len(self.columns) - 1):

            key = self.get_columns_in(subset)
            varying_columns = self.get_columns_not_in(key)

            group_obj = self.df.groupby(list(key), observed=True)
            groups = group_obj.groups

            index_dicts[key] = {}
            if not self.optimize_memory:
                index_dicts[key]["identical"] = {}
            for group, indices in groups.items():
                if self.df.iloc[indices][varying_columns[0]].nunique() > 1:
                    index_dicts[key][group] = indices.to_list() # type: ignore
                elif not self.optimize_memory:
                    index_dicts[key]["identical"][group] = indices

        return index_dicts

    def get_columns_in(self, columns: Container[str]):
        return tuple(c for c in self.columns if c in columns)

    def get_columns_not_in(self, columns: Container[str]):
        return tuple(c for c in self.columns if c not in columns)

    def equal(self, indices: pd.Index, column: str):
        valid_indices = indices[indices < len(self.df)]  # Keep only valid indices
        if len(valid_indices) != len(indices):
            print(f"Warning: Some indices are out of bounds: {indices}")
        return self.df.iloc[valid_indices, :][column].nunique() == 1 \
            if len(valid_indices) > 0 else False

    def _filter_index_dicts(self, filter_fn: Callable[[list[int]], bool]):
        index_dicts: dict[tuple[str,...], dict[tuple[str,...], list[int]]] = {}
        for gkey, group_dict in self.index_dicts.items():
            index_dicts[gkey] = {
                rkey: indices for rkey, indices in group_dict.items()
                if filter_fn(indices)
            }
        return index_dicts

    def filter_by_length(self, length: int):
        """Filter index dictionaries to only include groups > 'threshold'"""
        if length < 1:
            raise ValueError(f"Filter 'length' must be greater than 1: {length}")
        return self._filter_index_dicts(lambda a: len(a) > length)

    def get_matching_indices(self, row_index: int, varying_columns: Container[str]):
        """Find all indices that differ along the columns specified."""
        assert isinstance(row_index, int), "row_index must be an 'int'"
        assert isinstance(varying_columns, list), "varying_columns must be a 'list'"
        column_key = self.get_columns_not_in(varying_columns)
        row_values = self.df.loc[row_index, slice(column_key)]
        row_key = tuple(row_values)
        indices = self.index_dicts.get(column_key, {}).get(row_key, [])
        return indices

    def get_matching_rows(self, row_index: int, varying_columns: Container[str]):
        """Return the matching rows as a DataFrame."""
        indices = self.get_matching_indices(row_index, varying_columns)
        return self.df.iloc[indices] if isinstance(indices, pd.Index) else pd.DataFrame()

    def group(self, varying_columns: Container[str]):
        column_key = self.get_columns_not_in(varying_columns)
        return self.index_dicts.get(column_key, {})

    def get_groups(self) -> Iterable[GroupedIndexResult]:
        """
        Returns samples of all matching columns and all one off pertabations.
        """
        for gkey, group in self.index_dicts.items():
            for rkey, indices in group.items():
                varying_column = self.get_columns_not_in(gkey)[0]
                group = self.df.iloc[indices, :].groupby(varying_column)
                groups = [v for v in group.groups.values() if len(v) > 1]
                for groupA, groupB in itertools.combinations(groups, 2):
                    yield GroupedIndexResult(
                        data=(groupA, groupB),
                        varying_column=varying_column,
                        group_key=gkey,
                        row_key=rkey,
                    )