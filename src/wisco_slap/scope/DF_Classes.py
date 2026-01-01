from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable, Any
import polars as pl

TimeLike = Union[int, float]


@dataclass
class SynDF:
    df: pl.DataFrame

    # ====== Custom methods go here ======

    def ts(self, t1: TimeLike, t2: TimeLike) -> SynDF:
        """
        Filter to time between t1 and t2 (inclusive).
        Assumes a column named 'time'.
        """
        mask = (pl.col("time") >= t1) & (pl.col("time") <= t2)
        return SynDF(self.df.filter(mask))

    def ss(self, dmd: int, source_id: int) -> SynDF:
        """
        Filter to a specific dmd and source (i.e. a 'single source' synapse).
        Assumes a column named 'dmd' and 'source-ID'.
        """
        mask = (pl.col("dmd") == dmd) & (pl.col("source-ID") == source_id)
        return SynDF(self.df.filter(mask))

    def dmd(self, dmd: int) -> SynDF:
        """
        Filter to a specific dmd.
        Assumes a column named 'dmd'.
        """
        mask = pl.col("dmd") == dmd
        return SynDF(self.df.filter(mask))

    def sid(self, source_id: int) -> SynDF:
        """
        Filter to a specific source-ID.
        Assumes a column named 'source-ID'.
        """
        mask = pl.col("source-ID") == source_id
        return SynDF(self.df.filter(mask))

    # ====== Delegation layer ======

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute/method access to the underlying pl.DataFrame.
        If a DataFrame method returns a pl.DataFrame, wrap it back in SynDF.
        """
        # This will only be called if the attribute wasn't found on SynDF itself.
        attr = getattr(self.df, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If a Polars method returns a DataFrame, wrap it so we keep SynDF
                if isinstance(result, pl.DataFrame):
                    return SynDF(result)
                return result

            # Preserve the original method's __name__ etc. if you care about introspection
            wrapper.__name__ = getattr(attr, "__name__", name)
            wrapper.__doc__ = getattr(attr, "__doc__", None)
            return wrapper

        # Non-callable attribute (like .schema, .columns, .height, etc.)
        return attr

    # Optional: some dunder methods to make it feel more DataFrame-like
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item) -> Any:
        return self.df[item]

    def __iter__(self):
        return iter(self.df)

    def __repr__(self) -> str:
        return f"SynDF(\n{self.df!r}\n)"


@dataclass
class SomaDF:
    """
    Thin wrapper around a Polars DataFrame containing soma fluorescence traces.

    - Keeps the full Polars API via delegation
    - Adds convenience methods like .ts(...)
    """

    df: pl.DataFrame

    # ====== Custom methods ======

    def ts(self, t1: TimeLike, t2: TimeLike) -> SomaDF:
        """
        Return a new SomaDF with rows where 'time' is between t1 and t2 (inclusive).

        Assumes a numeric column named 'time' in the underlying DataFrame.
        """
        mask = (pl.col("time") >= t1) & (pl.col("time") <= t2)
        return SomaDF(self.df.filter(mask))

    # ====== Delegation layer to underlying pl.DataFrame ======

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute/method access to the underlying pl.DataFrame.

        If a delegated method returns a pl.DataFrame, wrap it back into SomaDF
        so method chaining preserves the SomaDF type.
        """
        attr = getattr(self.df, name)

        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                if isinstance(result, pl.DataFrame):
                    return SomaDF(result)
                return result

            wrapper.__name__ = getattr(attr, "__name__", name)
            wrapper.__doc__ = getattr(attr, "__doc__", None)
            return wrapper

        return attr

    # ====== Some dunders for DataFrame-like behavior ======

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item) -> Any:
        return self.df[item]

    def __iter__(self):
        return iter(self.df)

    def __repr__(self) -> str:
        return f"SomaDF(\n{self.df!r}\n)"
