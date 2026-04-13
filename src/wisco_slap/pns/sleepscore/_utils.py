"""Utility functions: scaling, imputation, and small helpers."""

from __future__ import annotations

import numpy as np
import polars as pl


def robust_scale_per_session(
    df: pl.DataFrame,
    feature_cols: list[str],
    session_col: str = "session_id",
) -> pl.DataFrame:
    """Robust-scale (median / IQR) each feature independently within each session.

    Parameters
    ----------
    df : DataFrame containing the features and a session identifier column.
    feature_cols : columns to scale.
    session_col : name of the session identifier column.

    Returns
    -------
    DataFrame with scaled feature columns (other columns unchanged).
    """
    parts = []
    for _name, group in df.group_by(session_col, maintain_order=True):
        scaled = group.clone()
        for col in feature_cols:
            vals = scaled[col]
            med = vals.median()
            q75 = vals.quantile(0.75)
            q25 = vals.quantile(0.25)
            iqr_val = q75 - q25 if (q75 is not None and q25 is not None) else None
            if iqr_val is None or iqr_val == 0:
                iqr_val = 1.0
            if med is None:
                med = 0.0
            scaled = scaled.with_columns(
                ((pl.col(col) - med) / iqr_val).alias(col)
            )
        parts.append(scaled)
    return pl.concat(parts) if parts else df


def impute_per_session(
    df: pl.DataFrame,
    feature_cols: list[str],
    session_col: str = "session_id",
) -> pl.DataFrame:
    """Fill NaN/null values with the per-session median for each feature.

    Parameters
    ----------
    df : DataFrame containing the features and a session identifier column.
    feature_cols : columns to impute.
    session_col : name of the session identifier column.

    Returns
    -------
    DataFrame with nulls filled.
    """
    parts = []
    for _name, group in df.group_by(session_col, maintain_order=True):
        filled = group.clone()
        for col in feature_cols:
            med = filled[col].median()
            if med is None:
                med = 0.0
            filled = filled.with_columns(pl.col(col).fill_null(med).fill_nan(med).alias(col))
        parts.append(filled)
    return pl.concat(parts) if parts else df


def segment_indices(ts: np.ndarray, start: float, end: float) -> slice:
    """Return slice indexing timestamps in [start, end)."""
    i0 = int(np.searchsorted(ts, start, side="left"))
    i1 = int(np.searchsorted(ts, end, side="left"))
    return slice(i0, i1)
