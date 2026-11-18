"""Implementation of the three-pass hypnogram substate labeller.

Public entry point: ``wis.peri.anno.label_hypnogram_substates``. This package
has no dependency on exploratory notebooks, local presets, or Loupe.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from electro_py.hypno import Hypnogram

from .active_wake import ActiveWakeConfig, build_active_wake_context, select_active_wake
from .nrem_sigma import NremSigmaConfig, build_nrem_sigma_context, label_nrem_substates
from .wake_merge import WakeMergeConfig, build_wake_merge_context, merge_wake_substates

ALGORITHM_VERSION = "hypnogram-substates-v1"


def normalize_hypnogram(hypno: Hypnogram | pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    """Validate broad-state intervals and standardize their column names."""
    frame = hypno.df if isinstance(hypno, Hypnogram) else hypno
    if isinstance(frame, pd.DataFrame):
        frame = pl.from_pandas(frame)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("hypno must be a Hypnogram or a polars/pandas DataFrame.")
    renames = {"start_time": "start_s", "end_time": "end_s", "state": "label"}
    frame = frame.rename({
        a: b
        for a, b in renames.items()
        if a in frame.columns and b not in frame.columns
    })
    if any(not frame.schema[c].is_numeric() for c in ["start_s", "end_s"]):
        raise ValueError("Hypnogram times must be numeric TDT seconds, not datetimes.")
    frame = frame.select(
        pl.col("start_s").cast(pl.Float64),
        pl.col("end_s").cast(pl.Float64),
        pl.col("label").cast(pl.String),
    ).sort("start_s")
    if frame.is_empty():
        raise ValueError("The input hypnogram is empty.")
    start, end = frame["start_s"].to_numpy(), frame["end_s"].to_numpy()
    if (
        np.any(~np.isfinite(start))
        or np.any(~np.isfinite(end))
        or np.any(end <= start)
        or np.any(start[1:] < end[:-1] - 1e-8)
        or frame["label"].null_count()
    ):
        raise ValueError(
            "Hypnogram intervals must be finite, positive, non-overlapping and labelled."
        )
    if set(frame["label"]) & {
        "Active wake",
        "Quiet wake",
        "NREM SWS",
        "NREM sigma",
        "Clear Active wake",
    }:
        raise ValueError(
            "Pass the original broad Wake/NREM/REM hypnogram, not already subsetted labels."
        )
    return frame


def merge_labels(frame: pl.DataFrame) -> pl.DataFrame:
    """Merge contiguous equal labels, without filling timestamp gaps."""
    frame = frame.select("start_s", "end_s", "label").sort("start_s")
    if frame.is_empty():
        return frame
    split = (pl.col("label") != pl.col("label").shift(1)) | (
        pl.col("start_s") > pl.col("end_s").shift(1) + 1e-8
    )
    return (
        frame
        .with_columns(split.fill_null(True).cum_sum().alias("_bout"))
        .group_by("_bout", maintain_order=True)
        .agg(pl.col("start_s").first(), pl.col("end_s").last(), pl.col("label").first())
        .drop("_bout")
    )


def _config(value: Any, cls: type) -> Any:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__}, a parameter mapping, or None.")


def label_hypnogram_substates(
    hypno: Hypnogram | pl.DataFrame | pd.DataFrame,
    eeg: xr.DataArray,
    eye: pl.DataFrame | None = None,
    whisk: pl.DataFrame | None = None,
    *,
    step1_config: ActiveWakeConfig | Mapping[str, Any] | None = None,
    step2_config: WakeMergeConfig | Mapping[str, Any] | None = None,
    step3_config: NremSigmaConfig | Mapping[str, Any] | None = None,
    grid_origin_s: float | None = None,
    window_s: float = 4.0,
    nrem_window_s: float = 4.0,
    eye_likelihood_min: float = 0.75,
    return_details: bool = False,
) -> Hypnogram | dict[str, Any]:
    """Apply the complete, approved Wake and NREM substate labeller.

    All inputs must share the native TDT sync-block time axis, in seconds.
    Use the full block for stable Wake/NREM calibration, then trim the result.
    EEG is required; pupil/whisking are optional. Load cell is never used.

    Parameters
    ----------
    hypno
        Original broad-state intervals: Hypnogram, or table with
        start_s/end_s/label (or start_time/end_time/state). REM and other
        original labels remain unchanged. Existing substate inputs are rejected.
    eeg
        One-dimensional EEG DataArray with increasing ``time`` and ``fs``.
        Nyquist must exceed 55 Hz. High-gamma/muscle routes are used when the
        sampling rate supports their full frequency bands; never upsampled.
    eye, whisk
        Polars frames on the same clock. Eye columns: time, motion, diameter,
        pup_likelihood, optionally lid_likelihood. Whisk columns: time, whis.
        Missing tracking provides no motion or stillness evidence.
    step1_config
        ActiveWakeConfig or partial dict: eeg_seed_fold=2, assisted_seed_fold=2,
        edge_relax_fraction=.1, slow_activity_max=.15, slow_fraction_max=.5,
        motion_z=3, seed_min_s=2, smooth_seconds=3, peak_fraction_max=.65.
    step2_config
        WakeMergeConfig or partial dict: bridge_budget_s=8, edge_budget_s=2,
        quiet_slow_min=.5, still_motion_max=1.5, quiet_weight=4, quiet_min_s=2,
        smooth_seconds=3, peak_fraction_max=.65.
    step3_config
        NremSigmaConfig or partial dict: sigma_slow_min=1, sigma_fraction_min=.28,
        support_fraction=.7, bridge_max_s=3, seed_min_s=2,
        sigma_beta_density_min=1.5, bridge_score_min=.5, slow_barrier_ratio=.5,
        smooth_seconds=3, peak_fraction_max=.65.
    grid_origin_s
        Phase of the one-second grid; defaults to the first original interval.
        Recording edges and off-grid original boundaries are also retained.
    window_s, nrem_window_s
        Spectral window lengths for wake and NREM measurements, respectively.
        Four seconds by default. Decisions remain on a one-second grid.
    eye_likelihood_min
        Pupil tracking quality cutoff (default .75).
    return_details
        False returns a Hypnogram of final bouts. True returns a dict including
        ``hypnogram``, all one-second ``epochs``, final ``bouts``, ``original``,
        ``step1``, ``step2``, ``metadata`` and the NREM merge audit. Epochs retain
        original/intermediate/final labels, features, reasons and validity.

    Notes
    -----
    Step 1 selects clear activated/HF or low-slow Wake; Step 2 merges short
    weak-quiet gaps while protecting slow-and-still cores. Step 3 selects
    sustained sigma-rich NREM, grows supported edges and bridges short gaps.
    Other Wake defaults to Quiet wake; other NREM defaults to NREM SWS, also
    when EEG is unusable (explicitly flagged in details). Unannotated EEG time
    remains Unscored. No source files are changed or cached. Boundaries have
    a few seconds of spectral-window uncertainty; these are spectral substates,
    not individual spindle detections or validated confidence probabilities.
    """
    original = normalize_hypnogram(hypno)
    if not isinstance(eeg, xr.DataArray) or eeg.ndim != 1 or "time" not in eeg.coords:
        raise TypeError(
            "eeg must be a one-dimensional DataArray with a time coordinate."
        )
    if (
        not np.isfinite(window_s)
        or window_s < 2
        or not np.isfinite(nrem_window_s)
        or nrem_window_s < 2
        or not np.isfinite(eye_likelihood_min)
        or not 0 <= eye_likelihood_min <= 1
        or (grid_origin_s is not None and not np.isfinite(grid_origin_s))
    ):
        raise ValueError(
            "Invalid spectral windows, grid origin or pupil quality cutoff."
        )
    c1, c2, c3 = (
        _config(step1_config, ActiveWakeConfig),
        _config(step2_config, WakeMergeConfig),
        _config(step3_config, NremSigmaConfig),
    )
    if eye is None:
        eye = pl.DataFrame(schema={"time": pl.Float64})
    if whisk is None:
        whisk = pl.DataFrame(schema={"time": pl.Float64, "whis": pl.Float64})
    context, meta, highpass = build_active_wake_context(
        eeg,
        eye,
        whisk,
        original,
        grid_origin_s=grid_origin_s,
        window_s=window_s,
        eye_likelihood_min=eye_likelihood_min,
    )
    s1 = select_active_wake(context, c1)
    s1["metadata"] = meta
    context, meta = build_wake_merge_context(s1["epochs"], eeg, meta)
    s2 = merge_wake_substates(context, c2)
    s2["metadata"] = meta
    meta = {**meta, "nrem_window_s": nrem_window_s}
    context, meta = build_nrem_sigma_context(s2["epochs"], eeg, meta)
    result = label_nrem_substates(context, c3)
    result["metadata"] = {
        **meta,
        "pipeline_algorithm": ALGORITHM_VERSION,
        "step1_config": asdict(c1),
        "step2_config": asdict(c2),
        "step3_config": asdict(c3),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "missing_eye": eye.is_empty(),
        "missing_whisk": whisk.is_empty(),
    }
    result["hypnogram"] = Hypnogram(
        result["bouts"]
        .select("start_s", "end_s", "label")
        .rename({"start_s": "start_time", "end_s": "end_time", "label": "state"})
    )
    if return_details:
        return dict(
            **result,
            original=original,
            step1=s1,
            step2=s2,
            context=context,
            highpass=highpass,
        )
    return result["hypnogram"]
