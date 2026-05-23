"""Data loading, label handling, and epoch grid construction."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import polars as pl
import yaml

import wisco_slap as wis
import wisco_slap.defs as DEFS

from ._config import STATE_NAMES, STATE_TO_IDX


# ---------------------------------------------------------------------------
# Session data construction
# ---------------------------------------------------------------------------


def create_session(
    subject: str,
    exp: str,
    sb: int,
    t1: float | None = None,
    t2: float | None = None,
    store_chans: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    """Build a session dict containing all modality data for one sync block.

    Parameters
    ----------
    subject, exp, sb : identifiers for the recording.
    t1, t2 : optional time bounds (seconds) to trim the session.
    store_chans : EEG channel mapping; defaults to ``{"EEG_": [1]}``.

    Returns
    -------
    dict with keys: ``session_id``, ``eeg``, ``pupil``.
    """
    if store_chans is None:
        store_chans = {"EEG_": [1]}
    if t1 is None:
        t1 = 0
    if t2 is None:
        t2 = np.inf

    session: dict[str, Any] = {}
    session["session_id"] = f"{subject}_{exp}_sb-{sb}"

    # EEG
    ephys = wis.peri.ephys.load_single_ephys_block(
        subject, exp, ["EEG_"], sync_block=sb, store_chans=store_chans,
    )
    eeg = ephys["EEG_"].sel(time=slice(t1, t2))
    session["eeg"] = {
        "signal": eeg.values,
        "timestamps": eeg.time.values,
        "fs": float(eeg.fs),
    }

    # Whisking
    whis = wis.peri.vid.load_whisking_df(subject, exp, sb)

    # Pupil / eyelid
    eye = wis.peri.vid.load_eye_metric_df(subject, exp, sb)

    vid_timestamps = whis["time"].to_numpy()
    session["pupil"] = {
        "diameter": eye["diameter"].to_numpy(),
        "motion": eye["motion"].to_numpy(),
        "eyelid": eye["lid"].to_numpy(),
        "eyelid_norm": eye["lid_norm"].to_numpy(),
        "pup_likelihood": (
            eye["pup_likelihood"].to_numpy()
            if "pup_likelihood" in eye.columns
            else np.ones(len(eye), dtype=float)
        ),
        "lid_likelihood": (
            eye["lid_likelihood"].to_numpy()
            if "lid_likelihood" in eye.columns
            else np.ones(len(eye), dtype=float)
        ),
        "whisking": whis["whis"].to_numpy(),
        "timestamps": vid_timestamps,
        "fs": 10.0,
    }
    return session


# ---------------------------------------------------------------------------
# Scoring time windows (optional trim bounds per sync block)
# ---------------------------------------------------------------------------


def load_scoring_times(
    subject: str, exp: str, sb: int,
) -> tuple[list[float | None], list[float | None]]:
    """Return (starts, ends) time bounds from sb_scoring_times.yaml, if defined."""
    path = os.path.join(DEFS.anmat_root, "sb_scoring_times.yaml")
    if not os.path.exists(path):
        return ([None], [None])
    with open(path) as f:
        t = yaml.safe_load(f)
    sb_name = f"sync_block-{sb}"
    try:
        entry = t[subject][exp][sb_name]
        return (entry["starts"], entry["ends"])
    except (KeyError, TypeError):
        return ([None], [None])


# ---------------------------------------------------------------------------
# Label loading and validation
# ---------------------------------------------------------------------------

# Mapping from various label spellings to canonical state names
_LABEL_MAPPING: dict[str, str] = {
    "NREM": "NREM",
    "NREM_SWS": "NREM",
    "NREM_LIGHT": "NREM",
    "NREM_SPINDLE": "NREM",
    "NR": "NREM",
    "SWS": "NREM",
    "REM": "REM",
    "REM_PHASIC": "REM",
    "R": "REM",
    "WAKE": "Wake",
    "WAKE_QUIET": "Wake",
    "WAKE_BRIEF_AROUSAL": "Wake",
    "ARTIFACT": "Wake",
    "W": "Wake",
}

_OBSOLETE_LABELS: set[str] = {"IS", "INTERMEDIATE_STATE", "INTERMEDIATE"}


def validate_and_standardize_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize label names and reject obsolete or unknown labels.

    Parameters
    ----------
    df : DataFrame with columns ``start_s``, ``end_s``, ``label``.

    Returns
    -------
    Cleaned DataFrame with canonical labels, sorted by time.
    """
    required = {"start_s", "end_s", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Label DataFrame must have columns {required}, got {set(df.columns)}")

    df = df.sort("start_s", "end_s")

    raw_source_col = "raw_label" if "raw_label" in df.columns else "label"
    raw_labels = [
        "" if label is None else str(label).strip()
        for label in df.get_column(raw_source_col).to_list()
    ]
    normalized_labels = [label.upper() for label in raw_labels]

    unknown_labels = sorted({label for label in normalized_labels if label not in _LABEL_MAPPING})
    if unknown_labels:
        obsolete_labels = [label for label in unknown_labels if label in _OBSOLETE_LABELS]
        unsupported_labels = [
            label for label in unknown_labels if label not in _OBSOLETE_LABELS
        ]
        parts: list[str] = []
        if obsolete_labels:
            parts.append(
                f"obsolete labels are not supported anymore: {obsolete_labels}"
            )
        if unsupported_labels:
            parts.append(f"unknown labels: {unsupported_labels}")
        raise ValueError(
            "Unrecognized sleep labels in training data ("
            + "; ".join(parts)
            + "). Supported aliases map to canonical states "
            f"{STATE_NAMES}."
        )

    mapped_labels = [_LABEL_MAPPING[label] for label in normalized_labels]
    return df.with_columns(
        pl.Series("raw_label", raw_labels),
        pl.Series("label", mapped_labels),
    )


def prepare_training_labels(
    labels_df: pl.DataFrame | None,
    *,
    exclude_artifact: bool = False,
) -> pl.DataFrame | None:
    """Return labels for classifier training, optionally dropping Artifact intervals."""
    if labels_df is None or labels_df.is_empty() or not exclude_artifact:
        return labels_df

    if "raw_label" not in labels_df.columns:
        raise ValueError(
            "exclude_artifact=True requires labels with a 'raw_label' column. "
            "Reload labels with load_labels_csv(...) or validate_and_standardize_labels(...)."
        )

    raw_label = (
        pl.col("raw_label")
        .cast(pl.String)
        .str.strip_chars()
        .str.to_uppercase()
        .fill_null("")
    )
    return labels_df.filter(raw_label != "ARTIFACT")


def load_labels_csv(path: str) -> pl.DataFrame:
    """Load a label CSV file and validate it.

    Parameters
    ----------
    path : path to CSV with columns ``start_s``, ``end_s``, ``label``.

    Returns
    -------
    Validated polars DataFrame.
    """
    df = pl.read_csv(path)
    return validate_and_standardize_labels(df)


# ---------------------------------------------------------------------------
# Epoch grid and label assignment
# ---------------------------------------------------------------------------


def build_epoch_grid(
    sessions: list[dict],
    epoch_len_s: float,
) -> dict[str, np.ndarray]:
    """Create epoch edge arrays per session from the common time range of EEG and pupil.

    Parameters
    ----------
    sessions : list of session dicts (must have ``eeg`` and ``pupil`` keys).
    epoch_len_s : epoch length in seconds.

    Returns
    -------
    dict mapping session_id to 1-D array of epoch edges (length n_epochs + 1).
    """
    edges_by_session: dict[str, np.ndarray] = {}
    for sess in sessions:
        sid = sess["session_id"]
        starts, ends = [], []
        for key in ("eeg", "pupil"):
            ts = sess[key]["timestamps"]
            if len(ts) == 0:
                raise ValueError(f"Session {sid}: {key} has empty timestamps")
            starts.append(float(ts[0]))
            ends.append(float(ts[-1]))
        start_s = max(starts)
        end_s = min(ends)
        if end_s <= start_s + epoch_len_s:
            raise ValueError(f"Session {sid}: insufficient overlap across modalities")
        n_epochs = int(np.floor((end_s - start_s) / epoch_len_s))
        edges_by_session[sid] = start_s + np.arange(n_epochs + 1) * epoch_len_s
    return edges_by_session


def intervals_to_epoch_labels(
    edges: np.ndarray,
    labels_df: pl.DataFrame | None,
    min_overlap_frac: float = 0.0,
) -> np.ndarray:
    """Map label intervals to fixed epochs via maximum overlap.

    Unlabeled epochs get -1.

    Parameters
    ----------
    edges : epoch edge array of length n_epochs + 1.
    labels_df : polars DataFrame with ``start_s``, ``end_s``, ``label`` columns (already validated).
    min_overlap_frac : minimum fraction of epoch covered by a label to assign it.

    Returns
    -------
    1-D int array of length n_epochs with state indices or -1.
    """
    n = len(edges) - 1
    y = np.full(n, -1, dtype=int)
    if labels_df is None or labels_df.is_empty():
        return y

    starts = labels_df["start_s"].to_numpy()
    ends = labels_df["end_s"].to_numpy()
    labs = np.array([STATE_TO_IDX[s] for s in labels_df["label"].to_list()])

    epoch_len = float(edges[1] - edges[0]) if n > 0 else 0.0
    min_ov = min_overlap_frac * epoch_len

    i = 0
    for e in range(n):
        a, b = float(edges[e]), float(edges[e + 1])
        while i < len(starts) and ends[i] <= a:
            i += 1
        j = i
        best_lab, best_ov = -1, 0.0
        while j < len(starts) and starts[j] < b:
            ov = max(0.0, min(b, ends[j]) - max(a, starts[j]))
            if ov >= min_ov and ov > best_ov:
                best_ov = ov
                best_lab = int(labs[j])
            j += 1
        y[e] = best_lab
    return y
