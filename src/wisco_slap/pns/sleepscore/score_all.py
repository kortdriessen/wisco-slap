"""Batch scoring entry point: score individual sessions or all subjects."""

from __future__ import annotations

import os
from typing import Any

import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS

from ._config import ScoreConfig
from ._data import create_session, load_scoring_times
from ._model import load_model, score_session


def autoscore_session(
    subject: str,
    exp: str,
    sync_block: int,
    model: dict[str, Any],
    store_chans: dict[str, list[int]] | None = None,
    overwrite: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score a single session and save results.

    Parameters
    ----------
    subject, exp, sync_block : identifiers.
    model : trained model dict.
    store_chans : EEG channel mapping.
    overwrite : whether to overwrite existing results.

    Returns
    -------
    bout_df, epoch_df : scoring results as polars DataFrames.
    """
    times = load_scoring_times(subject, exp, sync_block)
    if len(times[0]) > 1:
        raise NotImplementedError("Multiple scoring times per sync block not yet supported")
    t1 = times[0][0]
    t2 = times[1][0]

    session = create_session(subject, exp, sync_block, t1, t2, store_chans)
    bout_df, epoch_df = score_session(model, session)

    # Save results
    hypno_dir = os.path.join(
        DEFS.anmat_root, subject, exp,
        "scoring_data", f"sync_block-{sync_block}",
        "hypnograms", "model_labelled",
    )
    wis.util.check_dir(hypno_dir)

    epoch_path = os.path.join(hypno_dir, "epochs.parquet")
    bout_path = os.path.join(hypno_dir, "bouts.parquet")

    if not os.path.exists(epoch_path) or overwrite:
        epoch_df.write_parquet(epoch_path)

    if not os.path.exists(bout_path) or overwrite:
        bout_df.write_parquet(bout_path)

    return bout_df, epoch_df


def autoscore_all_subjects(
    model_path: str | None = None,
    model: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Score all subjects/experiments/sync_blocks.

    Parameters
    ----------
    model_path : path to saved model file (used if ``model`` is None).
    model : pre-loaded model dict; takes precedence over ``model_path``.
    overwrite : whether to overwrite existing results.
    """
    if model is None:
        if model_path is None:
            raise ValueError("Must provide either model or model_path")
        model = load_model(model_path)

    si = wis.meta.get.sync_info()
    for subject in si:
        for exp in si[subject]:
            for sb in si[subject][exp]["sync_blocks"]:
                try:
                    autoscore_session(subject, exp, sb, model, overwrite=overwrite)
                    print(f"  Scored {subject} {exp} sync_block-{sb}")
                except Exception as e:
                    print(f"  Error scoring {subject} {exp} sync_block-{sb}: {e}")
