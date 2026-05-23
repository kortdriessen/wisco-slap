import os
import warnings

import polars as pl
from electro_py.hypno.hypno import Hypnogram

from wisco_slap.defs import anmat_root
from wisco_slap.meta import get_acq_timing
from wisco_slap.meta.get import sync_info as _get_sync_info


def hypno_csv(path: str, reformat: bool = True) -> pl.DataFrame:
    """Load an annotation CSV file into a polars DataFrame."""
    df = pl.read_csv(path)
    if reformat:
        if not all(col in df.columns for col in ["start_s", "end_s", "label"]):
            if all(
                col in df.columns
                for col in ["start_time", "end_time", "state", "duration"]
            ):
                # formatting is already done
                return df
            else:
                raise ValueError(
                    f"Format of csv not recognized, try without reformat: {path}"
                )
        else:
            # reformat the csv
            df = df.rename({
                "label": "state",
                "start_s": "start_time",
                "end_s": "end_time",
            })
            df = df.with_columns(
                (pl.col("end_time") - pl.col("start_time")).alias("duration")
            )
            df = df.sort("start_time")
    return df


def bout_hypno(
    subject: str, exp: str, loc: str, acq: str, reformat: bool = True
) -> Hypnogram | None:
    """Load the bout-level hypnogram for the sync block containing this acquisition.

    Resolves the sync block from ``sync_info.yaml`` and prefers the validated
    hypnogram (``bout_hypno_VALIDATED.csv``) over the model-scored fallback
    (``model_scored/bout_hypno.csv``).

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    loc : str
        Location name.
    acq : str
        Acquisition name.
    reformat : bool
        Passed through to :func:`hypno_csv`.

    Returns
    -------
    pl.DataFrame | None
        The bout hypnogram, or ``None`` if neither file exists for this sync
        block (in which case a warning is emitted).
    """
    si = _get_sync_info()
    sb = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
    hypno_dir = os.path.join(
        anmat_root, subject, exp, "sync_block_data", f"sync_block-{sb}", "hypnograms"
    )
    validated = os.path.join(hypno_dir, "bout_hypno_VALIDATED.csv")
    model_scored = os.path.join(hypno_dir, "model_scored", "bout_hypno.csv")
    if os.path.exists(validated):
        return Hypnogram(hypno_csv(validated, reformat=reformat))
    if os.path.exists(model_scored):
        return Hypnogram(hypno_csv(model_scored, reformat=reformat))
    warnings.warn(
        f"No bout hypnogram found for {subject}/{exp} sync_block-{sb} "
        f"(checked {validated} and {model_scored})."
    )
    return None


def acq_sleep_coverage(
    subject: str, exp: str, loc: str, acq: str
) -> tuple[Hypnogram, float, float, float, float]:
    """Compute the fractional sleep-state occupancy during one acquisition.

    Loads the sync block's bout hypnogram, clips it to the acquisition's
    [start, stop] window, and reports the fraction of that window spent in
    each scored state.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    loc : str
        Location name.
    acq : str
        Acquisition name.

    Returns
    -------
    hypno_acq : Hypnogram
        The bout hypnogram clipped to the acquisition window.
    duration : float
        Acquisition duration in seconds (from :func:`get_acq_timing`).
    fraction_wake : float
        Fraction of ``duration`` scored as Wake (0.0 if state absent).
    fraction_nrem : float
        Fraction of ``duration`` scored as NREM (0.0 if state absent).
    fraction_rem : float
        Fraction of ``duration`` scored as REM (0.0 if state absent).
    """
    start, stop, duration, _sb = get_acq_timing(subject, exp, loc, acq)
    hypno = bout_hypno(subject, exp, loc, acq)
    if hypno is None:
        raise FileNotFoundError(
            f"No bout hypnogram available for {subject}/{exp} {loc}--{acq}; "
            "cannot compute sleep coverage."
        )
    hypno_acq = hypno.trim(start, stop)
    fa = hypno_acq.fractional_occupancy()
    fractions = dict(zip(fa["state"].to_list(), fa["fraction"].to_list(), strict=True))
    fraction_wake = fractions.get("Wake", 0.0)
    fraction_nrem = fractions.get("NREM", 0.0)
    fraction_rem = fractions.get("REM", 0.0)
    return hypno_acq, duration, fraction_wake, fraction_nrem, fraction_rem
