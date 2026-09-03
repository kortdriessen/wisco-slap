import os
import warnings

import polars as pl

from electro_py.hypno.hypno import Hypnogram
from wisco_slap.defs import anmat_root
from wisco_slap.meta import get_acq_timing
from wisco_slap.meta.get import sync_info as _get_sync_info


def hypno_csv(path: str, reformat: bool = True) -> pl.DataFrame | Hypnogram:
    """Load an annotation CSV file into a polars DataFrame."""
    df = pl.read_csv(path)
    if reformat:
        if not all(col in df.columns for col in ["start_s", "end_s", "label"]):
            if all(
                col in df.columns
                for col in ["start_time", "end_time", "state", "duration"]
            ):
                # formatting is already done
                return Hypnogram(df)
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
            return Hypnogram(df)
    return df


def sb_bout_hypno(
    subject: str,
    exp: str,
    sb: int,
    reformat: bool = True,
    version: str = "model",
) -> Hypnogram | pl.DataFrame | None:
    """Load the bout-level hypnogram of one sync block.

    Times in the hypnogram are TDT-block-relative seconds, i.e. the same
    axis as :func:`wisco_slap.get.ephys`, ``frame_times`` and the
    eye/whisk dataframes for that block.

    Parameters
    ----------
    subject : str
        Subject name.
    exp : str
        Experiment name.
    sb : int
        Sync block number.
    reformat : bool
        Whether to reformat the loaded DataFrame into a ``Hypnogram``.
    version : str
        ``"validated"`` loads ``bout_hypno_VALIDATED.csv``; ``"model"``
        (default) loads ``model_scored/bout_hypno.csv``.

    Returns
    -------
    pl.DataFrame | Hypnogram | None
        The bout hypnogram, or ``None`` (with a warning) if the requested
        version does not exist for this sync block.
    """
    hypno_dir = os.path.join(
        anmat_root, subject, exp, "sync_block_data", f"sync_block-{sb}", "hypnograms"
    )
    validated_path = os.path.join(hypno_dir, "bout_hypno_VALIDATED.csv")
    if os.path.exists(validated_path) and version == "validated":
        return hypno_csv(validated_path, reformat=reformat)

    model_scored_path = os.path.join(hypno_dir, "model_scored", "bout_hypno.csv")
    if os.path.exists(model_scored_path) and version == "model":
        return hypno_csv(model_scored_path, reformat=reformat)
    warnings.warn(
        f"No bout hypnogram found for {subject}/{exp} sync_block-{sb} "
        f"(checked {validated_path} and {model_scored_path}).",
        stacklevel=2,
    )
    return None


def bout_hypno(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    reformat: bool = True,
    version: str = "model",
) -> Hypnogram | pl.DataFrame | None:
    """Load the bout-level hypnogram for the sync block containing this acquisition.

    Resolves the sync block from ``sync_info.yaml`` and delegates to
    :func:`sb_bout_hypno`.

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
        Whether to reformat the loaded DataFrame.
    version : str
        The version of the hypnogram to load. If "validated",
        the validated version is preferred, otherwise the model-scored version is used.

    Returns
    -------
    pl.DataFrame | Hypnogram | None
        The bout hypnogram, or ``None`` if neither file exists for this sync
        block (in which case a warning is emitted).
        Returns pl.DataFrame if ``reformat=False``, otherwise a Hypnogram.
    """
    si = _get_sync_info()
    sb = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
    return sb_bout_hypno(subject, exp, sb, reformat=reformat, version=version)


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


# ==================================================================================
# Fleet-wide sleep-coverage table
# ==================================================================================
_TABLE_STATES = ("NREM", "REM", "Wake")


def acq_sleep_table(version: str = "model") -> pl.DataFrame:
    """One row per acquisition in ``acquisition_master.yaml`` with its duration and
    the seconds / fraction spent in each sleep state, sorted by duration.

    For every acquisition the sync block's bout hypnogram is clipped to the
    acquisition's ``[start_time, end_time]`` window (TDT-block-relative, from
    :func:`wisco_slap.meta.get.acq_timing`) and the clipped bout durations are
    summed per state. Fractions are ``state_s / duration_s`` where ``duration_s``
    is the wall-clock span of the acquisition (``end_time - start_time``), so
    ``nrem + rem + wake + unscored == 1`` for every row.

    Acquisitions whose scopex zarr, ephys offset, or hypnogram are missing are
    still listed (so the table is a complete inventory) with the reason in
    ``note`` and null state columns.

    Parameters
    ----------
    version : str
        Hypnogram version passed to :func:`bout_hypno` (``"model"`` or
        ``"validated"``).

    Returns
    -------
    pl.DataFrame
        Columns: ``subject, exp, loc, acq, sync_block, duration_s, active_s,
        nrem_s, nrem_frac, rem_s, rem_frac, wake_s, wake_frac, unscored_s,
        unscored_frac, note``. ``active_s`` is the sum of per-trial durations
        (excludes inter-trial / inter-epoch gaps); ``duration_s`` is what the
        state seconds are measured against.
    """
    from wisco_slap.meta.get import acq_master

    rows: list[dict] = []
    for subject, exps in acq_master().items():
        for exp, acq_ids in exps.items():
            for acq_id in acq_ids:
                loc, acq = acq_id.split("--")
                row: dict = {
                    "subject": subject,
                    "exp": exp,
                    "loc": loc,
                    "acq": acq,
                    "sync_block": None,
                    "duration_s": None,
                    "active_s": None,
                    **{f"{s.lower()}_s": None for s in _TABLE_STATES},
                    **{f"{s.lower()}_frac": None for s in _TABLE_STATES},
                    "unscored_s": None,
                    "unscored_frac": None,
                    "note": "",
                }
                notes: list[str] = []
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        timing = get_acq_timing(subject, exp, loc, acq)
                    if any("multi-epoch and spans" in str(w.message) for w in caught):
                        notes.append("multi-epoch, spans >1 sync block")
                except FileNotFoundError:
                    row["note"] = "no scopex zarr (run wis.pns.scopex_mon.exp_data)"
                    rows.append(row)
                    continue
                except KeyError as e:
                    row["note"] = f"not in sync_info.yaml (KeyError: {e})"
                    rows.append(row)
                    continue
                except ValueError as e:
                    row["note"] = f"no timing: {e}"
                    rows.append(row)
                    continue

                span = timing.end_time - timing.start_time
                row["sync_block"] = timing.sync_block
                row["duration_s"] = span
                row["active_s"] = timing.duration

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hypno = bout_hypno(subject, exp, loc, acq, version=version)
                if hypno is None:
                    notes.append("no hypnogram")
                    row["note"] = "; ".join(notes)
                    rows.append(row)
                    continue

                clipped = hypno.trim(timing.start_time, timing.end_time).df
                per_state = dict(
                    clipped.group_by("state").agg(pl.col("duration").sum()).iter_rows()
                )
                extra = set(per_state) - set(_TABLE_STATES)
                if extra:
                    notes.append(f"unexpected states ignored: {sorted(extra)}")
                scored = 0.0
                for s in _TABLE_STATES:
                    sec = float(per_state.get(s, 0.0))
                    scored += sec
                    row[f"{s.lower()}_s"] = sec
                    row[f"{s.lower()}_frac"] = sec / span if span > 0 else None
                unscored = max(span - scored, 0.0)
                row["unscored_s"] = unscored
                row["unscored_frac"] = unscored / span if span > 0 else None
                if unscored / span > 0.05:
                    notes.append(f"{unscored / span:.0%} of acq unscored")
                row["note"] = "; ".join(notes)
                rows.append(row)

    schema = {
        "subject": pl.Utf8,
        "exp": pl.Utf8,
        "loc": pl.Utf8,
        "acq": pl.Utf8,
        "sync_block": pl.Int64,
        "duration_s": pl.Float64,
        "active_s": pl.Float64,
        **{f"{s.lower()}_s": pl.Float64 for s in _TABLE_STATES},
        **{f"{s.lower()}_frac": pl.Float64 for s in _TABLE_STATES},
        "unscored_s": pl.Float64,
        "unscored_frac": pl.Float64,
        "note": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema)
    return df.sort("duration_s", descending=True, nulls_last=True)


def _fmt_state(sec: float | None, frac: float | None) -> str:
    if sec is None or frac is None:
        return "—"
    return f"{sec:.0f} s ({frac:.2f})"


def acq_sleep_table_markdown(df: pl.DataFrame, version: str = "model") -> str:
    """Render the output of :func:`acq_sleep_table` as a GitHub-flavoured markdown
    table (one line per acquisition, sorted as given)."""
    import datetime as _dt

    n = df.height
    n_scored = df.filter(pl.col("nrem_s").is_not_null()).height
    total_h = (df["duration_s"].fill_null(0.0).sum()) / 3600
    header = [
        "# Acquisition durations and sleep-state coverage",
        "",
        f"Generated {_dt.datetime.now():%Y-%m-%d %H:%M} by "
        f"`wis.get.write_acq_sleep_table()` · hypnogram version: `{version}` · "
        f"{n} acquisitions ({n_scored} with a hypnogram) · "
        f"{total_h:.1f} h total recording.",
        "",
        "Each state cell is `seconds (fraction of the acquisition)`. Fractions are "
        "relative to **duration** — the wall-clock span of the acquisition on the "
        "TDT axis — so NREM + REM + Wake + unscored = 1.00 on every row. "
        "**active** is the summed per-trial imaging time (no inter-trial / "
        "inter-epoch gaps). Machine-readable copy: `acq_sleep_table.csv` beside "
        "this file.",
        "",
        "| # | subject | exp | loc--acq | sb | duration | active | NREM | REM | "
        "Wake | unscored | note |",
        "|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    lines = []
    for i, r in enumerate(df.iter_rows(named=True), start=1):
        dur = "—" if r["duration_s"] is None else f"{r['duration_s']:.0f} s"
        act = "—" if r["active_s"] is None else f"{r['active_s']:.0f} s"
        sb = "—" if r["sync_block"] is None else str(r["sync_block"])
        lines.append(
            f"| {i} | {r['subject']} | {r['exp']} | {r['loc']}--{r['acq']} | {sb} | "
            f"{dur} | {act} | "
            f"{_fmt_state(r['nrem_s'], r['nrem_frac'])} | "
            f"{_fmt_state(r['rem_s'], r['rem_frac'])} | "
            f"{_fmt_state(r['wake_s'], r['wake_frac'])} | "
            f"{_fmt_state(r['unscored_s'], r['unscored_frac'])} | "
            f"{r['note']} |"
        )
    return "\n".join(header + lines) + "\n"


def write_acq_sleep_table(
    out_dir: str | None = None, version: str = "model"
) -> pl.DataFrame:
    """Build :func:`acq_sleep_table` and write it as ``acq_sleep_table.csv`` and
    ``acq_sleep_table.md`` (default location: the ``analysis_materials`` root,
    next to ``ExSum_audits.csv``). Returns the DataFrame."""
    out_dir = anmat_root if out_dir is None else out_dir
    df = acq_sleep_table(version=version)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "acq_sleep_table.csv")
    md_path = os.path.join(out_dir, "acq_sleep_table.md")
    df.write_csv(csv_path)
    with open(md_path, "w") as f:
        f.write(acq_sleep_table_markdown(df, version=version))
    print(f"Wrote {csv_path}\nWrote {md_path}")
    return df
