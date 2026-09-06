# ====================================================================================
# MAIN FUNCTIONS FOR WORKING WITH DATA ANNOTATIONS (see SeeQt package for annotation software)
# ====================================================================================

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

import electro_py as epy
import wisco_slap.defs as DEFS


def load_annotation_csv(path: str, reformat: bool = True) -> pd.DataFrame:
    """Load an annotation CSV file into a pandas dataframe."""
    df = pd.read_csv(path)
    if reformat:
        if not all(col in df.columns for col in ["start_s", "end_s", "label"]):
            if all(
                col in df.columns
                for col in ["start_time", "end_time", "state", "duration"]
            ):
                # fomatting is already done
                return df
            else:
                raise ValueError(
                    f"Format of csv not recognized, try without reformat: {path}"
                )
        else:
            # reformat the csv
            df = df.rename(columns={"label": "state"})
            df = df.rename(columns={"start_s": "start_time", "end_s": "end_time"})
            df["duration"] = df["end_time"] - df["start_time"]
            df = df.sort_values(by="start_time")
            df = df.reset_index(drop=True)
    return df


def load_auto_hypno(
    subject,
    exp,
    sync_block,
    filter_unclear: bool = None,
    filter_on="smooth",
    rename=True,
):
    hypno_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/hypnograms/model_labelled"
    hypno_name = "raw_epochs.csv"
    epoch_df = pd.read_csv(f"{hypno_dir}/{hypno_name}")
    if filter_unclear is not None:
        if filter_on == "smooth":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["NREM_smooth", "REM_smooth", "Wake_smooth"]].max(axis=1)
                < filter_unclear,
                "unclear",
            )
        elif filter_on == "raw":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["P_NREM", "P_REM", "P_Wake"]].max(axis=1) < filter_unclear,
                "unclear",
            )
        else:
            raise ValueError(f"Invalid filter_on: {filter_on}")
    epoch_df = epy.hypno.utils.merge_consecutive_labels(epoch_df)
    if rename:
        epoch_df = epoch_df.rename(columns={"label": "state"})
        epoch_df = epoch_df.rename(
            columns={"start_s": "start_time", "end_s": "end_time"}
        )
        epoch_df["duration"] = epoch_df["end_time"] - epoch_df["start_time"]
        epoch_df = epoch_df.sort_values(by="start_time")
        epoch_df = epoch_df.reset_index(drop=True)
    return epoch_df


def load_manual_hypno(
    subject,
    exp,
    sync_block,
    name="manual",
    filter_unclear: bool | None = None,
    filter_on="smooth",
    rename=True,
):
    hypno_dir = f"{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/hypnograms"
    hypno_name = f"{name}.csv"
    epoch_df = pd.read_csv(f"{hypno_dir}/{hypno_name}")
    if filter_unclear is not None:
        if filter_on == "smooth":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["NREM_smooth", "REM_smooth", "Wake_smooth"]].max(axis=1)
                < filter_unclear,
                "unclear",
            )
        elif filter_on == "raw":
            epoch_df["label"] = epoch_df["label"].mask(
                epoch_df[["P_NREM", "P_REM", "P_Wake"]].max(axis=1) < filter_unclear,
                "unclear",
            )
        else:
            raise ValueError(f"Invalid filter_on: {filter_on}")
    epoch_df = epy.hypno.utils.merge_consecutive_labels(epoch_df)
    if rename:
        epoch_df = epoch_df.rename(columns={"label": "state"})
        epoch_df = epoch_df.rename(
            columns={"start_s": "start_time", "end_s": "end_time"}
        )
        epoch_df["duration"] = epoch_df["end_time"] - epoch_df["start_time"]
        epoch_df = epoch_df.sort_values(by="start_time")
        epoch_df = epoch_df.reset_index(drop=True)
    return epoch_df


# ====================================================================================
# PER-BOUT PERIPHERAL CONTEXT (bandpower, pupil/whisking, SWA released, bout structure)
# ====================================================================================


def _per_bout_stats(
    hypno: epy.hypno.Hypnogram,
    times: np.ndarray,
    values: np.ndarray,
    name: str,
    stats: tuple[str, ...] = ("mean", "std"),
) -> epy.hypno.Hypnogram:
    """Per-bout summary stats of a 1-D series sampled at `times` (NaN-aware).

    Samples are assigned to bouts by half-open [start_time, end_time) windows,
    exactly like ``epy.hypno.utils.add_bandpower_per_bout``. Adds one column per
    requested stat, named ``{name}_{stat}``; ``stats`` may contain "mean", "std",
    "cv" (std/mean) and "n" (number of finite samples). NaN where a bout holds
    no finite sample (std/cv need >= 2).
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]

    df = hypno.df
    lo = np.searchsorted(times, df["start_time"].to_numpy(), side="left")
    hi = np.searchsorted(times, df["end_time"].to_numpy(), side="left")

    finite = np.isfinite(values)
    v = np.where(finite, values, 0.0)
    ccnt = np.concatenate([[0], np.cumsum(finite)])
    csum = np.concatenate([[0.0], np.cumsum(v)])
    csq = np.concatenate([[0.0], np.cumsum(v * v)])

    n = ccnt[hi] - ccnt[lo]
    n_safe = np.where(n > 0, n, 1)
    mean = np.where(n > 0, (csum[hi] - csum[lo]) / n_safe, np.nan)
    var = np.where(n > 1, (csq[hi] - csq[lo]) / n_safe - mean**2, np.nan)
    std = np.sqrt(np.clip(var, 0.0, None))

    cols: dict[str, np.ndarray] = {}
    if "mean" in stats:
        cols[f"{name}_mean"] = mean
    if "std" in stats:
        cols[f"{name}_std"] = std
    if "cv" in stats:
        cols[f"{name}_cv"] = np.where(
            mean != 0, std / np.where(mean != 0, mean, 1), np.nan
        )
    if "n" in stats:
        cols[f"{name}_n"] = n.astype(float)
    return epy.hypno.Hypnogram(
        df.with_columns([pl.Series(k, c) for k, c in cols.items()])
    )


def _rolling_nanmean(x: np.ndarray, win: int, min_frac: float = 0.25) -> np.ndarray:
    """Centred rolling NaN-mean; NaN where fewer than ``min_frac*win`` finite bins."""
    win = max(int(win), 1)
    finite = np.isfinite(x)
    v = np.where(finite, x, 0.0)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    ccnt = np.concatenate([[0], np.cumsum(finite)])
    i = np.arange(len(x))
    lo = np.clip(i - win // 2, 0, len(x))
    hi = np.clip(i - win // 2 + win, 0, len(x))
    n = ccnt[hi] - ccnt[lo]
    out = np.where(
        n >= max(1, min_frac * win),
        (csum[hi] - csum[lo]) / np.where(n > 0, n, 1),
        np.nan,
    )
    return out


def add_peripheral_context_to_hypno(
    hypno: epy.hypno.Hypnogram,
    eeg: xr.DataArray,
    eye: pl.DataFrame,
    whisk: pl.DataFrame,
    state_ref: epy.hypno.Hypnogram | None = None,
    ratios: tuple[tuple[str, str], ...] = (
        ("delta1", "delta2"),
        ("delta", "theta"),
        ("delta", "sigma"),
        ("delta", "beta"),
        ("delta", "high_gamma"),
        ("theta", "high_gamma"),
    ),
    norm_state: str | None = "NREM",
    validity_mask: xr.DataArray | None = None,
    eye_likelihood_min: float | None = None,
    swa_band: str = "delta",
    swa_state: str = "NREM",
    swa_smooth_sec: float = 600.0,
    bp_window_sec: float = 4.0,
    bp_step_sec: float = 1.0,
    bp_kwargs: dict | None = None,
) -> epy.hypno.Hypnogram:
    """Attach every available peripheral / contextual measure to each bout.

    All inputs must share the TDT-block-relative float time axis (they do when
    they come from ``wis.get.fetch_data_pipe('peripherals', ...)``).

    Parameters
    ----------
    hypno
        The (typically diced) float-time hypnogram to annotate.
    eeg
        1-D EEG DataArray with a ``time`` coord and ``fs`` attr.
    eye, whisk
        ``eye_metrics`` and ``whisk_df`` frames (``time`` column).
    state_ref
        The *undiced* hypnogram used for everything that depends on the true
        bout structure (SWA-released integrals, parent-bout timing, neighbour
        states). Defaults to ``hypno`` itself; pass the original hypnogram when
        ``hypno`` is diced, otherwise dropped remainders / short bouts are lost.
    ratios
        ``(numerator, denominator)`` band pairs -> ``{num}_over_{den}`` columns
        (ratio of the per-bout mean powers).
    norm_state
        Reference state for per-recording normalisation of the bandpowers: each
        band's per-bout mean is divided by the median of that band over all STFT
        bins falling in ``norm_state`` bouts of ``state_ref`` (the *undiced*
        hypnogram, so dicing does not change the reference). This removes the
        subject-specific gain and spectral tilt of the implant, so ``{band}_norm``
        is comparable across subjects. NREM is the default because it is the
        state least contaminated by movement artefacts; note that by
        construction ``{band}_norm`` is ~1 in NREM, so what the normalised
        columns carry is within-recording dynamics relative to the NREM median
        (homeostatic decline, other states relative to NREM), not the absolute
        NREM level. ``None`` disables it.
    validity_mask
        Optional 1-D boolean DataArray on the same time axis (e.g. ``dn.is_valid``
        from a scopex trace), marking which imaging samples are usable. Adds
        ``valid_frac`` = fraction of the bout's samples that are True, and
        ``n_samples`` = how many samples fell in the bout at all (0 -> the bout
        lies outside the acquisition, and ``valid_frac`` is NaN there).
    eye_likelihood_min
        If given, pupil frames with ``pup_likelihood`` below this are masked
        before the pupil stats, and ``pupil_valid_frac`` is added.
    swa_band, swa_state, swa_smooth_sec
        Band / state / smoothing window defining "slow-wave activity" for the
        SWA-released columns.
    bp_window_sec, bp_step_sec
        STFT window / hop for ``bandpower_stft``. The 4 s default gives 0.25 Hz
        bins, which the 0.5-2 / 2-3.5 Hz delta sub-bands need (the 1 s default
        of ``bandpower_stft`` puts a single 1 Hz bin in each and double-counts
        the boundary bins).
    bp_kwargs
        Extra keyword arguments for ``epy.sigpro.spectro.bandpower_stft``.

    Added columns
    -------------
    - ``{band}`` mean bandpower per bout (all bands of ``bandpower_stft``),
      ``bp_total`` (sum over bands), ``{band}_rel`` (fraction of ``bp_total``),
      ``{num}_over_{den}`` ratios; with ``norm_state``: ``{band}_norm`` and
      ``{num}_over_{den}_norm`` (ratio of the normalised bands).
    - ``total_elapsed``, ``{state}_elapsed``, ``continuous_time_in_state``
      (see ``epy.hypno.utils.add_elapsed_columns``) computed on ``hypno``
      itself -- on a diced / acquisition-trimmed hypnogram these count only
      the chunks present (time before the first chunk and dropped remainders
      are invisible), so they are "elapsed among scored chunks".
    - ``total_elapsed_exp``, ``{state}_elapsed_exp``,
      ``continuous_time_in_state_exp``: the same quantities measured at each
      bout's start on ``state_ref``, the full continuous hypnogram of the
      experiment -- independent of chunking or trimming. ``NREM_elapsed_exp``
      is the true NREM time accrued since the recording began, and
      ``continuous_time_in_state_exp`` the true depth into the current
      uninterrupted episode (equal to ``time_into_parent_bout`` unless the
      parent bout continues an earlier contiguous same-state bout). NaN for
      bouts that fall outside ``state_ref``.
    - ``valid_frac`` / ``n_samples`` (only with ``validity_mask``).
    - Pupil / eyelid / whisking: ``pupil_diam_{mean,std,cv}``,
      ``pupil_motion_{mean,std}``, ``lid_{mean,std}`` (``lid_norm``),
      ``whisk_{mean,std}``.
    - SWA released (from ``state_ref``): ``swa_cum`` = integral of
      ``swa_band`` power over all earlier ``swa_state`` time (power x seconds),
      ``swa_cum_frac`` = that as a fraction of the whole recording's total;
      ``total_cum_delta`` / ``total_cum_delta_frac`` = the same integral over
      ALL scored time (any state) - SWA that occurred in Wake/REM counts too;
      ``swa_peak_time`` / ``time_since_swa_peak`` (peak of the
      ``swa_smooth_sec``-smoothed ``swa_state``-only ``swa_band`` trajectory),
      ``{swa_band}_frac_of_swa_peak`` = this bout's power / that peak.
    - Bout-structure context (from ``state_ref``): ``parent_bout_dur``,
      ``time_into_parent_bout``, ``time_to_parent_bout_end``,
      ``frac_through_parent_bout``, ``prev_state``, ``next_state``,
      ``since_last_{state}_end``, ``until_next_{state}_start``.
    """
    if hypno.time_kind is not epy.hypno.TimeKind.FLOAT:
        raise ValueError(
            "hypno must be float-time (seconds); call .as_float() first."
        )
    ref = hypno if state_ref is None else state_ref
    if ref.time_kind is not epy.hypno.TimeKind.FLOAT:
        raise ValueError("state_ref must be float-time (seconds).")

    # ---- 1. bandpower, relative power, ratios, elapsed time -----------------
    bp = epy.sigpro.spectro.bandpower_stft(
        signal=np.asarray(eeg.values, dtype=float),
        fs=float(eeg.fs),
        time_offset=float(eeg["time"].values[0]),
        window_sec=bp_window_sec,
        step_sec=bp_step_sec,
        **(bp_kwargs or {}),
    )
    bands = list(bp.data_vars)
    hypno = epy.hypno.utils.add_bandpower_per_bout(bp, hypno, band=bands)

    total = pl.sum_horizontal([pl.col(b) for b in bands])
    hypno = epy.hypno.Hypnogram(
        hypno.df.with_columns(total.alias("bp_total"))
        .with_columns(
            [(pl.col(b) / pl.col("bp_total")).alias(f"{b}_rel") for b in bands]
        )
        .with_columns(
            [
                (pl.col(num) / pl.col(den)).alias(f"{num}_over_{den}")
                for num, den in ratios
                if num in bands and den in bands
            ]
        )
    )
    if norm_state is not None:
        if norm_state not in ref.states:
            raise ValueError(
                f"norm_state {norm_state!r} not in state_ref states {ref.states}"
            )
        t_bp_all = np.asarray(bp["time"].values, dtype=float)
        in_norm = ref.mask_times_by_state(t_bp_all, [norm_state])
        norm_cols = []
        for b in bands:
            ref_med = float(
                np.nanmedian(np.asarray(bp[b].values, dtype=float)[in_norm])
            )
            norm_cols.append((pl.col(b) / ref_med).alias(f"{b}_norm"))
        hypno = epy.hypno.Hypnogram(
            hypno.df.with_columns(norm_cols).with_columns(
                [
                    (pl.col(f"{num}_norm") / pl.col(f"{den}_norm")).alias(
                        f"{num}_over_{den}_norm"
                    )
                    for num, den in ratios
                    if num in bands and den in bands
                ]
            )
        )
    hypno = epy.hypno.utils.add_elapsed_columns(hypno)

    # ---- 1b. imaging validity per bout ------------------------------------
    if validity_mask is not None:
        vm = validity_mask
        if not (
            isinstance(vm, xr.DataArray) and vm.ndim == 1 and "time" in vm.coords
        ):
            raise TypeError(
                "validity_mask must be a 1-D DataArray with a 'time' coord"
            )
        hypno = _per_bout_stats(
            hypno,
            vm["time"].values,
            np.asarray(vm.values, dtype=float),
            "valid",
            ("mean", "n"),
        )
        hypno = epy.hypno.Hypnogram(
            hypno.df.rename({"valid_mean": "valid_frac", "valid_n": "n_samples"})
        )

    # ---- 2. pupil / eyelid / whisking ---------------------------------------
    t_eye = eye["time"].to_numpy()
    diam = eye["diameter"].to_numpy().astype(float)
    motion = eye["motion"].to_numpy().astype(float)
    lid = eye["lid_norm"].to_numpy().astype(float)
    if eye_likelihood_min is not None:
        ok = eye["pup_likelihood"].to_numpy() >= eye_likelihood_min
        diam = np.where(ok, diam, np.nan)
        motion = np.where(ok, motion, np.nan)
        hypno = _per_bout_stats(
            hypno, t_eye, ok.astype(float), "pupil_valid", ("mean",)
        )
        hypno = epy.hypno.Hypnogram(
            hypno.df.rename({"pupil_valid_mean": "pupil_valid_frac"})
        )
    hypno = _per_bout_stats(hypno, t_eye, diam, "pupil_diam", ("mean", "std", "cv"))
    hypno = _per_bout_stats(hypno, t_eye, motion, "pupil_motion", ("mean", "std"))
    hypno = _per_bout_stats(hypno, t_eye, lid, "lid", ("mean", "std"))
    hypno = _per_bout_stats(
        hypno,
        whisk["time"].to_numpy(),
        whisk["whis"].to_numpy(),
        "whisk",
        ("mean", "std"),
    )

    # ---- 3. SWA released -----------------------------------------------------
    t_bp = np.asarray(bp["time"].values, dtype=float)
    swa = np.asarray(bp[swa_band].values, dtype=float)
    step = float(bp.attrs.get("step_sec", np.median(np.diff(t_bp))))
    in_state = ref.mask_times_by_state(t_bp, [swa_state])
    scored = ref.covers_time(t_bp)
    starts = hypno.df["start_time"].to_numpy()

    def _cum_energy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        energy = np.where(mask & np.isfinite(swa), swa * step, 0.0)
        cum = np.concatenate([[0.0], np.cumsum(energy)])
        at_start = cum[np.searchsorted(t_bp, starts, side="left")]
        total = cum[-1]
        frac = at_start / total if total > 0 else np.full_like(at_start, np.nan)
        return at_start, frac

    swa_cum, swa_cum_frac = _cum_energy(in_state)  # swa_state only
    total_cum, total_cum_frac = _cum_energy(scored)  # any scored state

    # Smoothed swa_state-only trajectory; the peak time is the mean time of the
    # bins that fed the winning window (not the window's nominal centre, which
    # sits outside the data at the recording edges).
    win = round(swa_smooth_sec / step)
    swa_smooth = _rolling_nanmean(np.where(in_state, swa, np.nan), win)
    t_smooth = _rolling_nanmean(np.where(in_state, t_bp, np.nan), win)
    if np.any(np.isfinite(swa_smooth)):
        i_peak = int(np.nanargmax(swa_smooth))
        swa_peak_time, swa_peak = float(t_smooth[i_peak]), float(swa_smooth[i_peak])
    else:
        swa_peak_time, swa_peak = np.nan, np.nan
    hypno = epy.hypno.Hypnogram(
        hypno.df.with_columns(
            pl.Series("swa_cum", swa_cum),
            pl.Series("swa_cum_frac", swa_cum_frac),
            pl.Series("total_cum_delta", total_cum),
            pl.Series("total_cum_delta_frac", total_cum_frac),
            pl.lit(swa_peak_time).alias("swa_peak_time"),
            (pl.col("start_time") - swa_peak_time).alias("time_since_swa_peak"),
            (pl.col(swa_band) / swa_peak).alias(f"{swa_band}_frac_of_swa_peak"),
        )
    )

    # ---- 4. parent-bout + neighbour-state context (from state_ref) ---------
    rdf = ref.df
    r_start = rdf["start_time"].to_numpy()
    r_end = rdf["end_time"].to_numpy()
    r_state = rdf["state"].to_numpy()
    ends = hypno.df["end_time"].to_numpy()
    mid = (starts + ends) / 2
    idx = ref.bout_index(mid)
    has = idx >= 0
    safe = np.where(has, idx, 0)
    p_start = np.where(has, r_start[safe], np.nan)
    p_end = np.where(has, r_end[safe], np.nan)
    p_dur = p_end - p_start

    def _neighbour(shift: int) -> list[str | None]:
        j = idx + shift
        ok = has & (j >= 0) & (j < len(r_state))
        jj = np.where(ok, j, 0)
        return [str(r_state[k]) if o else None for k, o in zip(jj, ok, strict=True)]

    # Elapsed-time columns measured on the full continuous hypnogram: the
    # parent bout's value at its start plus how far into the parent we are.
    # (The plain versions above only see the chunks present in ``hypno``.)
    into_parent = starts - p_start
    ref_el = epy.hypno.utils.add_elapsed_columns(ref).df
    p_state = np.where(has, r_state[safe], None)

    def _ref_col(name: str) -> np.ndarray:
        return np.where(has, ref_el[name].to_numpy()[safe], np.nan)

    exp_cols = [
        pl.Series("total_elapsed_exp", _ref_col("total_elapsed") + into_parent),
        pl.Series(
            "continuous_time_in_state_exp",
            _ref_col("continuous_time_in_state") + into_parent,
        ),
    ] + [
        pl.Series(
            f"{s}_elapsed_exp",
            _ref_col(f"{s}_elapsed") + np.where(p_state == s, into_parent, 0.0),
        )
        for s in ref.states
    ]

    new_cols = exp_cols + [
        pl.Series("parent_bout_dur", p_dur),
        pl.Series("time_into_parent_bout", starts - p_start),
        pl.Series("time_to_parent_bout_end", p_end - ends),
        pl.Series("frac_through_parent_bout", (mid - p_start) / p_dur),
        pl.Series("prev_state", _neighbour(-1), dtype=pl.Utf8),
        pl.Series("next_state", _neighbour(+1), dtype=pl.Utf8),
    ]
    for s in ref.states:
        s_end = np.sort(r_end[r_state == s])
        s_start = np.sort(r_start[r_state == s])
        # last bout of s that ended <= start
        j = np.searchsorted(s_end, starts, side="right") - 1
        since = np.where(j >= 0, starts - s_end[np.clip(j, 0, None)], np.nan)
        # first bout of s starting >= end
        k = np.searchsorted(s_start, ends, side="left")
        until = np.where(
            k < len(s_start),
            s_start[np.clip(k, None, max(len(s_start) - 1, 0))] - ends,
            np.nan,
        )
        new_cols += [
            pl.Series(f"since_last_{s}_end", since),
            pl.Series(f"until_next_{s}_start", until),
        ]

    return epy.hypno.Hypnogram(hypno.df.with_columns(new_cols))
