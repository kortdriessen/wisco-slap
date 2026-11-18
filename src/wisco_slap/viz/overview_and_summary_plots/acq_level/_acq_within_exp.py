"""Whole-experiment context figure for one acquisition.

Puts one acquisition in the context of its whole experiment on a single
wall-clock axis: the TDT sync blocks, every acquisition of the experiment (the
requested one filled, the others outlined), the ephys traces with a spectrogram,
and the behavioural signals (pupil diameter, pupil motion, whisking), with the
sleep-scoring hypnogram shaded over every trace panel. It is the quickest way to
see *when* an acquisition happened relative to the animal's sleep/wake cycle and
whether the surrounding sync-block data (ephys / pupil / whisking / scoring) are
all present.

Every component is loaded on its own; whatever cannot be loaded is skipped and
listed in a report printed under the figure (nothing is fatal except the
acquisition being unknown to ``sync_info.yaml``). The report also says how each
y-range / colour scale was chosen and how much of the data it hides, so a panel
that is being flattened by artefact spikes is flagged rather than silently
misleading.
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import fluxplot as fp
import fluxplot.colors as flex
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.signal import spectrogram as _spectrogram

import wisco_slap as wis
from wisco_slap.viz.overview_and_summary_plots._spec import PlotSpec

#: Ephys stores drawn as trace panels, top to bottom.
DEFAULT_TRACE_STORES: tuple[str, ...] = ("EEG_", "loal")
#: Each ephys trace is decimated to this rate (Hz) for drawing.
TRACE_PLOT_HZ = 100.0
#: Per-trace y-limits come from these percentiles of the pooled decimated
#: samples (artefact spikes otherwise flatten the whole trace).
TRACE_CLIP_PCT: tuple[float, float] = (0.05, 99.95)
#: Pupil samples whose DLC likelihood is below this are blanked (diameter and motion).
MIN_PUP_LIKELIHOOD = 0.5
#: Behavioural panels and the percentiles that map to 0 and 1 in their robust
#: min-max normalisation (pooled over all sync blocks, then clipped). Same idea
#: as ``wis.util.snr.robust_minmax`` / ``min_max_dlc_dfs``.
DEFAULT_BEHAV_NORM_PCT: dict[str, tuple[float, float]] = {
    "pupil": (1, 99),
    "pupil motion": (1, 99),
    "whisking": (1, 99.9),
}
#: A spectrogram panel is drawn directly beneath this trace panel.
DEFAULT_SPEC_STORE = "EEG_"
#: Welch window (s) and hop (s) of the spectrogram: 0.25 Hz resolution, one
#: column per 2 s.
SPEC_WINDOW_S, SPEC_STEP_S = 4.0, 2.0
#: Frequency range (Hz) shown in the spectrogram.
SPEC_F_RANGE: tuple[float, float] = (0.5, 20.0)
#: Colour limits of the spectrogram from these percentiles of log10 power (all
#: blocks pooled).
SPEC_CLIP_PCT: tuple[float, float] = (1, 99)
SPEC_CMAP = "viridis"
#: Hypnogram shading colour per state; states not listed are not shaded.
STATE_COLORS = {"Wake": flex.green600, "NREM": flex.blue600, "REM": flex.magenta600}
STATE_COLORS.update({s: wis.defs.state_colors[s] for s in
                     ['Active wake', 'Quiet wake', 'NREM SWS', 'NREM sigma']})
HYPNO_ALPHA = 0.25
#: Lines / bars sit at zorder 2: the shading is drawn on top so a dense trace
#: cannot hide the state colour.
HYPNO_ZORDER = 3
#: Width (inches) of the figure; height follows from the number of panels.
FIG_WIDTH = 32.0
#: y-scale report flags a panel when the raw extent exceeds this multiple of the
#: shown range, or when more than this percentage of samples lie outside it.
YSCALE_OVERSHOOT_FLAG = 20.0
YSCALE_OUTSIDE_FLAG_PCT = 5.0

_BEHAV_STYLE = {
    "pupil": dict(color=flex.purple600, linewidth=0.8),
    "pupil motion": dict(color=flex.purple600, linewidth=0.6),
    "whisking": dict(color=flex.cyan600, linewidth=0.6),
}
_BEHAV_LABEL = {
    "pupil": "pupil diameter",
    "pupil motion": "pupil motion",
    "whisking": "whisking (masked frame diff)",
}
#: ``eye_metrics`` column feeding each pupil panel.
_EYE_COLUMNS = {"pupil": "diameter", "pupil motion": "motion"}


def acq_within_exp_path(subject: str, exp: str, loc: str, acq: str) -> Path:
    """Return where :func:`acq_within_exp_plot` saves its PNG for one acq.

    ``<annotation_root>/OVERVIEW_FIGURES/acq_level/acq_within_exp/
    <subject>--<exp>--<loc>--<acq>.png``
    """
    return Path(
        os.path.join(
            wis.defs.annotation_root,
            "OVERVIEW_FIGURES",
            "acq_level",
            "acq_within_exp",
            f"{subject}--{exp}--{loc}--{acq}.png",
        )
    )


def _to_datetime(value: str | datetime) -> datetime:
    """Parse a ``sync_info.yaml`` timestamp (string or already a datetime)."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _wall(t_seconds: np.ndarray, block_start: datetime) -> np.ndarray:
    """TDT-block-relative seconds -> wall-clock ``datetime64[us]`` array."""
    micros = (np.asarray(t_seconds, dtype=float) * 1e6).astype("timedelta64[us]")
    return np.datetime64(block_start) + micros


def _err(e: Exception) -> str:
    """One-line ``Type: message`` description of an exception for the report."""
    msg = str(e).splitlines()[0][:120] if str(e) else ""
    return f"{type(e).__name__}: {msg}"


def _robust_norm01(
    values: np.ndarray, qlo: float, qhi: float
) -> tuple[np.ndarray, float, float]:
    """Robust min-max normalisation of ``values`` to ``[0, 1]``.

    ``qlo``/``qhi`` percentiles of the finite values map to 0 and 1; the result
    is clipped to ``[0, 1]``. NaNs are preserved. Returns ``(normalised, lo,
    hi)``.

    Raises
    ------
    ValueError
        If the percentiles are not finite or not strictly increasing (all-NaN
        input, or a constant signal), so the caller can fall back to raw values.
    """
    v = np.asarray(values, dtype=float)
    lo, hi = (
        np.nanpercentile(v, [qlo, qhi]) if np.isfinite(v).any() else (np.nan, np.nan)
    )
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        raise ValueError(f"cannot normalise: percentiles {lo}, {hi}")
    return np.clip((v - lo) / (hi - lo), 0, 1), float(lo), float(hi)


def _yscale_line(name: str, values: np.ndarray, lo: float, hi: float, how: str) -> str:
    """Describe how the shown range ``(lo, hi)`` relates to the data.

    Reports the percentage of finite samples outside the shown range and the raw
    extent (max − min) as a multiple of the shown range, flagged with
    ``<-- CHECK`` when either exceeds :data:`YSCALE_OVERSHOOT_FLAG` /
    :data:`YSCALE_OUTSIDE_FLAG_PCT`.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or not np.isfinite([lo, hi]).all() or hi <= lo:
        return f"{name}: range could not be set ({how}); no finite data"
    outside = 100 * float(np.mean((v < lo) | (v > hi)))
    overshoot = float((v.max() - v.min()) / (hi - lo))
    flagged = overshoot > YSCALE_OVERSHOOT_FLAG or outside > YSCALE_OUTSIDE_FLAG_PCT
    flag = "  <-- CHECK" if flagged else ""
    return (
        f"{name}: shown [{lo:.3g}, {hi:.3g}] ({how}); {outside:.2f}% of samples "
        f"outside; raw extent is {overshoot:.1f}x the shown range{flag}"
    )


def acq_within_exp_plot(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    trace_stores: tuple[str, ...] = DEFAULT_TRACE_STORES,
    spec_store: str = DEFAULT_SPEC_STORE,
    behav_norm_pct: dict[str, tuple[float, float]] | None = None,
    hypno_version: str = "model",
    fig_width: float = FIG_WIDTH,
    dpi: int = 150,
    verbose: bool = True,
    overwrite: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one acquisition in the context of its whole experiment.

    Rows, top to bottom, all sharing one wall-clock x-axis that spans every
    sync block of the experiment:

    1. Sync blocks — one bar per TDT block, ``ephys_start → ephys_end``, green
       when its SYNC file is fine, red when corrupt.
    2. Acquisitions — one bar per acquisition in ``sync_info.yaml``, placed via
       :func:`wisco_slap.meta.get.acq_timing`. The requested ``loc/acq`` is
       filled orange (bold label); every other acquisition is an outline only.
    3. One panel per ephys store in ``trace_stores`` (decimated to
       :data:`TRACE_PLOT_HZ`, y-limits from :data:`TRACE_CLIP_PCT`), with a
       spectrogram panel directly beneath ``spec_store``.
    4. One panel per behavioural signal in ``behav_norm_pct`` (pupil diameter,
       pupil motion from ``eye_metrics``; whisking from ``whisk_df``), each
       robust-min-max normalised to ``[0, 1]`` with percentiles pooled over all
       sync blocks so blocks are comparable.

    The bout hypnogram of each sync block is shaded on top of every panel except
    the spectrogram (its colours would be tinted).

    Every component is loaded independently; one that is missing or unreadable
    is skipped and listed in the report printed after the figure, together with
    a y-scale line per panel (see :func:`_yscale_line`).

    The figure is saved to :func:`acq_within_exp_path` when no file exists there
    yet, or when ``overwrite=True``; otherwise the existing file is left alone.
    The figure is always built and returned.

    Parameters
    ----------
    subject, exp, loc, acq
        Acquisition identifiers, e.g. ``"alnair", "exp_1", "loc_D", "acq_1"``.
    trace_stores
        Ephys stores drawn as trace panels, top to bottom.
    spec_store
        Which of ``trace_stores`` gets a spectrogram beneath it.
    behav_norm_pct
        Behavioural panels to draw and their normalisation percentiles; default
        :data:`DEFAULT_BEHAV_NORM_PCT`. Keys must be a subset of
        ``{"pupil", "pupil motion", "whisking"}``.
    hypno_version
        Passed to :func:`wisco_slap.get.sb_bout_hypno` (``"model"`` or
        ``"validated"``).
    fig_width
        Figure width in inches; the height follows from the panel count.
    dpi
        Resolution of the saved PNG.
    verbose
        Print the y-scale report and the list of skipped components.
    overwrite
        Re-save even if the PNG already exists.

    Returns
    -------
    (fig, axes)
        The figure and its 1-D array of axes (``2 + len(trace_stores) + 1 +
        len(behav_norm_pct)`` rows).

    Raises
    ------
    KeyError
        If ``loc--acq`` is not listed for ``subject/exp`` in ``sync_info.yaml``.
    ValueError
        If ``spec_store`` is not one of ``trace_stores`` or ``behav_norm_pct``
        names an unknown panel.
    """
    fp.style.use_lighttable()
    wis.util.checks.validate_ids(exp, loc, acq)
    if behav_norm_pct is None:
        behav_norm_pct = dict(DEFAULT_BEHAV_NORM_PCT)
    if spec_store not in trace_stores:
        raise ValueError(
            f"spec_store {spec_store!r} must be one of trace_stores {trace_stores}"
        )
    unknown = set(behav_norm_pct) - set(_BEHAV_STYLE)
    if unknown:
        raise ValueError(
            f"unknown behavioural panel(s) {sorted(unknown)}; "
            f"known: {sorted(_BEHAV_STYLE)}"
        )

    si = wis.meta.get.sync_info()[subject][exp]
    sync_blocks, acquisitions = si["sync_blocks"], si["acquisitions"]
    this_acq = f"{loc}--{acq}"
    if this_acq not in acquisitions:
        raise KeyError(
            f"{this_acq} is not in sync_info for {subject}/{exp}; "
            f"known: {sorted(acquisitions)}"
        )

    # panel order, top to bottom: blocks, acquisitions, each trace (the
    # spectrogram right under its trace), then the behavioural panels
    panel_names = ["blocks", "acquisitions"]
    for store in trace_stores:
        panel_names.append(store)
        if store == spec_store:
            panel_names.append(f"{store} spectrogram")
    panel_names += list(behav_norm_pct)
    height_ratios = [1 if p in ("blocks", "acquisitions") else 2.5 for p in panel_names]
    f, ax = plt.subplots(
        len(panel_names),
        1,
        figsize=(fig_width, 3 + 1.75 * (len(panel_names) - 2)),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = dict(zip(panel_names, ax, strict=True))
    ax_blocks, ax_acqs = axes["blocks"], axes["acquisitions"]
    ax_traces = {store: axes[store] for store in trace_stores}
    ax_spec = axes[f"{spec_store} spectrogram"]
    ax_behav = {name: axes[name] for name in behav_norm_pct}
    ax_bottom = ax[-1]  # carries the x-axis label / tick format / limits (shared)
    # not the spectrogram: shading would tint its colormap
    shaded_axes = [ax_acqs, *ax_traces.values(), *ax_behav.values()]

    missing: list[
        str
    ] = []  # components that could not be loaded/drawn (with the reason)
    yscale_report: list[str] = []  # how each y-range / colour scale was chosen

    # Bar rows (blocks, acquisitions): the single-line label is centred INSIDE
    # the bar so it can never collide with the axes border however short the
    # row is; text sits above the hypnogram shading.
    bar_y, bar_h = 0.0, 0.8
    bar_text_kw = dict(
        ha="center", va="center", fontsize=11, zorder=HYPNO_ZORDER + 1, clip_on=True
    )

    # --- blocks: one bar per sync block, ephys_start -> ephys_end (wall clock) ---
    for sb_num, info in sorted(sync_blocks.items()):
        start, end = _to_datetime(info["ephys_start"]), _to_datetime(info["ephys_end"])
        corrupt = info.get("corrupt", "Unknown")
        color = (
            "tab:red"
            if corrupt is True
            else "tab:green"
            if corrupt is False
            else "tab:gray"
        )
        ax_blocks.barh(
            bar_y,
            end - start,
            left=start,
            height=bar_h,
            color=color,
            alpha=0.6,
            edgecolor="k",
        )
        ax_blocks.text(
            start + (end - start) / 2,
            bar_y,
            f"block-{sb_num}  (corrupt={corrupt})",
            **bar_text_kw,
        )
    ax_blocks.set_ylim(-0.5, 0.5)
    ax_blocks.set_yticks([])
    ax_blocks.set_ylabel("sync blocks")
    ax_blocks.set_title(
        f"{subject} / {exp} / {loc} / {acq} — sync blocks (ephys_start → ephys_end)"
    )

    # --- acquisitions: one bar each on its sync block's wall-clock axis; the
    # chosen acq filled, the others outlined ---
    for acq_id, entry in sorted(
        acquisitions.items(), key=lambda kv: str(kv[1].get("dat_start", ""))
    ):
        loc_, acq_ = acq_id.split("--")
        try:
            block_start = _to_datetime(
                sync_blocks[int(entry["sync_block"])]["ephys_start"]
            )
            timing = wis.meta.get_acq_timing(subject, exp, loc_, acq_)
        except (
            Exception
        ) as e:  # no sync block / no scopex zarr / unresolved ephys offset ...
            missing.append(f"{acq_id}: acquisition timing — {_err(e)}")
            continue
        start = block_start + timedelta(seconds=timing.start_time)
        end = block_start + timedelta(seconds=timing.end_time)
        is_this = acq_id == this_acq
        ax_acqs.barh(
            bar_y,
            end - start,
            left=start,
            height=bar_h,
            edgecolor="k",
            linewidth=1.2,
            color=flex.orange400 if is_this else "none",
            alpha=0.8 if is_this else 1.0,
        )
        ax_acqs.text(
            start + (end - start) / 2,
            bar_y,
            f"{acq_id}  ({timing.duration / 60:.1f} min)",
            fontweight="bold" if is_this else "normal",
            **bar_text_kw,
        )
    ax_acqs.set_ylim(-0.5, 0.5)
    ax_acqs.set_yticks([])
    ax_acqs.set_ylabel("acquisitions")

    # --- per sync block: ephys traces (+ spectrogram tiles), behavioural
    # signals (collected, drawn after pooling), hypnogram ---
    states_seen: set[str] = set()
    trace_samples: dict[str, list[np.ndarray]] = {store: [] for store in trace_stores}
    spec_tiles: list[
        tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = []  # (wall-clock columns, freqs, log10 power)
    behav_segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in behav_norm_pct
    }
    for sb_num, info in sorted(sync_blocks.items()):
        block_start = _to_datetime(info["ephys_start"])
        tag = f"sync_block-{sb_num}"

        # Each component is loaded and drawn on its own; a missing or malformed
        # file skips just that component.
        for store, a in ax_traces.items():
            try:
                trace = wis.get.ephys(subject, exp, sb_num, stores=(store,))[store]
                fs = float(trace.attrs.get("fs", TRACE_PLOT_HZ))
                step = max(1, int(round(fs / TRACE_PLOT_HZ)))
                a.plot(
                    _wall(trace.time.values[::step], block_start),
                    trace.values[::step],
                    color=flex.base850,
                    linewidth=0.4,
                )
                trace_samples[store].append(
                    np.asarray(trace.values[::step], dtype=float)
                )
            except Exception as e:
                missing.append(f"{tag}: ephys store {store!r} — {_err(e)}")
                continue
            if store == spec_store:
                try:
                    nperseg = int(round(SPEC_WINDOW_S * fs))
                    freqs, spg_t, spg = _spectrogram(
                        np.asarray(trace.values, dtype=np.float64),
                        fs,
                        window="hann",
                        nperseg=nperseg,
                        noverlap=nperseg - int(round(SPEC_STEP_S * fs)),
                        detrend="constant",
                    )
                    keep = (freqs >= SPEC_F_RANGE[0]) & (freqs <= SPEC_F_RANGE[1])
                    # spg_t are window centres in seconds from the first sample
                    # -> TDT-block seconds -> wall clock
                    t_cols = _wall(float(trace.time.values[0]) + spg_t, block_start)
                    spec_tiles.append((t_cols, freqs[keep], np.log10(spg[keep])))
                except Exception as e:
                    missing.append(f"{tag}: {store} spectrogram — {_err(e)}")

        eye_panels = [name for name in behav_norm_pct if name in _EYE_COLUMNS]
        if eye_panels:
            try:
                eye = wis.get.eye_metrics(subject, exp, sb_num)
                t_eye = _wall(eye["time"].to_numpy(), block_start)
                bad = eye["pup_likelihood"].to_numpy() < MIN_PUP_LIKELIHOOD
                for name in eye_panels:
                    v = eye[_EYE_COLUMNS[name]].to_numpy().astype(float)
                    v[bad] = np.nan
                    behav_segments[name].append((t_eye, v))
            except Exception as e:
                missing.append(f"{tag}: eye_metrics — {_err(e)}")

        if "whisking" in behav_segments:
            try:
                whisk = wis.get.whisk_df(subject, exp, sb_num)
                behav_segments["whisking"].append((
                    _wall(whisk["time"].to_numpy(), block_start),
                    whisk["whis"].to_numpy().astype(float),
                ))
            except Exception as e:
                missing.append(f"{tag}: whisk_df — {_err(e)}")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore"
                )  # the loader warns when absent; we record it ourselves
                hypno = wis.get.sb_bout_hypno(
                    subject, exp, sb_num, version=hypno_version
                )
            if hypno is None:
                raise FileNotFoundError(f"no {hypno_version!r} bout hypnogram")
            for row in hypno.df.iter_rows(named=True):
                color = STATE_COLORS.get(row["state"])
                if color is None:
                    continue
                states_seen.add(row["state"])
                for a in shaded_axes:
                    a.axvspan(
                        block_start + timedelta(seconds=row["start_time"]),
                        block_start + timedelta(seconds=row["end_time"]),
                        color=color,
                        alpha=HYPNO_ALPHA,
                        linewidth=0,
                        zorder=HYPNO_ZORDER,
                    )
        except Exception as e:
            missing.append(f"{tag}: hypnogram — {_err(e)}")

    # ephys traces: y-limits from percentiles of the decimated samples of all
    # blocks pooled
    for store, a in ax_traces.items():
        how = f"{TRACE_CLIP_PCT[0]}-{TRACE_CLIP_PCT[1]} pct"
        if trace_samples[store]:
            pooled = np.concatenate(trace_samples[store])
            lo, hi = np.nanpercentile(pooled, TRACE_CLIP_PCT)
            if np.isfinite([lo, hi]).all() and hi > lo:
                a.set_ylim(lo, hi)
            yscale_report.append(_yscale_line(store, pooled, lo, hi, how))
        a.set_ylabel(f"{store} (clipped to {how})")

    # behavioural signals: robust min-max to [0, 1], percentiles pooled over
    # blocks, fixed axis
    for name, (qlo, qhi) in behav_norm_pct.items():
        a, segs = ax_behav[name], behav_segments[name]
        if segs:
            pooled = np.concatenate([v for _, v in segs])
            try:
                _, lo, hi = _robust_norm01(pooled, qlo, qhi)
            except ValueError as e:
                missing.append(f"{name}: normalisation failed ({e}); drawn raw")
                for t, v in segs:
                    a.plot(t, v, **_BEHAV_STYLE[name])
            else:
                for t, v in segs:
                    a.plot(t, np.clip((v - lo) / (hi - lo), 0, 1), **_BEHAV_STYLE[name])
                a.set_ylim(-0.05, 1.05)
                yscale_report.append(
                    _yscale_line(
                        name, pooled, lo, hi, f"0-1 = {qlo:g}-{qhi:g} pct, clipped"
                    )
                )
        a.set_ylabel(f"{_BEHAV_LABEL[name]}\n(norm. {qlo:g}-{qhi:g} pct)")

    # spectrogram: one colour scale shared by every sync block; the colorbar
    # sits outside the axes so the panel keeps the same width as the others
    if spec_tiles:
        pooled_lp = np.concatenate([lp.ravel() for _, _, lp in spec_tiles])
        vmin, vmax = np.nanpercentile(pooled_lp, SPEC_CLIP_PCT)
        mesh = None
        for x_cols, f_kept, lp in spec_tiles:
            mesh = ax_spec.pcolormesh(
                x_cols,
                f_kept,
                lp,
                cmap=SPEC_CMAP,
                vmin=vmin,
                vmax=vmax,
                shading="nearest",
                rasterized=True,
            )
        cbar = f.colorbar(mesh, cax=ax_spec.inset_axes([1.005, 0.0, 0.008, 1.0]))
        cbar.set_label("log10 power", fontsize=9)
        yscale_report.append(
            _yscale_line(
                f"{spec_store} spectrogram colour",
                pooled_lp,
                vmin,
                vmax,
                f"{SPEC_CLIP_PCT[0]}-{SPEC_CLIP_PCT[1]} pct",
            )
        )
    ax_spec.set_ylim(*SPEC_F_RANGE)
    ax_spec.set_ylabel(
        f"{spec_store} spectrogram (Hz)\n"
        f"{SPEC_WINDOW_S:g} s window, {SPEC_STEP_S:g} s step"
    )

    # --- shared x axis, decorated on the bottom panel ---
    ax_bottom.set_xlabel("Wall-clock time")
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    starts = [_to_datetime(v["ephys_start"]) for v in sync_blocks.values()]
    ends = [_to_datetime(v["ephys_end"]) for v in sync_blocks.values()]
    pad = (max(ends) - min(starts)) * 0.02
    ax_bottom.set_xlim(min(starts) - pad, max(ends) + pad)
    handles = [
        Patch(color=STATE_COLORS[s], alpha=HYPNO_ALPHA + 0.3, label=s)
        for s in STATE_COLORS
        if s in states_seen
    ]
    if handles:
        ax_traces[trace_stores[0]].legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.005, 1),
            frameon=False,
            title="hypnogram",
        )

    if verbose:
        print(
            f"y-scale report (CHECK = >{YSCALE_OVERSHOOT_FLAG:g}x raw overshoot or "
            f">{YSCALE_OUTSIDE_FLAG_PCT:g}% of samples outside the shown range):\n  "
            + "\n  ".join(yscale_report)
        )
        if missing:
            print(
                f"{subject} / {exp} — skipped (not plotted):\n  " + "\n  ".join(missing)
            )
        else:
            print(f"{subject} / {exp} — every component was found and plotted.")

    out_path = acq_within_exp_path(subject, exp, loc, acq)
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f.savefig(out_path, dpi=dpi)
        print(f"Saved acq-within-exp overview to {out_path}")
    else:
        print(f"Acq-within-exp overview already exists (overwrite=False): {out_path}")
    return f, ax


PLOT_SPEC = PlotSpec(
    name="acq_within_exp",
    plot=acq_within_exp_plot,
    path=acq_within_exp_path,
)
