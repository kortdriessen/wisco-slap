from __future__ import annotations

import os
from pathlib import Path

import fluxplot as fp
import fluxplot.colors as flex
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xarray as xr

import electro_py as epy
import wisco_slap as wis


def _non_nan_fraction(data: xr.DataArray) -> float:
    """Return the fraction of array values that are not NaN."""
    values = np.asarray(data.data)
    if values.size == 0:
        return float("nan")
    return float(np.count_nonzero(~np.isnan(values)) / values.size)


def _shade_nan_gaps(ax: plt.Axes, data: xr.DataArray) -> None:
    """Shade full-height intervals where any trace has a NaN value."""
    values = np.asarray(data.data)
    times = np.asarray(data.time.values, dtype=float)
    if values.size == 0 or times.size == 0:
        return

    time_axis = data.get_axis_num("time")
    non_time_axes = tuple(i for i in range(values.ndim) if i != time_axis)
    nan_by_time = np.isnan(values)
    if non_time_axes:
        nan_by_time = np.any(nan_by_time, axis=non_time_axes)

    if times.size == 1:
        edges = np.array([times[0] - 0.5, times[0] + 0.5])
    else:
        edges = np.empty(times.size + 1)
        edges[1:-1] = (times[:-1] + times[1:]) / 2
        edges[0] = times[0] - (times[1] - times[0]) / 2
        edges[-1] = times[-1] + (times[-1] - times[-2]) / 2

    changes = np.diff(np.pad(nan_by_time.astype(np.int8), 1))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    for start, stop in zip(starts, stops, strict=True):
        ax.axvspan(
            edges[start],
            edges[stop],
            color=flex.magenta300,
            linewidth=0,
            zorder=2,
        )


# Height (inches) allotted to each panel row of an overview figure. The original
# single-soma layout was 8 rows at 14 inches tall.
_ROW_HEIGHT_IN = 14 / 8
_FIG_WIDTH_IN = 25
# Rolling window (samples) for the noise-floor estimate used to normalise the
# denoised glutamate traces into a matched-filter-like SNR image.
_SNR_WINDOW = 300 * 200
_PERIPHERAL_Y_LABELS = ["EEG (mV)", "Load Cell", "Pup. Dia.", "Pup. Mot.", "Whisk"]


def _resolve_out_path(
    out_dir: str | Path, default_subdir: str, filename: str
) -> tuple[Path, Path]:
    """Resolve the output directory and PNG path for an overview figure.

    ``out_dir == "default"`` maps to ``<cwd>/<default_subdir>``. The directory
    is not created here; that happens only when saving.
    """
    if out_dir == "default":
        out_dir = Path.cwd() / default_subdir
    else:
        out_dir = Path(out_dir)
    return out_dir, out_dir / filename


def _syn_snr_image(dn: xr.DataArray) -> xr.DataArray:
    """Divide denoised glutamate traces by their rolling noise floor.

    Returns ``dn`` unchanged when it has no synapses (rolling statistics are
    undefined on an empty array).
    """
    if dn.sizes["syn_id"] == 0:
        return dn
    dn_sigma = wis.util.snr.rolling_std(dn, window=_SNR_WINDOW)
    return dn / dn_sigma


def _plot_peripheral_panels(
    axes: np.ndarray,
    eeg: xr.DataArray,
    loal: xr.DataArray,
    eye,
    whisk,
    hypno_full,
) -> None:
    """Draw EEG, load cell, pupil diameter/motion and whisking onto five axes.

    Each axis is also shaded by sleep state using ``hypno_full``.
    """
    loal_abs = np.abs(loal.values)
    loal_upper = np.percentile(loal_abs, 99.5)
    loal_lower = np.percentile(loal_abs, 0.1)
    eye, whisk = wis.util.snr.min_max_dlc_dfs(eye, whisk)

    axes[0].plot(eeg.time.values, eeg.data, color=flex.base850, linewidth=0.5)
    axes[0].set_ylim(
        np.percentile(eeg.data, 0.2),
        np.percentile(eeg.data, 99.8),
    )
    axes[1].plot(loal.time.values, loal.data, color=flex.base850, linewidth=0.5)
    axes[1].set_ylim(loal_lower, loal_upper)
    axes[2].plot(
        eye["time"].to_numpy(), eye["diameter_norm"].to_numpy(), color=flex.purple600
    )
    axes[3].plot(
        eye["time"].to_numpy(),
        eye["motion_norm"].to_numpy(),
        color=flex.purple600,
        linewidth=0.3,
    )
    axes[4].plot(
        whisk["time"].to_numpy(),
        whisk["whis_norm"].to_numpy(),
        color=flex.orange600,
        linewidth=0.5,
    )
    for axis, label in zip(axes, _PERIPHERAL_Y_LABELS, strict=True):
        axis.set_ylabel(label)
        epy.plot.hypno.shade_hypno_for_me(hypno_full, ax=axis, alpha=0.2)


def _plot_soma_panel(ax: plt.Axes, soma_ca: xr.DataArray, spikes) -> None:
    """Draw a somatic dF/F trace with inferred spike times ticked along the top."""
    ax.plot(
        soma_ca.time.values,
        soma_ca.data,
        color=flex.magenta700,
        linewidth=0.5,
    )
    spike_times = spikes["time"].to_numpy()
    ax.scatter(
        spike_times,
        np.full(spike_times.shape, 0.98),
        transform=ax.get_xaxis_transform(),
        color=flex.red500,
        s=8,
        linewidths=0,
        zorder=3,
    )
    ax.set_ylabel("Soma dF/F")


def _plot_syn_heatmap_panel(ax: plt.Axes, mfsnr: xr.DataArray, dmd: int) -> None:
    """Draw the synaptic SNR heatmap for one DMD, with NaN-gap shading and a title.

    The title reports the dendrite types present, the DMD depth, the synapse
    count and the fraction of non-NaN samples. When the DMD has no synapses
    the image is skipped and only a summary title is written.
    """
    da = mfsnr.sx.dmd(dmd)
    n_syns = da.sizes["syn_id"]
    if n_syns > 0:
        t = da.time.values
        ax.imshow(
            da.data,
            aspect="auto",
            extent=(t[0], t[-1], n_syns - 0.5, -0.5),
            cmap=flex.maps.emerald,
            vmin=1,
            vmax=3.5,
        )
        _shade_nan_gaps(ax, da)
    valid_fraction = _non_nan_fraction(da)
    valid_text = f"{valid_fraction:.1%}" if np.isfinite(valid_fraction) else "n/a"

    if n_syns == 0:
        ax.set_yticks([])
        title = f"dmd-{dmd} | n_syns=0 | valid={valid_text}"
    else:
        strty = "".join(f"{dt} |" for dt in np.unique(da["dend_type"].values))
        depth = da["dmd-depth"].values[0]
        title = (
            f"dmd-{dmd} | {strty} depth={depth:.1f}um | n_syns={n_syns}"
            f" | valid={valid_text}"
        )
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Syn. ID")


def _set_shared_xlim(ax0: plt.Axes, x_axes: list[np.ndarray]) -> None:
    """Set the x-limits to the union of every stream's time range.

    With ``sharex=True`` setting the first axis propagates to all others.
    """
    xmin = min(float(np.nanmin(x)) for x in x_axes)
    xmax = max(float(np.nanmax(x)) for x in x_axes)
    ax0.set_xlim(xmin, xmax)


def _somas_with_ca_traces(subject: str, exp: str, loc: str, acq: str) -> list[str]:
    """Return the soma IDs that have a calcium ROI trace in this acquisition.

    These are the ROI labels recorded in ``dmd_info.yaml`` (populated from the
    ExperimentSummary ROI list), which is exactly the set of somas with a
    saved ``ROI_Fraw`` trace. Synapse-label-only IDs such as ``somaUK1`` never
    appear there. Order is dmd-1 somas then dmd-2 somas, in ROI order. A soma
    ID listed under both DMDs is returned once (the dmd-1 entry, matching how
    the ``"soma"`` data pipe resolves it).
    """
    somas_by_dmd = wis.scope.anat.get_somas_by_dmd(subject, exp, loc, acq)
    soma_ids: list[str] = []
    for dmd in ("dmd-1", "dmd-2"):
        for soma_id in somas_by_dmd.get(dmd, []):
            if soma_id in soma_ids:
                print(
                    f"{subject} {exp} {loc} {acq}: {soma_id!r} is listed under more "
                    "than one DMD; using the first occurrence only."
                )
                continue
            soma_ids.append(soma_id)
    return soma_ids


def all_soma_overview(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    out_dir: str | Path = "default",
    dpi: int = 300,
    save: bool = True,
    overwrite: bool = True,
) -> tuple[plt.Figure, np.ndarray] | Path:
    """Build and save the MAIN_OVERVIEW figure for every soma in an acquisition.

    After the five peripheral rows (EEG, load cell, pupil diameter, pupil
    motion, whisking) there is one block of three rows per soma: the soma's
    dF/F trace (with inferred spikes) followed by the dmd-1 and dmd-2
    glutamate-SNR heatmaps of the dendritic synapses attributed to that soma.

    "Every soma" means every soma with a calcium ROI trace, i.e. the ROI labels
    recorded in ``dmd_info.yaml`` (dmd-1 somas first, then dmd-2). IDs that only
    exist as synapse labels (``somaUK1`` etc.) are not included.

    Parameters
    ----------
    subject, exp, loc, acq
        Acquisition identifiers, e.g. ``"bellatrix", "exp_1", "loc_Z", "acq_1"``.
    out_dir
        Directory to write the PNG into. Created if it does not exist. The
        default is ``<cwd>/all_soma_overview``.
    dpi
        Resolution of the saved PNG.
    save
        Whether to save the generated figure. When ``False``, the figure is
        still built and returned.
    overwrite
        Whether to overwrite an existing file at the output path. If ``False`` and
        the file exists, nothing is built and the existing path is returned.

    Returns
    -------
    (fig, axes)
        The figure and its array of ``5 + 3 * n_somas`` axes; or the existing
        output ``Path`` when ``save=True``, ``overwrite=False`` and the file
        already exists.

    Raises
    ------
    ValueError
        If the acquisition has no soma with a calcium trace.
    """
    fp.style.use_lighttable()
    wis.util.checks.validate_ids(exp, loc, acq)
    out_dir, out_path = _resolve_out_path(
        out_dir, "all_soma_overview", f"{subject}--{exp}--{loc}--{acq}--all_somas.png"
    )
    if save and out_path.exists() and not overwrite:
        print(f"File already exists: {out_path}. Skipping.")
        return out_path

    soma_ids = _somas_with_ca_traces(subject, exp, loc, acq)
    if not soma_ids:
        raise ValueError(
            f"{subject} {exp} {loc} {acq} has no soma with a calcium trace "
            "(no ROI labels in dmd_info.yaml)."
        )

    # One synapse load for the whole acquisition; per-soma subsets are cheap.
    dn_all = wis.get.fetch_data_pipe(
        pipeline="synapse",
        subject=subject,
        exp=exp,
        loc=loc,
        acq=acq,
        soma_id=None,
        trace="denoised",
        channel=0,
        dend_only=True,
        valid_mask=False,
    )
    somas = []
    for soma_id in soma_ids:
        dn = dn_all.sx.somaid(soma_id)
        if dn.sizes["syn_id"] > 0:
            dn = wis.validity.attach_validity(dn)
        soma_ca, _spike_prob, spikes = wis.get.fetch_data_pipe(
            pipeline="soma",
            subject=subject,
            exp=exp,
            loc=loc,
            acq=acq,
            soma_id=soma_id,
            trace="Fraw",
            channel=0,
            return_dFF=True,
        )
        somas.append((soma_id, soma_ca, spikes, _syn_snr_image(dn)))
    ephys, eye, whisk, _hypno_acq, hypno_full, acq_dur = wis.get.fetch_data_pipe(
        pipeline="peripherals",
        subject=subject,
        exp=exp,
        loc=loc,
        acq=acq,
        stores=["EEG_", "loal"],
    )
    eeg = ephys["EEG_"]
    loal = ephys["loal"]

    n_rows = 5 + 3 * len(somas)
    f, ax = plt.subplots(
        n_rows, 1, figsize=(_FIG_WIDTH_IN, _ROW_HEIGHT_IN * n_rows), sharex=True
    )
    acq_mins = round(acq_dur / 60, 1)
    f.suptitle(
        f"{subject} -- {exp} -- {loc} -- {acq} | {len(somas)} somas | {acq_mins} mins",
        fontsize=18,
    )
    _plot_peripheral_panels(ax[:5], eeg, loal, eye, whisk, hypno_full)

    x_axes = [
        eeg.time.values,
        loal.time.values,
        eye["time"].to_numpy(),
        whisk["time"].to_numpy(),
        dn_all.time.values,
    ]
    for i, (soma_id, soma_ca, spikes, mfsnr) in enumerate(somas):
        row = 5 + 3 * i
        soma_dmd, soma_depth = wis.meta.get.soma_dmd(
            subject, exp, loc, acq, soma_id, return_depth=True
        )
        _plot_soma_panel(ax[row], soma_ca, spikes)
        ax[row].set_title(
            f"{soma_id} | dmd-{soma_dmd} | Soma Depth: {soma_depth:.1f}um",
            fontsize=12,
        )
        _plot_syn_heatmap_panel(ax[row + 1], mfsnr, 1)
        _plot_syn_heatmap_panel(ax[row + 2], mfsnr, 2)
        x_axes.append(soma_ca.time.values)
    _set_shared_xlim(ax[0], x_axes)

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        f.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return f, ax


def plot_nan_values(sx_array, ax=None, gap_factor=1.5):
    """Plot the NaN structure of a scopex array.

    Each row is one synapse, x is time. Green where the value is finite,
    magenta where the value is NaN, blank where the time axis itself has a gap
    (consecutive samples spaced > `gap_factor` x the median sampling interval).

    Parameters
    ----------
    sx_array : xr.DataArray
        Scopex DataArray with `time` and `syn_id` dims. If a `channel` dim of
        size 1 is present it is squeezed; >1 errors.
    ax : matplotlib.axes.Axes, optional
    gap_factor : float
        Intervals between consecutive original time samples larger than
        `gap_factor * median(dt)` are rendered as transparent gap cells.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.colors import ListedColormap

    da = sx_array
    if "channel" in da.dims:
        if da.sizes["channel"] == 1:
            da = da.squeeze("channel", drop=True)
        else:
            raise ValueError(
                f"`sx_array` has {da.sizes['channel']} channels — select one first."
            )
    if "time" not in da.dims or "syn_id" not in da.dims:
        raise ValueError("`sx_array` must have `time` and `syn_id` dims.")
    da = da.transpose("syn_id", "time")

    time = np.asarray(da["time"].values, dtype=float)
    if time.size < 2:
        raise ValueError("Need at least 2 timepoints to plot.")
    dts = np.diff(time)
    dt = float(np.median(dts))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Could not determine a positive median sampling interval.")

    # Uniform grid spanning the full time range; original samples drop into
    # their nearest cell, every other cell stays NaN (= rendered transparent).
    n_grid = int(np.round((time[-1] - time[0]) / dt)) + 1
    grid_idx = np.clip(np.round((time - time[0]) / dt).astype(int), 0, n_grid - 1)

    is_nan = np.isnan(da.values)  # (n_syn, n_time)
    n_syn = is_nan.shape[0]

    # 0 = valid (green), 1 = nan (magenta), NaN = gap (transparent)
    img = np.full((n_syn, n_grid), np.nan, dtype=float)
    img[:, grid_idx] = is_nan.astype(float)

    # Force-blank cells that fall inside a "real" time gap (>gap_factor*dt
    # between consecutive original samples), so a single straggling sample at
    # the end of a gap doesn't paint a long magenta/green stripe.
    gap_starts = np.where(dts > gap_factor * dt)[0]
    for k in gap_starts:
        a = grid_idx[k] + 1
        b = grid_idx[k + 1]
        if b > a:
            img[:, a:b] = np.nan

    cmap = ListedColormap([flex.green500, flex.blue600])
    cmap.set_bad("none")

    if ax is None:
        _, ax = plt.subplots(figsize=(14, max(2.0, 0.18 * n_syn)))

    ax.imshow(
        img,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(time[0] - dt / 2, time[0] + (n_grid - 0.5) * dt, -0.5, n_syn - 0.5),
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("synapse")
    ax.set_yticks(np.arange(n_syn))
    ax.set_yticklabels([str(s) for s in np.asarray(da["syn_id"].values)], fontsize=6)
    valid_frac = float(np.mean(~is_nan))
    ax.set_title(
        f"NaN map — {n_syn} syns, {time.size} samples, valid frac: {valid_frac:.3f}"
    )
    return ax


def exp_nan_plot(subject, exp, loc, acq, save=True, array=None):
    if array is None:
        denoised = wis.get.syn_dF(
            subject, exp, loc, acq, trace="denoised", channels=[0]
        )
        dn = wis.get.combine_scopex_arrays(denoised)
    else:
        dn = array
    hypno = wis.get.bout_hypno(subject, exp, loc, acq)
    hacq = hypno.trim(dn.time.values[0], dn.time.values[-1])
    f, ax = plt.subplots(
        2,
        1,
        figsize=(28, 15),
        gridspec_kw={"height_ratios": [1, 6]},
        layout="constrained",
    )
    plot_nan_values(dn, ax=ax[1])
    state_colors = {}
    state_colors["NREM"] = (2, flex.blue600)
    state_colors["REM"] = (3, flex.magenta600)
    state_colors["Wake"] = (1, flex.green600)
    state_colors.update({s: (1 if parent == 'Wake' else 2, wis.defs.state_colors[s])
                         for s, parent in wis.defs.substate_parent.items()})
    state_colors['Unscored'] = (0, '#777777')
    f, ax[0] = epy.plot.hypno.plot_basic_hypnogram(
        hacq,
        f,
        ax[0],
        xlim=(dn.time.values[0], dn.time.values[-1]),
        style_path=None,
        single_tone=False,
        state_colors=state_colors,
    )
    total_duration = dn.time.values[-1] - dn.time.values[0]
    broad_state = pl.col('state').replace(wis.defs.substate_parent)
    nrem_duration = hacq.df.filter(broad_state == "NREM")["duration"].sum()
    wake_duration = hacq.df.filter(broad_state == "Wake")["duration"].sum()
    rem_duration = hacq.df.filter(pl.col("state") == "REM")["duration"].sum()
    f.suptitle(
        f"{subject}, {exp}, {loc}, {acq} | Total Duration = {float(total_duration):.1f} s | "
        f"NREM: {float(nrem_duration):.1f} s | Wake: {float(wake_duration):.1f} s | REM: {float(rem_duration):.1f} s",
        fontsize=18,
    )
    if save:
        supplementary_plots_dir = os.path.join(
            wis.defs.annotation_root, subject, exp, loc, acq, "supplementary_plots"
        )
        if not os.path.exists(supplementary_plots_dir):
            os.makedirs(supplementary_plots_dir, exist_ok=True)
        path = os.path.join(supplementary_plots_dir, "Hypno_NaNs.png")
        f.savefig(path, dpi=300)
    return f, ax
