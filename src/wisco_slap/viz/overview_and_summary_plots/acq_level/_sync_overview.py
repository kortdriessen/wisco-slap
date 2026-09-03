"""Basic per-acquisition sync summary figure.

Shows the raw SYNC HDF5 channels for the sync block an acquisition belongs
to, with the ephys-start and acquisition-start samples that ``sync_info.yaml``
encodes marked on them, plus a trigger-line duty panel. It is the quickest
visual check that ``ephys_offset`` landed on the right acquisition.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import fluxplot as fp
import fluxplot.colors as flex
import matplotlib.pyplot as plt
import numpy as np

import slap2_py as spy
import wisco_slap as wis
from wisco_slap.viz.overview_and_summary_plots._spec import PlotSpec

#: Sample rate (Hz) of the DataRec SYNC HDF5 files.
SYNC_FS = 5000
#: Half-width (samples) of the zoomed panels around each marked sample.
DEFAULT_ZOOM_SAMPLES = 300
#: The wide ephys view runs from SYNC sample 0 to this many seconds after
#: ephys_start_sample.
EPHYS_WIDE_AFTER_S = 10.0
#: Half-width (s) of the wide scope-trigger view around acq_start_sample.
SCOPE_WIDE_HALF_S = 60.0
#: Tolerance (samples) when checking for an ephys edge at the ephys start.
EDGE_TOL_SAMPLES = 3
#: |ephys_offset - wall-clock gap| above which the plot prints a warning.
GAP_WARN_S = 10.0
#: |acq start - nearest acquisition-signature onset| above which to warn.
ONSET_WARN_S = 0.5
#: Bin width (s) of the trigger-line duty-cycle panel.
DUTY_BIN_S = 1.0
#: Seconds after acq start over which the trigger duty is checked.
DUTY_CHECK_S = 10.0
#: Trigger-line duty expected while acquiring (observed ≈ 0.98-0.995); a lower
#: value right after acq_start_sample means the marked start sits in preview
#: scanning rather than the acquisition.
ACQ_DUTY_MIN = 0.9


def basic_sync_summary_path(subject: str, exp: str, loc: str, acq: str) -> Path:
    """Return where :func:`basic_sync_summary_plot` saves its PNG for one acq.

    ``<annotation_root>/OVERVIEW_FIGURES/acq_level/basic_sync_summary/
    <subject>--<exp>--<loc>--<acq>.png``
    """
    return Path(
        os.path.join(
            wis.defs.annotation_root,
            "OVERVIEW_FIGURES",
            "acq_level",
            "basic_sync_summary",
            f"{subject}--{exp}--{loc}--{acq}.png",
        )
    )


def _zoom_window(center: int, half_width: int, n_samples: int) -> tuple[int, int, int]:
    """Return ``(start, stop, center_in_window)`` for a slice around ``center``.

    The window ``[start, stop)`` is ``center ± half_width`` clipped to
    ``[0, n_samples]``, and ``center_in_window`` is where ``center`` lands
    inside that slice (``half_width`` unless clipped at the left edge). When
    ``center`` lies entirely outside the array the window is empty
    (``start == stop``).
    """
    start = min(max(0, center - half_width), n_samples)
    stop = max(start, min(n_samples, center + half_width))
    return start, stop, center - start


def _to_datetime(value: str | datetime) -> datetime:
    """Parse a ``sync_info.yaml`` timestamp (string or already a datetime)."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _has_edge_near(signal: np.ndarray, sample: int, tol: int, win: int = 20) -> bool:
    """Shared with the sync metadata code (``wisco_slap.meta.sync.ephys_edge_near``)."""
    return wis.meta.sync.ephys_edge_near(signal, sample, tol=tol, win=win)


def _duty_after(scope: np.ndarray, sample: int, half: float, seconds: float) -> float:
    """Fraction of trigger samples above ``half`` in ``[sample, sample + seconds)``."""
    a = max(0, sample)
    b = min(scope.size, sample + int(seconds * SYNC_FS))
    if b <= a:
        return float("nan")
    return float(np.mean(scope[a:b] > half))


def _locate_starts(
    scope: np.ndarray,
    ephys: np.ndarray,
    offset_vdaq_s: float,
    wall_gap_s: float,
    stored_ephys_start: int | None,
) -> tuple[int, int, np.ndarray, bool, str, str, list[str]]:
    """Recover the ``(ephys_start_sample, acq_start_sample)`` sync_info encodes.

    ``ephys_offset_uncorrected`` is ``(acq start − ephys_start_sample) / fs`` on
    the vDaq sample axis. When ``sync_info.yaml`` stores the block's
    ``ephys_start_sample`` (persisted by ``update_sync_info`` since 2026-08-31)
    it is used directly. Otherwise work backwards: take the
    acquisition-signature onset (:func:`slap2_py.utils.drec.generate_acq_segment_df`;
    scope-UP window as a fallback) nearest the wall-clock gap as the acq start and
    subtract the offset — this recovers a manual override that was never
    persisted. The auto-detector only serves as a cross-check.

    Returns
    -------
    ephys_start_sample, acq_start_sample, onsets, edge_ok, ephys_note, acq_note,
    warnings
        ``onsets`` are all acquisition-signature onset samples found in the
        scope channel (may be empty); ``edge_ok`` is whether an ephys edge sits
        within ``EDGE_TOL_SAMPLES`` of ``ephys_start_sample``.
    """
    offset_samples = int(round(offset_vdaq_s * SYNC_FS))
    warnings: list[str] = []
    try:
        detected = wis.meta.sync.detect_ephys_start_sample(ephys, fs=SYNC_FS)
        detector_note = f"auto-detector: {detected}"
    except ValueError:
        detected = None
        detector_note = "auto-detector: not detected"

    segments = spy.utils.drec.generate_acq_segment_df(scope, fs=SYNC_FS)
    onsets = segments["start_idx"].to_numpy().astype(int)
    cands, cand_kind = onsets, "acquisition-signature onset"
    if cands.size == 0:
        windows = spy.utils.drec.generate_scope_index_df(scope, fs=SYNC_FS)
        cands = windows["start_idx"].to_numpy().astype(int)
        cand_kind = "scope-UP window start (no acquisition-duty segment found)"

    if stored_ephys_start is not None:
        ephys_start_sample = int(stored_ephys_start)
        acq_start_sample = ephys_start_sample + offset_samples
        ephys_note = f"stored in sync_info · {detector_note}"
        acq_note = "ephys_start_sample + ephys_offset"
    elif cands.size:
        # Select like the matcher does: compare rise-anchored offsets to the
        # wall-clock gap. Without a detected ephys start the raw file time is
        # the best available approximation (ephys usually starts within ~10 s
        # of the SYNC file).
        anchor = detected if detected is not None else 0
        i = int(np.argmin(np.abs((cands - anchor) / SYNC_FS - wall_gap_s)))
        acq_start_sample = int(cands[i])
        ephys_start_sample = acq_start_sample - offset_samples
        acq_note = f"{cand_kind} {i + 1}/{cands.size}"
        ephys_note = f"inferred = acq start − ephys_offset · {detector_note}"
    else:
        ephys_start_sample = detected if detected is not None else 0
        acq_start_sample = ephys_start_sample + offset_samples
        acq_note = "no scope-UP window found; ephys_start_sample + ephys_offset"
        ephys_note = (
            detector_note if detected is not None else "NOT DETECTED, assuming 0"
        )
        warnings.append("no scope-UP window (≥ 30 s) found in the SYNC scope channel")

    if detected is not None and abs(detected - ephys_start_sample) > EDGE_TOL_SAMPLES:
        warnings.append(
            f"ephys_start_sample {ephys_start_sample} DISAGREES with the "
            f"auto-detector ({detected})"
        )
    edge_ok = _has_edge_near(ephys, ephys_start_sample, EDGE_TOL_SAMPLES)
    if not edge_ok:
        warnings.append(
            f"no ephys edge within ±{EDGE_TOL_SAMPLES} samples of ephys_start_sample "
            f"{ephys_start_sample}"
        )
    return (
        ephys_start_sample,
        acq_start_sample,
        onsets,
        edge_ok,
        ephys_note,
        acq_note,
        warnings,
    )


def basic_sync_summary_plot(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    *,
    zoom_samples: int = DEFAULT_ZOOM_SAMPLES,
    dpi: int = 150,
    overwrite: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the SYNC-file ephys and scope-trigger channels for one acquisition.

    Seven rows, top to bottom:

    1. Full ``electrophysiology`` channel with the ephys-start sample that
       ``sync_info.yaml`` encodes (see :func:`_locate_starts`).
    2. Wide view of (1): SYNC sample 0 to ``EPHYS_WIDE_AFTER_S`` after that
       sample (absolute sample index on the x-axis).
    3. Zoom of (1) around that sample — there should be an ephys edge there.
    4. Full ``slap2_acquiring_trigger`` channel with the acquisition-start
       sample ``ephys_start_sample + ephys_offset_uncorrected * fs``.
    5. Wide view of (4): ``± SCOPE_WIDE_HALF_S`` around that sample (absolute
       sample index; orange dotted = wall-clock-implied start, as in row 7).
    6. Zoom of (4) around that sample — there should be a scope rising edge.
    7. Trigger-line duty per second (fraction of samples high). An acquisition
       runs at duty ≈ 1; preview scanning is much lower. Marks the acq start
       (green) and the wall-clock-implied start ``ephys_start_sample +
       (dat_start − ephys_start)·fs`` (orange dotted) — they should coincide,
       and both should sit where the duty jumps to ≈ 1.

    Panel titles report whether an ephys edge sits at the ephys start, what
    the auto-detector says, how far the acq start is from the nearest
    acquisition-signature onset, ``ephys_offset`` vs the wall-clock gap
    (their difference should be ~0 for a correct sync), and the trigger duty
    right after the acq start. Anything suspicious is also printed as a
    WARNING.

    The figure is saved to :func:`basic_sync_summary_path` when no file exists
    there yet, or when ``overwrite=True``; otherwise the existing file is left
    alone. The figure is always built and returned.

    Parameters
    ----------
    subject, exp, loc, acq
        Acquisition identifiers, e.g. ``"alnair", "exp_1", "loc_D", "acq_1"``.
    zoom_samples
        Half-width, in SYNC samples (5 kHz), of the two zoomed panels.
    dpi
        Resolution of the saved PNG.
    overwrite
        Re-save even if the PNG already exists.

    Returns
    -------
    (fig, axes)
        The figure and its ``(7,)`` array of axes.

    Raises
    ------
    RuntimeError
        If the acquisition's SYNC HDF5 file is corrupt and cannot be loaded.
    """
    fp.style.use_lighttable()
    wis.util.checks.validate_ids(exp, loc, acq)

    si = wis.meta.get.sync_info()
    acq_info = si[subject][exp]["acquisitions"][f"{loc}--{acq}"]
    acq_sb = acq_info["sync_block"]
    block_info = si[subject][exp]["sync_blocks"][acq_sb]
    n_epochs = acq_info.get("n_epochs", 1)
    # ephys_offset is TDT-clock seconds; sample arithmetic on the SYNC file
    # needs the raw vDaq measurement (identical when no clock ratio was applied).
    ephys_offset_seconds = float(acq_info["ephys_offset"])
    offset_vdaq_s = acq_info.get("ephys_offset_uncorrected")
    if offset_vdaq_s is None:
        offset_vdaq_s = ephys_offset_seconds * float(
            acq_info.get("clock_ratio_applied", 1.0)
        )
    offset_vdaq_s = float(offset_vdaq_s)

    scope, ephys = wis.meta.sync.load_sync_block(subject, exp, acq_sb)
    if isinstance(scope, str):  # load_sync_block returns ("Corrupt", "Corrupt")
        raise RuntimeError(
            f"SYNC file for {subject}/{exp} sync_block-{acq_sb} is corrupt; "
            f"cannot draw a sync summary for {loc}--{acq}."
        )
    scope = np.asarray(scope)
    ephys = np.asarray(ephys)

    wall_gap_s = (
        _to_datetime(acq_info["dat_start"]) - _to_datetime(block_info["ephys_start"])
    ).total_seconds()
    (
        ephys_start_sample,
        acq_start_sample,
        onsets,
        edge_ok,
        ephys_note,
        acq_note,
        warnings,
    ) = _locate_starts(
        scope, ephys, offset_vdaq_s, wall_gap_s, block_info.get("ephys_start_sample")
    )
    gap_delta = ephys_offset_seconds - wall_gap_s
    if abs(gap_delta) > GAP_WARN_S:
        warnings.append(
            f"ephys_offset ({ephys_offset_seconds:.3f} s) differs from the wall-clock "
            f"gap ({wall_gap_s:.1f} s) by {gap_delta:+.1f} s"
        )
    if onsets.size:
        onset_delta_s = float(
            (onsets[np.argmin(np.abs(onsets - acq_start_sample))] - acq_start_sample)
            / SYNC_FS
        )
        onset_note = f"nearest acquisition-signature onset: {onset_delta_s:+.3f} s"
        if abs(onset_delta_s) > ONSET_WARN_S:
            warnings.append(
                f"acq_start_sample is {onset_delta_s:+.1f} s from the nearest "
                "acquisition-signature onset (trigger duty ≥ 0.9)"
            )
    else:
        onset_note = "no acquisition-signature onset found"
    duty_x, duty, half = spy.utils.drec.trigger_duty(
        scope, fs=SYNC_FS, bin_s=DUTY_BIN_S
    )
    duty_after = _duty_after(scope, acq_start_sample, half, DUTY_CHECK_S)
    wallclock_start_sample = ephys_start_sample + int(round(wall_gap_s * SYNC_FS))
    if duty_after < ACQ_DUTY_MIN:
        warnings.append(
            f"trigger-line duty in the {DUTY_CHECK_S:.0f} s after acq_start_sample is "
            f"{duty_after:.2f} (≈1 expected while acquiring) — the marked start sits "
            "in preview scanning, not the acquisition"
        )
    for w in warnings:
        print(f"WARNING {subject}/{exp} {loc}--{acq} (sync_block-{acq_sb}): {w}")

    f, ax = plt.subplots(7, 1, figsize=(28, 24), layout="constrained")
    ax_ephys, ax_ephys_wide, ax_ephys_zoom = ax[0], ax[1], ax[2]
    ax_scope, ax_scope_wide, ax_scope_zoom, ax_duty = ax[3], ax[4], ax[5], ax[6]

    # --- ephys channel: full + wide + zoom -----------------------------------
    ax_ephys.plot(ephys, color=flex.base850, linewidth=0.5)
    ax_ephys.axvline(ephys_start_sample, color=flex.red600, linestyle="--")
    ax_ephys.set_title(
        f"SYNC electrophysiology — ephys_start_sample = {ephys_start_sample} "
        f"({ephys_start_sample / SYNC_FS:.3f} s; {ephys_note}) · "
        f"ephys edge at sample: {'yes' if edge_ok else 'NO'}"
    )
    w2 = min(ephys.size, max(0, ephys_start_sample + int(EPHYS_WIDE_AFTER_S * SYNC_FS)))
    if w2 > 0:
        ax_ephys_wide.plot(np.arange(w2), ephys[:w2], color=flex.base850, linewidth=0.6)
        ax_ephys_wide.axvline(ephys_start_sample, color=flex.red600, linestyle="--")
        ax_ephys_wide.set_title(
            f"wide: SYNC sample 0 → ephys_start_sample + {EPHYS_WIDE_AFTER_S:.0f} s "
            f"(0 → {w2 / SYNC_FS:.1f} s)"
        )
    else:
        ax_ephys_wide.set_title(
            f"wide view unavailable: ephys_start_sample = {ephys_start_sample} lies "
            "before the SYNC file"
        )
    z1, z2, center = _zoom_window(ephys_start_sample, zoom_samples, ephys.size)
    if z2 > z1:
        ax_ephys_zoom.plot(ephys[z1:z2], color=flex.base850, linewidth=0.8)
        ax_ephys_zoom.axvline(center, color=flex.red600, linestyle="--")
        ax_ephys_zoom.set_title(f"zoom: ephys_start_sample ± {zoom_samples} samples")
    else:
        ax_ephys_zoom.set_title(
            f"zoom unavailable: ephys_start_sample = {ephys_start_sample} lies "
            f"outside the SYNC file (0..{ephys.size - 1}) — check ephys_offset"
        )

    # --- scope trigger channel: full + wide + zoom ---------------------------
    ax_scope.plot(scope, color=flex.base850, linewidth=0.5)
    ax_scope.axvline(acq_start_sample, color=flex.green600, linestyle="--")
    ax_scope.set_title(
        f"SYNC slap2_acquiring_trigger — acq_start_sample = {acq_start_sample} "
        f"({acq_start_sample / SYNC_FS:.3f} s; {acq_note}) · {onset_note}\n"
        f"ephys_offset = {ephys_offset_seconds:.3f} s "
        f"(TDT; vDaq {offset_vdaq_s:.4f} s) "
        f"vs wall-clock gap {wall_gap_s:.1f} s (Δ {gap_delta:+.1f} s)"
    )
    half_wide = int(SCOPE_WIDE_HALF_S * SYNC_FS)
    z1, z2, _ = _zoom_window(acq_start_sample, half_wide, scope.size)
    if z2 > z1:
        ax_scope_wide.plot(
            np.arange(z1, z2), scope[z1:z2], color=flex.base850, linewidth=0.6
        )
        ax_scope_wide.axvline(acq_start_sample, color=flex.green600, linestyle="--")
        ax_scope_wide.axvline(
            wallclock_start_sample, color=flex.orange600, linestyle=":"
        )
        ax_scope_wide.set_title(
            f"wide: acq_start_sample ± {SCOPE_WIDE_HALF_S:.0f} s "
            f"({z1 / SYNC_FS:.1f} → {z2 / SYNC_FS:.1f} s) — green dashed: "
            "acq_start_sample · orange dotted: wall-clock-implied start"
        )
    else:
        ax_scope_wide.set_title(
            f"wide view unavailable: acq_start_sample = {acq_start_sample} lies "
            f"outside the SYNC file (0..{scope.size - 1})"
        )
    z1, z2, center = _zoom_window(acq_start_sample, zoom_samples, scope.size)
    if z2 > z1:
        ax_scope_zoom.plot(scope[z1:z2], color=flex.base850, linewidth=0.8)
        ax_scope_zoom.axvline(center, color=flex.green600, linestyle="--")
        ax_scope_zoom.set_title(f"zoom: acq_start_sample ± {zoom_samples} samples")
    else:
        ax_scope_zoom.set_title(
            f"zoom unavailable: acq_start_sample = {acq_start_sample} lies outside "
            f"the SYNC file (0..{scope.size - 1}) — check ephys_offset / ephys start"
        )

    # --- trigger-line duty ------------------------------------------------------
    if duty.size:
        ax_duty.step(duty_x, duty, where="post", color=flex.base850, linewidth=0.8)
    ax_duty.set_ylim(-0.02, 1.05)
    ax_duty.axvline(acq_start_sample, color=flex.green600, linestyle="--")
    ax_duty.axvline(wallclock_start_sample, color=flex.orange600, linestyle=":")
    ax_duty.set_title(
        f"trigger-line duty per {DUTY_BIN_S:.0f} s (fraction of samples high) — "
        "green dashed: acq_start_sample · orange dotted: wall-clock-implied start "
        f"(sample {wallclock_start_sample}) · duty in {DUTY_CHECK_S:.0f} s after "
        f"acq start = {duty_after:.2f}"
    )

    for axis in ax:
        axis.set_xlabel(f"sample ({SYNC_FS} Hz)")

    f.suptitle(
        f"{subject} / {exp} / {loc}--{acq} / sync_block-{acq_sb} / n_epochs={n_epochs}"
    )

    out_path = basic_sync_summary_path(subject, exp, loc, acq)
    if overwrite or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f.savefig(out_path, dpi=dpi)
        print(f"Saved sync summary to {out_path}")
    else:
        print(f"Sync summary already exists (overwrite=False): {out_path}")
    return f, ax


PLOT_SPEC = PlotSpec(
    name="basic_sync_summary",
    plot=basic_sync_summary_plot,
    path=basic_sync_summary_path,
)
