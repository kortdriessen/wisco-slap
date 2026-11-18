"""NREM SWS slow-wave locking for one soma, matching perfect_sw_plot.qmd."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import electro_py as epy
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import wisco_slap as wis
from wisco_slap.viz._ephys_locking import ALIGN_COLUMNS, DEFAULT_LAGS, wave_locked_overview
from wisco_slap.viz.overview_and_summary_plots._spec import PlotSpec


def single_soma_sw_locking_path(
    subject: str, exp: str, loc: str, acq: str, soma_id: str
) -> Path:
    """PNG in OVERVIEW_FIGURES/single_soma/single_soma_sw_locking."""
    return (
        Path(wis.defs.annotation_root) / "OVERVIEW_FIGURES" / "single_soma"
        / "single_soma_sw_locking"
        / f"{subject}--{exp}--{loc}--{acq}--{soma_id}.png"
    )


def detect_stage_waves(eeg, hypno, *, stage: str = "NREM SWS"):
    """Calibrate gates on ``stage`` and return only waves with troughs in it.

    Match the notebook's negative polarity, 0.15–1 s negative half-wave,
    and MAD k=2.5. Never accept the detector's whole-recording fallback when
    the requested stage has no EEG samples. ``None`` means no such stage.
    """
    bouts = hypno.df.filter(pl.col("state") == stage)
    if bouts.is_empty():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="compute_on_stages=.*matched no samples.*")
        try:
            waves = epy.wavedet.detect_slow_waves(
                sig=eeg, polarity="negative", dur_neg=(0.15, 1.0),
                amp_threshold="mad", amp_threshold_params={"k": 2.5},
                hypno=hypno, compute_on_stages=[stage],
            )
        except UserWarning as exc:
            if "matched no samples" in str(exc):
                return None
            raise
    return waves.keep_stage(stage)


def _no_waves_figure(member: str, message: str) -> tuple[plt.Figure, dict]:
    """Visible coverage record when a wave-triggered average is undefined."""
    fig, ax = plt.subplots(figsize=(7.2, 12.5), layout="constrained")
    ax.axis("off")
    ax.text(0.5, 0.8, member.replace("--", " / "), ha="center", wrap=True)
    ax.text(0.5, 0.55, message, ha="center", va="center", wrap=True, fontsize=13)
    ax.text(0.5, 0.35, "No wave-triggered average can be estimated.\n"
            "Detection was not recalibrated on another sleep stage.", ha="center")
    return fig, {"notice": ax}


def single_soma_sw_locking_plot(
    subject: str, exp: str, loc: str, acq: str, soma_id: str,
    *, overwrite: bool = False, stage: str = "NREM SWS",
    align: str = "trough", dpi: int = 180,
) -> tuple[plt.Figure, dict] | None:
    """Load, plot and save one soma using the approved notebook's defaults.

    The heatmap uses denoised glutamate in LS-noise units, minus each
    synapse's recording mean. EEG, synaptic sum, soma Fsvd dF/F and CASCADE
    spike probability retain their loader-provided ephys time axes. Gates
    are calibrated on full-sync-block NREM SWS; only NREM SWS troughs with
    the default ±1.5 s window inside the synapse time range are plotted.

    Save the exact selected wave table and a JSON provenance/count sidecar.
    With no usable waves, save an explicitly labelled coverage figure.
    Existing PNGs are skipped unless ``overwrite=True``.
    """
    if align not in ALIGN_COLUMNS:
        raise ValueError(f"align must be one of {sorted(ALIGN_COLUMNS)}")
    out = single_soma_sw_locking_path(subject, exp, loc, acq, soma_id)
    if out.exists() and not overwrite:
        return None
    member = "--".join((subject, exp, loc, acq, soma_id))
    sb = wis.meta.sync.get_acq_sync_block(subject, exp, loc, acq)
    eeg = wis.get.ephys(subject, exp, sb, stores=["EEG_"])["EEG_"]
    hypno = wis.get.bout_hypno(subject, exp, loc, acq, apply_substates=True)
    if hypno is None:
        raise FileNotFoundError(f"No hypnogram for {member}")
    waves = detect_stage_waves(eeg, hypno, stage=stage)
    metadata = dict(
        member=member, sync_block=str(sb), stage=stage, align=align,
        compute_on_stages=[stage], plot_stages=[stage],
        polarity="negative", dur_neg=[0.15, 1.0],
        amp_threshold="mad", amp_threshold_params={"k": 2.5},
        calibration_scope="full sync block", hypnogram="default refined loader",
        window_s=[float(DEFAULT_LAGS[0]), float(DEFAULT_LAGS[-1])],
        synapse_units="denoised / LS sigma0", heatmap_units="delta",
        soma_trace="Fsvd dF/F", population="sum", n_synapses=None,
        n_detected_stage_waves=0 if waves is None else waves.df.height,
        n_plotted_waves=0, status="no_waves", reason="",
    )
    selected = pl.DataFrame()
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False}):
        if waves is None or waves.df.is_empty():
            metadata["reason"] = f"No {stage} waves detected in the sync block."
            fig, axes = _no_waves_figure(member, metadata["reason"])
        else:
            common = dict(subject=subject, exp=exp, loc=loc, acq=acq,
                          soma_id=soma_id, channel=1, dend_only=True, valid_mask=True)
            ls = wis.get.fetch_data_pipe("synapse", trace="ls", **common).sx.unify_valid_samples()
            dn = wis.get.fetch_data_pipe("synapse", trace="denoised", **common).sx.unify_valid_samples()
            dn_sigma = wis.scope.pursuit.denoised_sigma_units(dn, ls)
            del dn, ls
            if dn_sigma.sizes.get("syn_id", 0) == 0 or dn_sigma.sizes.get("time", 0) < 2:
                raise ValueError(f"No usable dendritic synapse array for {member}")
            margin = float(np.max(np.abs(DEFAULT_LAGS)))
            t0, t1 = np.asarray(dn_sigma.time.values)[[0, -1]]
            col = ALIGN_COLUMNS[align]
            selected = waves.df.filter(
                (pl.col("stage") == stage)
                & (pl.col(col) - margin > t0) & (pl.col(col) + margin < t1)
            )
            metadata.update(n_synapses=dn_sigma.sizes["syn_id"],
                            synapse_time_range_s=[float(t0), float(t1)],
                            n_plotted_waves=selected.height)
            if selected.is_empty():
                metadata["reason"] = f"No {stage} waves with a complete ±1.5 s window\ninside this soma acquisition."
                fig, axes = _no_waves_figure(member, metadata["reason"])
            else:
                soma_ca, spike_prob, _ = wis.get.fetch_data_pipe(
                    "soma", subject=subject, exp=exp, loc=loc, acq=acq,
                    soma_id=soma_id, trace="Fsvd", channel=0, return_dFF=True,
                )
                fig, axes = wave_locked_overview(
                    dn_sigma, selected, eeg, soma_ca, spike_prob,
                    stages=stage, align=align, units_label="σ0",
                )
                axes["eeg"].set_title(
                    member.replace("--", " / ") + "\n" + axes["eeg"].get_title(),
                    fontsize=8.5, pad=34,
                )
                metadata["status"] = "generated"
        out.parent.mkdir(parents=True, exist_ok=True)
        pending = out.with_suffix(".pending.png")
        fig.savefig(pending, dpi=dpi, facecolor="white", transparent=False, bbox_inches="tight")
        pending.replace(out)
    if selected.width:
        selected.write_parquet(out.with_suffix(".waves.parquet"))
    else:
        out.with_suffix(".waves.parquet").unlink(missing_ok=True)
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return fig, axes


PLOT_SPEC = PlotSpec(
    name="single_soma_sw_locking", plot=single_soma_sw_locking_plot,
    path=single_soma_sw_locking_path,
)
