"""Loading one soma's inputs for the ON/OFF detector, with an on-disk cache.

:func:`load_acq` is the entry point: it runs the glutamate pursuit detector on
the soma's acquisition, keeps that soma's synapses over the analysis window,
and caches everything the detector and viewer need (~1-2 min to build, instant
afterwards). The cache lives at :func:`acq_cache_dir` unless a ``cache_dir``
is given.

A cache is rebuilt automatically when it was made by an older
:data:`CACHE_VERSION`, for different ids or duration, or from a different
ExperimentSummary version than the current mirror. It is **not** rebuilt when
synapse labels (soma-ID, dend-ID) change in WISynaptic: pass ``rebuild=True``
after relabelling.
"""

from __future__ import annotations

import json
import os
import shutil
import time as _time
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from ... import defs
from .core import CA_CHANNEL, GLU_CHANNEL, GLU_TAU_S, AcqData

#: Bump when the cached content changes, so every cache rebuilds on next load.
CACHE_VERSION = 1

#: Files a complete cache holds (``info.json`` is written last).
_CACHE_FILES = (
    "syn.parquet",
    "events.parquet",
    "time.npy",
    "detectable.npy",
    "mf_snr.npy",
    "dn_sum.npy",
    "hypno.parquet",
)


def acq_cache_dir(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma: str,
    duration_s: float | None = None,
) -> Path:
    """Default cache directory for one soma (and analysis duration)."""
    tag = "full" if duration_s is None else f"first{duration_s:g}s"
    return Path(defs.bistab_cache_root) / subject / exp / f"{loc}--{acq}--{soma}--{tag}"


def _exsum_version(subject: str, exp: str, loc: str, acq: str) -> str | None:
    """Version stem of the current ExSum mirror (``SummaryLoCo-...``), if any."""
    import wisco_slap as wis

    try:
        path = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    except (FileNotFoundError, ValueError):
        return None
    if not path or path == "NO_ESUM_MIRROR":
        return None
    return os.path.basename(path).split(".mat")[0]


def build_acq_cache(
    out: Path,
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma: str,
    duration_s: float | None = None,
    key: str | None = None,
) -> None:
    """Load scopex traces, run pursuit, and save one soma's detector inputs to ``out``.

    Noise models are fitted across each DMD's whole population (as in
    WISynaptic), then the soma's synapses and the first ``duration_s`` seconds
    (``None`` = the whole acquisition) are kept.
    """
    import wisco_slap as wis

    t_start = _time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Invalidate first: a crash mid-build must not leave a cache that loads.
    for name in ("info.json", f"dff_ch{CA_CHANNEL}.npy", f"dff_ch{GLU_CHANNEL}.npy"):
        (out / name).unlink(missing_ok=True)

    sx = dict(channels=[GLU_CHANNEL])
    dn = wis.get.combine_scopex_arrays(
        wis.get.syn_dF(subject, exp, loc, acq, trace="denoised", **sx)
    ).sel(channel=GLU_CHANNEL)
    ls = wis.get.combine_scopex_arrays(
        wis.get.syn_dF(subject, exp, loc, acq, trace="ls", **sx)
    ).sel(channel=GLU_CHANNEL)
    soma_ids = np.asarray(dn["soma-ID"].values).astype(str)
    soma_mask = soma_ids == soma
    if not soma_mask.any():
        raise ValueError(
            f"No synapses labelled {soma!r} in {subject} {exp} {loc} {acq}; "
            f"soma-IDs present: {sorted(set(soma_ids))}"
        )
    res = wis.scope.pursuit.detect_events(dn, ls, tau_s=GLU_TAU_S, verbose=False)
    mf = wis.scope.pursuit.matchfilt_snr(dn, ls, tau_s=GLU_TAU_S, nan_floor=True)
    dns = wis.scope.pursuit.denoised_sigma_units(dn, ls)
    det = wis.scope.pursuit.detectable_mask(ls, tau_s=GLU_TAU_S)

    t_all = np.asarray(dn.time.values, dtype=float)
    n_frames = (
        t_all.size
        if duration_s is None
        else int(np.searchsorted(t_all, t_all[0] + duration_s))
    )
    sel = dict(syn_id=np.asarray(dn.syn_id.values)[soma_mask])

    def cut(da: xr.DataArray) -> np.ndarray:
        return np.asarray(da.sel(**sel).transpose("syn_id", "time").values)[
            :, :n_frames
        ]

    meta = pl.DataFrame({
        "syn_id": sel["syn_id"].astype(str),
        "dend-ID": [str(v) for v in dn.sel(**sel)["dend-ID"].values],
        "dend_type": [str(v) for v in dn.sel(**sel)["dend_type"].values],
        "pos": np.asarray(dn.sel(**sel)["pos"].values, dtype=float),
        "dmd": np.asarray(dn.sel(**sel)["dmd"].values, dtype=int),
        "dmd-depth": np.asarray(dn.sel(**sel)["dmd-depth"].values, dtype=float),
    })
    meta.write_parquet(out / "syn.parquet")
    ev = res.events.filter(
        pl.col("syn_id").cast(pl.String).is_in(meta["syn_id"].to_list())
        & (pl.col("idx") < n_frames)
    ).select("syn_id", "idx", "time", "z_shot", "amp")
    ev.with_columns(pl.col("syn_id").cast(pl.String)).write_parquet(
        out / "events.parquet"
    )
    np.save(out / "time.npy", t_all[:n_frames])
    np.save(out / "detectable.npy", cut(det).astype(bool))
    np.save(out / "mf_snr.npy", cut(mf).astype(np.float16))
    dn_soma = cut(dns).astype(np.float64)
    dn_sum = np.where(np.isfinite(dn_soma).all(0), np.nansum(dn_soma, 0), np.nan)
    np.save(out / "dn_sum.npy", dn_sum.astype(np.float32))

    hyp = wis.get.bout_hypno(subject, exp, loc, acq).df
    t0, t1 = float(t_all[0]), float(t_all[n_frames - 1])
    hyp = hyp.filter((pl.col("end_time") > t0) & (pl.col("start_time") < t1)).select(
        pl.col("start_time").clip(t0, t1).alias("start_s"),
        pl.col("end_time").clip(t0, t1).alias("end_s"),
        pl.col("state").alias("label"),
    )
    hyp.write_parquet(out / "hypno.parquet")
    info = dict(
        cache_version=CACHE_VERSION,
        key=key,
        ids=[subject, exp, loc, acq, soma],
        n_syn=meta.height,
        n_frames=n_frames,
        t0=t0,
        duration_s=duration_s,
        exsum_version=_exsum_version(subject, exp, loc, acq),
        pursuit=dict(tau_s=GLU_TAU_S, z_floor=3.0, sigma_window_s=300.0),
        created=_time.strftime("%Y-%m-%dT%H:%M:%S"),
        build_s=round(_time.time() - t_start, 1),
    )
    (out / "info.json").write_text(json.dumps(info, indent=2, default=str))


def _stale_reason(
    d: Path, ids: tuple[str, ...], duration_s: float | None, check_exsum: bool
) -> str | None:
    """Why the cache in ``d`` cannot be used (``None`` = it can)."""
    info_path = d / "info.json"
    if not info_path.exists() or not all((d / f).exists() for f in _CACHE_FILES):
        return "missing"
    info = json.loads(info_path.read_text())
    if info.get("cache_version") != CACHE_VERSION:
        return f"cache_version {info.get('cache_version')} != {CACHE_VERSION}"
    if list(info.get("ids", [])) != list(ids):
        return f"built for {info.get('ids')}"
    # caches written before the move to wisco_slap stored the duration as "window_s"
    cached = info.get("duration_s", info.get("window_s"))
    if (cached is None) != (duration_s is None) or (
        cached is not None and abs(float(cached) - float(duration_s)) > 1e-9
    ):
        return f"built for duration_s={cached}"
    if check_exsum and info.get("exsum_version") is not None:
        now = _exsum_version(*ids[:4])
        if now is not None and now != info["exsum_version"]:
            return f"ExSum changed ({info['exsum_version']} -> {now})"
    return None


def load_acq(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma: str,
    *,
    duration_s: float | None = None,
    cache_dir: str | Path | None = None,
    key: str | None = None,
    rebuild: bool = False,
    check_exsum: bool = True,
    verbose: bool = True,
) -> AcqData:
    """One soma's detector inputs (builds the cache on first use, ~1-2 min).

    Parameters
    ----------
    subject, exp, loc, acq, soma
        The soma (``soma`` is the synapse ``soma-ID`` label, e.g. ``"soma1"``).
    duration_s
        Keep only the first ``duration_s`` seconds of the acquisition
        (``None`` = all of it). Useful for comparing cells over equal time.
    cache_dir
        Where to keep the cache; default :func:`acq_cache_dir`.
    key
        Short name for this soma (used in labels and cohort tables); default
        ``"subject-exp-loc-acq-soma"``.
    rebuild
        Rebuild even if the cache looks fresh (e.g. after relabelling synapses).
    check_exsum
        Rebuild when the ExSum mirror has changed since the cache was built.
    """
    ids = (subject, exp, loc, acq, soma)
    d = Path(cache_dir) if cache_dir is not None else acq_cache_dir(*ids, duration_s)
    key = key or "-".join(ids)
    reason = (
        "rebuild=True" if rebuild else _stale_reason(d, ids, duration_s, check_exsum)
    )
    if reason is not None:
        if verbose:
            print(f"bistab: building cache for {key} ({reason}) -> {d}")
        build_acq_cache(d, *ids, duration_s=duration_s, key=key)
    return AcqData(
        key=key,
        ids=ids,
        time=np.load(d / "time.npy"),
        syn=pl.read_parquet(d / "syn.parquet"),
        events=pl.read_parquet(d / "events.parquet"),
        detectable=np.load(d / "detectable.npy"),
        mf_snr=np.load(d / "mf_snr.npy", mmap_mode="r"),
        dn_sum=np.load(d / "dn_sum.npy"),
        hypno=pl.read_parquet(d / "hypno.parquet"),
        info=json.loads((d / "info.json").read_text()),
        cache_dir=d,
    )


def clear_acq_cache(
    subject: str,
    exp: str,
    loc: str,
    acq: str,
    soma: str,
    duration_s: float | None = None,
) -> None:
    """Delete the default cache directory of one soma (and duration)."""
    shutil.rmtree(
        acq_cache_dir(subject, exp, loc, acq, soma, duration_s), ignore_errors=True
    )


def load_eeg(acq: AcqData) -> xr.DataArray:
    """EEG (``EEG_`` store) over the analysis window."""
    import wisco_slap as wis

    subject, exp, loc, a, _ = acq.ids
    sb = wis.meta.get.sync_info()[subject][exp]["acquisitions"][f"{loc}--{a}"][
        "sync_block"
    ]
    eeg = wis.get.ephys(subject, exp, sb, stores=("EEG_",))["EEG_"]
    return eeg.sel(time=slice(float(acq.time[0]), float(acq.time[-1])))


def load_dff(acq: AcqData, channel: int) -> np.ndarray:
    """LS dF/F0 of one channel for the soma's synapses.

    ``(n_syn, n_frames)`` float16 in the same synapse order and analysis window
    as ``acq``. dF/F0 rather than dF so slow brightness changes (bleaching)
    cancel when comparing early vs late. Cached in ``acq.cache_dir`` when it is
    set (cleared whenever the main cache is rebuilt).
    """
    import wisco_slap as wis

    path = (
        None if acq.cache_dir is None else Path(acq.cache_dir) / f"dff_ch{channel}.npy"
    )
    if path is not None and path.exists():
        return np.load(path, mmap_mode="r")
    subject, exp, loc, a, _ = acq.ids
    sx = dict(channels=[channel])
    ls = wis.get.combine_scopex_arrays(
        wis.get.syn_dF(subject, exp, loc, a, trace="ls", **sx)
    )
    f0 = wis.get.combine_scopex_arrays(wis.get.syn_F0(subject, exp, loc, a, **sx))
    ids = acq.syn["syn_id"].to_list()
    dff = ls.sel(channel=channel, syn_id=ids) / f0.sel(channel=channel, syn_id=ids)
    arr = np.asarray(dff.transpose("syn_id", "time").values)[:, : acq.time.size].astype(
        np.float16
    )
    if path is None:
        return arr
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def load_ca(acq: AcqData) -> np.ndarray:
    """Synaptic calcium (jRGECO1a, channel 0) as LS dF/F0; see :func:`load_dff`."""
    return load_dff(acq, CA_CHANNEL)
