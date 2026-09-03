# ============================================
# Meta File that this handles: sync_info.yaml
# ============================================

import os

import numpy as np
import pandas as pd
import yaml

import electro_py as epy
import slap2_py as spy
import wisco_slap as wis
import wisco_slap.defs as DEFS
from wisco_slap.meta.get import sync_info


def load_sync_block(
    subject: str, exp: str, sync_block: int
) -> tuple[np.ndarray, np.ndarray] | tuple[str, str]:
    """Load a sync block for a given subject and experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    sync_block : int
        The sync block number.
    """
    sync_blocks, sync_paths = get_all_sync_paths(subject, exp)
    sync_path = sync_paths[sync_block - 1]
    try:
        print(f"Loading sync block {sync_block} from {sync_path}")
        full_sync = spy.utils.drec.load_datarec_file(sync_path)
        return (
            full_sync["slap2_acquiring_trigger"][:],
            full_sync["electrophysiology"][:],
        )
    except OSError:
        recovered = recovered_sync_path(subject, exp, sync_block)
        if recovered is None:
            return "Corrupt", "Corrupt"
        print(
            f"Raw SYNC file is corrupt; using recovered copy {recovered} "
            "(see slap2_py.utils.sync_recovery)"
        )
        full_sync = spy.utils.drec.load_datarec_file(recovered)
        return (
            full_sync["slap2_acquiring_trigger"][:],
            full_sync["electrophysiology"][:],
        )


def recovered_sync_path(subject: str, exp: str, sync_block: int) -> str | None:
    """Path of a recovered copy of this block's SYNC file, or None.

    Recovered files live at ``defs.sync_recovery_root/<subject>/<exp>/`` under
    the raw file's basename (written by ``slap2_py.utils.sync_recovery`` /
    ``wisco-slap/recover_sync_layout.py``).
    """
    sync_blocks, sync_paths = get_all_sync_paths(subject, exp)
    if sync_block - 1 >= len(sync_paths):
        return None
    candidate = os.path.join(
        DEFS.sync_recovery_root,
        subject,
        exp,
        os.path.basename(sync_paths[sync_block - 1]),
    )
    return candidate if os.path.isfile(candidate) else None


def sync_source(subject: str, exp: str, sync_block: int) -> str:
    """``"raw"`` if the raw SYNC file opens, ``"recovered"`` if only a
    recovered copy does, else ``"corrupt"``."""
    sync_blocks, sync_paths = get_all_sync_paths(subject, exp)
    try:
        with spy.utils.drec.load_datarec_file(sync_paths[sync_block - 1]):
            return "raw"
    except OSError:
        return (
            "recovered" if recovered_sync_path(subject, exp, sync_block) else "corrupt"
        )


def get_all_sync_paths(subject, exp) -> tuple[list[int], list[str]]:
    """Get all sync paths for a given subject and experiment.

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.

    Returns
    -------
    tuple[list[int], list[str]]
        A list of sync paths.
    """
    exp_dir = f"{DEFS.data_root}/{subject}/{exp}"
    sync_files = [f for f in os.listdir(exp_dir) if "SYNC_" in f]
    sync_blocks = [int(f.split("SYNC_")[-1].split(".")[0]) for f in sync_files]
    sync_blocks.sort()
    sync_paths = [os.path.join(exp_dir, f) for f in sync_files]
    sync_paths.sort()
    return sync_blocks, sync_paths


#: Warn when the best-matching acquisition onset is further than this from
#: the wall-clock gap between the DAT filename timestamp and the TDT block start.
MATCH_WARN_S = 30.0


def load_pulse_channel(subject: str, exp: str, sync_block: int) -> np.ndarray | None:
    """Return the shared-pulse channel ('rz_pulse') from a SYNC file, or None.

    The RZ2 pulse-generator gizmo (standing setup since 2026-08-31) is teed
    into both a TDT store ('p_in') and a vDaq DataRecorder channel
    ('rz_pulse'); blocks recorded before that simply lack the channel.
    """
    _sbs, sps = get_all_sync_paths(subject, exp)
    try:
        f = spy.utils.drec.load_datarec_file(sps[_sbs.index(sync_block)])
    except OSError:
        # mirror load_sync_block's fallback to a recovered copy
        recovered = recovered_sync_path(subject, exp, sync_block)
        if recovered is None:
            return None
        f = spy.utils.drec.load_datarec_file(recovered)
    if "rz_pulse" not in f:
        return None
    return f["rz_pulse"][:]


def _rising_edges_subsample(x: np.ndarray, thresh: float) -> np.ndarray:
    """Fractional sample indices where x rises through thresh."""
    x = np.asarray(x, dtype=float)
    k = np.flatnonzero((x[:-1] < thresh) & (x[1:] >= thresh))
    return k + (thresh - x[k]) / (x[k + 1] - x[k])


def calibrate_block_from_pulses(
    pulse_vdaq: np.ndarray,
    tdt_block_path: str,
    coarse_anchor_idx: float,
    fs: float = 5000.0,
) -> dict | None:
    """Per-block clock calibration from the shared RZ2 pulse train.

    Pairs rising edges of the vDaq 'rz_pulse' channel with the same physical
    edges on the TDT side (the 'p_in' loopback store; 'spul' epoc onsets as a
    fallback, which adds the ~0.27 ms DAC->analog-in loopback latency to the
    anchor) and fits ``vdaq_idx = anchor + t_tdt * fs * ratio``.

    ``coarse_anchor_idx`` (the #Enable-rise sample, accurate to ~+/-0.5 ms)
    disambiguates pulse pairing; the returned anchor supersedes it at
    ~+/-0.05 ms precision.

    Returns ``{"pulse_anchor_idx", "pulse_clock_ratio", "pulse_n",
    "pulse_resid_ms", "pulse_source"}`` or None (with a printed reason) when
    the block has no usable train or the fit fails sanity gates — callers
    should then fall back to the enable-rise + fleet-constant method.
    """
    import tdt as _tdt

    # TDT-side pulse times (seconds on the block's time base)
    t_tdt, source = None, None
    try:
        blk = _tdt.read_block(tdt_block_path, store="p_in")
        pin = np.ravel(blk.streams.p_in.data)
        thr = (np.percentile(pin, 1) + np.percentile(pin, 99.9)) / 2
        if thr > 0.1:  # a real train, not noise
            t_tdt = _rising_edges_subsample(pin, thr) / float(blk.streams.p_in.fs)
            source = "p_in"
    except Exception:
        pass
    if t_tdt is None or len(t_tdt) < 30:
        try:
            blk = _tdt.read_block(tdt_block_path, evtype=["epocs"])
            t_tdt = np.asarray(blk.epocs.spul.onset, dtype=float)
            source = "spul_epocs"
        except Exception:
            print("pulse calibration: no p_in store / spul epocs in TDT block")
            return None
    if len(t_tdt) < 30:
        print(f"pulse calibration: only {len(t_tdt)} TDT pulses — skipping")
        return None

    # vDaq-side edges
    thr_v = (np.percentile(pulse_vdaq, 1) + np.percentile(pulse_vdaq, 99.9)) / 2
    if thr_v < 0.1:
        print("pulse calibration: vDaq rz_pulse channel carries no train")
        return None
    e_vdaq = _rising_edges_subsample(pulse_vdaq, thr_v)

    # pair via the coarse anchor; window well under the pulse period
    period = float(np.median(np.diff(t_tdt)))
    win = min(0.25, 0.4 * period)
    t_v = e_vdaq / fs
    pairs = []
    for t in t_tdt:
        j = np.argmin(np.abs(t_v - (t + coarse_anchor_idx / fs)))
        if abs(t_v[j] - (t + coarse_anchor_idx / fs)) < win:
            pairs.append((e_vdaq[j], t))
    pairs = np.array(pairs)
    if len(pairs) < 30:
        print(f"pulse calibration: only {len(pairs)} paired pulses — skipping")
        return None

    X = np.vstack([np.ones(len(pairs)), pairs[:, 1]]).T
    (anchor, slope), *_ = np.linalg.lstsq(X, pairs[:, 0], rcond=None)
    ratio = slope / fs
    resid_ms = (pairs[:, 0] - X @ np.array([anchor, slope])) / fs * 1e3
    resid_sd = float(resid_ms.std(ddof=1))
    if abs(ratio - 1) > 100e-6 or resid_sd > 2.0:
        print(
            f"pulse calibration failed sanity gates (ratio {(ratio - 1) * 1e6:+.1f} ppm, "
            f"resid SD {resid_sd:.2f} ms) — falling back to enable-rise method"
        )
        return None
    print(
        f"pulse calibration [{source}]: {len(pairs)} pulses, anchor "
        f"{anchor:.1f} (vs rise {coarse_anchor_idx:.0f}: "
        f"{(anchor - coarse_anchor_idx) / fs * 1e3:+.2f} ms), ratio "
        f"{(ratio - 1) * 1e6:+.3f} ppm, resid SD {resid_sd:.3f} ms"
    )
    return {
        "pulse_anchor_idx": float(anchor),
        "pulse_clock_ratio": float(ratio),
        "pulse_n": int(len(pairs)),
        "pulse_resid_ms": resid_sd,
        "pulse_source": source,
    }


def resolve_ephys_start_sample(
    ephys: np.ndarray,
    override: int | None = None,
    stored: int | None = None,
    fs: int = 5000,
) -> tuple[int, str]:
    """Pick the ephys-ON sample for a sync block: override > stored > detector.

    Returns ``(sample, source)`` with source ``"override"``, ``"stored"`` or
    ``"detector"``. ``stored`` is the value persisted in ``sync_info.yaml``
    (``sync_blocks[<b>].ephys_start_sample``) by a previous run; reusing it
    keeps ``redo=True`` reproducible for blocks where the auto-detector fails
    and a manual override was supplied once.
    """
    if override is not None:
        return int(override), "override"
    if stored is not None:
        return int(stored), "stored"
    return detect_ephys_start_sample(ephys, fs=fs), "detector"


def ephys_edge_near(
    signal: np.ndarray, sample: int, tol: int = 3, win: int = 20
) -> bool:
    """True if the ephys sync line switches ON (steps up) within ``sample ± tol``.

    Compares the ``win`` samples just before ``sample - tol`` with the ``win``
    samples just after ``sample + tol``: the post level must exceed 3x the
    largest pre magnitude (the rule :func:`detect_ephys_start_sample` applies
    globally) and the step must be > 5x the larger of the two windows' standard
    deviations. Works for a 3.3 V TTL as well as for the SYNC file's ephys
    copy, whose "on" step is often only ~0.1 V on a quiet baseline.
    """
    signal = np.asarray(signal)
    if not 0 <= sample < signal.size:
        return False
    pre = signal[max(0, sample - tol - win) : max(0, sample - tol)]
    post = signal[sample + tol + 1 : sample + tol + 1 + win]
    if pre.size < 5 or post.size < 5:
        return False
    pre_level = float(np.median(pre))
    post_level = float(np.median(post))
    noise = max(float(np.std(pre)), float(np.std(post)), 1e-9)
    return post_level - pre_level > 5 * noise and post_level > 3 * float(
        np.max(np.abs(pre))
    )


def detect_ephys_start_sample(ephys: np.ndarray, fs: int = 5000) -> int:
    """Return the first SYNC sample where the ephys sync line switches on.

    The threshold is ``3 x max(ephys[:2 s])``, i.e. the line is assumed to be
    quiet for the first two seconds of the SYNC file.

    Raises
    ------
    ValueError
        If no sample exceeds the threshold — typically because the ephys line
        was already high when the SYNC recording started. Pass an explicit
        ``ephys_start_sample`` to :func:`update_sync_info` in that case.
    """
    thresh = np.max(ephys[: int(fs * 2)]) * 3
    samples = np.where(ephys > thresh)[0]
    if samples.size == 0:
        raise ValueError(
            "ephys start not detected: no SYNC sample exceeds 3 x max(first 2 s) "
            f"(= {thresh:.3f}); the ephys line was probably already high at the "
            "start of the SYNC file. Pass ephys_start_sample=... explicitly."
        )
    return int(samples[0])


def get_acq_sync_block(subject, exp, loc, acq):
    si = sync_info()
    key = f"{loc}--{acq}"
    return si[subject][exp]["acquisitions"][key]["sync_block"]


def get_all_exp_sync_blocks(subject, exp):
    si = sync_info()
    return list(si[subject][exp]["sync_blocks"].keys())


def get_ephys_coverage(subject, exp):
    coverages = {}

    sb, sp = get_all_sync_paths(subject, exp)
    block_paths = wis.peri.ephys.get_block_paths(subject, exp)
    for block in sb:
        path = block_paths[block - 1]
        td = epy.tdt.io.load_tdt_block(path, 0, 1)
        td_start = pd.Timestamp(td.info["start_date"])
        td_end = pd.Timestamp(td.info["stop_date"])
        coverages[block] = (td_start, td_end)
    return coverages


def assign_acqs_to_sync_blocks(subject, exp, acqs=None, coverages=None):
    assignments = {}
    if coverages is None:
        coverages = get_ephys_coverage(subject, exp)
    if acqs is None:
        acqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for a in acqs:
        loc, acq = a.split("--")
        est_start = _est_start_time_from_file_name(subject, exp, loc, acq)
        assert est_start is not None, f"Estimated start time could not be found for {a}"
        for block in coverages.keys():
            if coverages[block][0] <= est_start <= coverages[block][1]:
                assignments[a] = block
                break
    return assignments


def load_all_sync_blocks(subject, exp):
    scopes = {}
    ephyses = {}
    sb, sp = get_all_sync_paths(subject, exp)
    for block in sb:
        scopes[block], ephyses[block] = load_sync_block(subject, exp, block)
    return scopes, ephyses


def match_scope_up_window(
    scope,
    ephys,
    dat_start,
    ephys_start_ts,
    fs=5000,
    ephys_start_sample=None,
    clock_ratio=None,
):
    """Pure offset-matcher: given a SYNC file's scope and ephys channels, a
    microscope acquisition wall-clock start, and the TDT block's wall-clock
    start, return the best-matching ``ephys_offset`` in seconds.

    Extracted from ``compute_ephys_offset`` so it can be called per-epoch
    (with a different ``dat_start`` for each epoch) as well as per-acq.

    Candidate acquisition starts are the acquisition-signature onsets from
    :func:`slap2_py.utils.drec.generate_acq_segment_df` (trigger duty >= 0.9
    per second, sustained >= 10 s, onset refined to the sample); the one whose
    rise-anchored offset is closest to the wall-clock gap
    ``|dat_start - ephys_start_ts|`` wins. Scope-UP windows from
    :func:`generate_scope_index_df` are used only if no such segment exists.

    Parameters
    ----------
    scope : np.ndarray
        The ``slap2_acquiring_trigger`` channel from the SYNC HDF5 file.
    ephys : np.ndarray
        The ``electrophysiology`` channel from the SYNC HDF5 file.
    dat_start : pd.Timestamp
        Wall-clock start time of the microscope acquisition/epoch.
    ephys_start_ts : pd.Timestamp
        Wall-clock start time of the TDT block (from ``sync_info.yaml``
        ``sync_blocks[<block>].ephys_start``).
    fs : int, optional
        SYNC file sampling rate in Hz, by default 5000.
    ephys_start_sample : int or float, optional
        The vDaq sample index of TDT t=0. If None, auto-detected via
        :func:`detect_ephys_start_sample` (the #Enable rise). Blocks with the
        shared pulse train pass the (fractional) pulse-fit anchor instead.
    clock_ratio : float, optional
        vDaq-seconds per TDT-second for this block. None uses the fleet
        constant ``defs.VDAQ_TDT_CLOCK_RATIO``; blocks with the shared pulse
        train pass their own measured ratio.

    Returns
    -------
    float
        ``ephys_offset`` in TDT-clock seconds — i.e. seconds from the TDT
        block's start to this acquisition/epoch's start. The raw vDaq-sample
        measurement is divided by ``clock_ratio`` (the ~-4 ppm clock-rate
        difference between the recorders), so the returned value is directly
        comparable to TDT-native time axes (EEG stores, pupil frame times).
    """
    if clock_ratio is None:
        clock_ratio = DEFS.VDAQ_TDT_CLOCK_RATIO
    # Candidate acquisition starts. The trigger line is high ~99 % of the time
    # while acquiring but toggles slowly during preview scanning, and the
    # hysteresis scope-UP detector merges a preview episode that runs straight
    # into an acquisition into one window whose start is the *preview* start
    # (this produced offsets wrong by 158-2312 s on three acqs, 2026-08-31).
    # So match on acquisition-signature onsets; fall back to scope-UP windows
    # only when no such segment exists.
    cands = spy.utils.drec.generate_acq_segment_df(scope, fs=fs)
    kind = "acquisition segment"
    if cands.empty:
        cands = spy.utils.drec.generate_scope_index_df(scope, fs=fs)
        kind = "scope-UP window (no acquisition-duty segment found)"
    if cands.empty:
        raise ValueError(
            "no acquisition segment or scope-UP window found in the SYNC scope channel"
        )
    if ephys_start_sample is None:
        ephys_start_sample = detect_ephys_start_sample(ephys, fs=fs)

    starts = cands["start_idx"].to_numpy().astype(int)
    diff_samples = starts - float(ephys_start_sample)
    est_start = pd.Timestamp(dat_start)
    ephys_start = pd.Timestamp(ephys_start_ts)
    wall_gap_abs = np.abs((ephys_start - est_start).total_seconds())
    differential = np.abs(diff_samples / fs - wall_gap_abs)
    i = int(np.argmin(differential))
    print(
        f"matched {kind} {i + 1}/{len(starts)} at SYNC sample {starts[i]}: "
        f"offset {diff_samples[i] / fs:.4f} s (vDaq), wall-clock gap "
        f"{wall_gap_abs:.1f} s, mismatch {differential[i]:.1f} s"
    )
    if differential[i] > MATCH_WARN_S:
        print(
            f"WARNING: best candidate is {differential[i]:.1f} s from the wall-clock "
            "gap — check the DAT timestamp / TDT block start / SYNC scope channel"
        )
    return diff_samples[i] / fs / clock_ratio


def compute_ephys_offset(
    subject,
    exp,
    loc,
    acq,
    si,
    scope=None,
    ephys=None,
    fs=5000,
    ephys_start_sample=None,
    clock_ratio=None,
):
    """Compute ephys_offset for an acq by looking up sync-block + dat_start
    in sync_info, loading the SYNC file if needed, and running the matcher.

    For multi-epoch acqs this returns epoch 1's offset (since ``dat_start``
    in sync_info is defined as epoch 1's filename timestamp).
    """
    acq_id = f"{loc}--{acq}"
    block = si[subject][exp]["acquisitions"][acq_id]["sync_block"]
    if scope is None and ephys is None:
        scope, ephys = load_sync_block(subject, exp, block)
    dat_start = pd.Timestamp(si[subject][exp]["acquisitions"][acq_id]["dat_start"])
    ephys_start_ts = pd.Timestamp(si[subject][exp]["sync_blocks"][block]["ephys_start"])
    return match_scope_up_window(
        scope,
        ephys,
        dat_start,
        ephys_start_ts,
        fs=fs,
        ephys_start_sample=ephys_start_sample,
        clock_ratio=clock_ratio,
    )


def find_corrupted_sync_offsets(subject, exp, loc, acq, si):
    start_time = pd.Timestamp(
        si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["dat_start"]
    )
    # find all the candidate start times of non-corrupt sync blocks
    starts = []
    acq_ids = []
    for acq_id in si[subject][exp]["acquisitions"].keys():
        if (
            si[subject][exp]["sync_blocks"][
                si[subject][exp]["acquisitions"][acq_id]["sync_block"]
            ]["corrupt"]
            is False
        ):
            starts.append(
                pd.Timestamp(si[subject][exp]["acquisitions"][acq_id]["dat_start"])
            )
            acq_ids.append(acq_id)

    if len(starts) == 0:
        print(f"No valid start times found for {subject} {exp} {loc} {acq}")
        return None

    # find the start time (and corresponding acq_id) that is closest to the acquisition start time
    closest_start = min(starts, key=lambda x: abs(x - start_time))
    closest_acq_id = acq_ids[starts.index(closest_start)]

    # get the ephys offset for the closest start time
    closest_ephys_offset = si[subject][exp]["acquisitions"][closest_acq_id][
        "ephys_offset"
    ]

    # find the pure difference in time between the closest start time and the acquisition start time
    if closest_start > start_time:
        pure_diff = (closest_start - start_time).total_seconds()
    else:
        pure_diff = (start_time - closest_start).total_seconds()

    # put that closest start time into the ephys time base, which should be precisely synchronized
    good_ephys_start = pd.Timestamp(
        si[subject][exp]["sync_blocks"][
            si[subject][exp]["acquisitions"][closest_acq_id]["sync_block"]
        ]["ephys_start"]
    )
    close_start_in_ephys = good_ephys_start + pd.Timedelta(seconds=closest_ephys_offset)

    # put the corrupt file in the same ephys time base
    if closest_start > start_time:
        corrupt_time_start_in_ephys = close_start_in_ephys - pd.Timedelta(
            seconds=pure_diff
        )
    else:
        corrupt_time_start_in_ephys = close_start_in_ephys + pd.Timedelta(
            seconds=pure_diff
        )

    # get the corrupt block's ephys start time
    corrupt_block_start = pd.Timestamp(
        si[subject][exp]["sync_blocks"][
            si[subject][exp]["acquisitions"][f"{loc}--{acq}"]["sync_block"]
        ]["ephys_start"]
    )

    # find the difference in time between the corrupt start time and the corrupt time start in ephys (everything should now be synced into the ephys time base)
    diff_in_time = (corrupt_time_start_in_ephys - corrupt_block_start).total_seconds()
    return diff_in_time


def _list_epoch_start_times(subject, exp, loc, acq):
    """Return the per-epoch start times parsed from the DMD1 first-cycle
    .dat filenames in this acq's directory.

    Returns a list of ``(filename, timestamp)`` tuples, sorted ascending by
    ``timestamp``. Single-epoch acqs produce a list of length 1; multi-epoch
    acqs (paused and restarted during scanning, typically re-assembled by the
    preprocessing pipeline's epoch functionality) produce length > 1. Returns
    an empty list if no matching files are found.

    Sorting is by the parsed timestamp, NOT by filename string — the
    ``acq-N_`` prefix is not guaranteed to be monotonic with wall clock (and
    lexicographic sort breaks at ``acq-10`` anyway).
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    if not os.path.isdir(acq_dir):
        return []
    found = []
    for f in os.listdir(acq_dir):
        if "_DMD1-CYCLE-000000.dat" in f:
            ts_str = f.split("_DMD1-CYCLE-000000.dat")[0][-15:]
            try:
                ts = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            except ValueError:
                continue
            found.append((f, ts))
    found.sort(key=lambda x: x[1])
    return found


def _est_start_time_from_file_name(subject, exp, loc, acq, verbose=True):
    """Parse the acquisition-start wall-clock time from the DMD1 first-cycle
    .dat filename.

    The SLAP2 acquisition software stamps the wall-clock start time into every
    raw .dat filename as ``_YYYYMMDD_HHMMSS_`` (1-second resolution). This is
    precise enough to anchor scope-up-window matching in the SYNC file,
    avoiding the need for a MATLAB-generated ``datStartTimes1.txt`` file.

    For multi-epoch acqs (multiple first-cycle .dat files, one per epoch),
    returns the earliest timestamp — i.e. epoch 1.
    """
    epochs = _list_epoch_start_times(subject, exp, loc, acq)
    if not epochs:
        if verbose:
            acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
            print(
                f"No file with _DMD1-CYCLE-000000.dat found in {acq_dir}, "
                f"returning None for estimated start time"
            )
        return None
    return epochs[0][1]


def _has_first_cycle_dat_file(subject, exp, loc, acq):
    """Return True if at least one DMD1 first-cycle .dat file exists for
    this acq.

    Used as the precondition check for sync-info population, replacing the
    previous ``datStartTimes1.txt`` existence check.
    """
    acq_dir = f"{DEFS.data_root}/{subject}/{exp}/{loc}/{acq}"
    if not os.path.isdir(acq_dir):
        return False
    for f in os.listdir(acq_dir):
        if "_DMD1-CYCLE-000000.dat" in f:
            return True
    return False


def update_sync_info(subject, exp, redo=False, ephys_start_sample=None):
    """update the sync info file for a given experiment.

    Parameters
    ----------
    subject : str
        subject name
    exp : str
        experiment name
    redo : bool, optional
        Recompute every sync block and acquisition entry for this experiment
        (the ephys_offset matcher is re-run). Existing ``dat_start`` values and
        each block's persisted ``ephys_start_sample`` are preserved, so a redo
        is reproducible even for blocks that once needed a manual override.
        Delete an entry from ``sync_info.yaml`` to force re-derivation.
    ephys_start_sample : dict, optional
        ``{"loc_X--acq_N": <sample>}`` manual ephys-ON sample overrides. An
        override applies to the whole sync block the acquisition belongs to and
        is persisted as ``sync_blocks[<b>].ephys_start_sample``.
    """
    si = sync_info()
    if si is None:
        si = {}
    if subject not in si:
        si[subject] = {}
    if exp not in si[subject]:
        si[subject][exp] = {}
        si[subject][exp]["sync_blocks"] = {}
        si[subject][exp]["acquisitions"] = {}

    # Acquisitions to (re)fit: new ones, or every one when redo=True. (Before
    # 2026-08-31 redo only rebuilt the sync_blocks metadata and silently left
    # every existing ephys_offset untouched.)
    acqs = []
    acqs_all = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    for acq_id in acqs_all:
        if redo or acq_id not in si[subject][exp]["acquisitions"].keys():
            loc, acq = acq_id.split("--")
            if _has_first_cycle_dat_file(subject, exp, loc, acq):
                acqs.append(acq_id)

    # Fast backfill of cheap-to-compute fields on existing entries. Only add
    # fields here that are (a) cheap — no SYNC loads / hysteresis / offset
    # math — and (b) a pure function of current raw-data state. For anything
    # expensive, or anything that re-touches the ephys_offset matcher, use
    # redo=True. Skips any entry whose raw files have disappeared (leaves
    # n_epochs absent rather than writing the non-sensical value 0, since an
    # acq by definition has >= 1 epoch).
    backfilled = False
    for acq_id, entry in si[subject][exp].get("acquisitions", {}).items():
        if acq_id in acqs:
            continue  # about to be freshly populated below
        loc, acq = acq_id.split("--")
        if "n_epochs" not in entry and _has_first_cycle_dat_file(
            subject, exp, loc, acq
        ):
            entry["n_epochs"] = len(_list_epoch_start_times(subject, exp, loc, acq))
            backfilled = True

    if len(acqs) == 0 and not redo:
        if backfilled:
            with open(f"{DEFS.anmat_root}/sync_info.yaml", "w") as f:
                yaml.dump(si, f)
            print(
                f"Backfilled missing cheap fields for {subject} {exp} "
                f"(no new acqs to fully process)."
            )
        else:
            print(
                f"All acquisitions already have sync info for {subject} {exp}. "
                f"Use redo=True to recompute."
            )
        return

    sb, sp = get_all_sync_paths(subject, exp)
    coverage = get_ephys_coverage(subject, exp)
    scope, ephys = load_all_sync_blocks(subject, exp)
    for block in sb:
        if block in si[subject][exp]["sync_blocks"].keys() and not redo:
            continue
        prev_block = si[subject][exp]["sync_blocks"].get(block, {})
        si[subject][exp]["sync_blocks"][block] = {}
        if prev_block.get("ephys_start_sample") is not None:
            si[subject][exp]["sync_blocks"][block]["ephys_start_sample"] = int(
                prev_block["ephys_start_sample"]
            )
        si[subject][exp]["sync_blocks"][block]["ephys_start"] = str(coverage[block][0])
        si[subject][exp]["sync_blocks"][block]["ephys_end"] = str(coverage[block][1])
        if type(scope[block]) is str and type(ephys[block]) is str:
            if scope[block] == "Corrupt" and ephys[block] == "Corrupt":
                si[subject][exp]["sync_blocks"][block]["corrupt"] = True
            else:
                si[subject][exp]["sync_blocks"][block]["corrupt"] = (
                    "Unknown - check sync files"
                )
        else:
            si[subject][exp]["sync_blocks"][block]["corrupt"] = False
            si[subject][exp]["sync_blocks"][block]["sync_source"] = sync_source(
                subject, exp, block
            )

    # Now we update the specific acquisitions with block assignments and ephys offsets
    # acqs = wis.meta.get.unique_acquisitions_per_experiment(subject, exp)
    assignments = assign_acqs_to_sync_blocks(subject, exp, acqs, coverage)

    # Per-block pulse calibrations, computed lazily (once per block per run).
    pulse_cal_cache: dict[int, dict | None] = {}

    # Manual overrides are per acquisition in the call signature but physically
    # per sync block (one SYNC file per block), so apply each to its block.
    block_overrides: dict[int, int] = {}
    for acq_id, sample in (ephys_start_sample or {}).items():
        if acq_id in assignments:
            block_overrides[assignments[acq_id]] = int(sample)

    for acq_id in acqs:
        loc, acq = acq_id.split("--")
        epochs = _list_epoch_start_times(subject, exp, loc, acq)
        assert epochs, (
            f"No DMD1 first-cycle .dat file found for {subject}/{exp}/{loc}/{acq} "
            f"despite _has_first_cycle_dat_file passing — race condition?"
        )
        prev_acq = si[subject][exp]["acquisitions"].get(acq_id, {})
        # Keep a previously stored dat_start (some carry sub-second precision
        # from older sources; the filename timestamp is 1-s resolution).
        dat_start = prev_acq.get("dat_start") or str(epochs[0][1])  # epoch 1
        n_epochs = len(epochs)
        block = assignments[acq_id]
        si[subject][exp]["acquisitions"][acq_id] = {}
        si[subject][exp]["acquisitions"][acq_id]["sync_block"] = block
        si[subject][exp]["acquisitions"][acq_id]["dat_start"] = str(dat_start)
        si[subject][exp]["acquisitions"][acq_id]["n_epochs"] = n_epochs
        if si[subject][exp]["sync_blocks"][block]["corrupt"] is False:
            block_entry = si[subject][exp]["sync_blocks"][block]
            start_sample_to_use, start_source = resolve_ephys_start_sample(
                ephys[block],
                override=block_overrides.get(block),
                stored=block_entry.get("ephys_start_sample"),
            )
            block_entry["ephys_start_sample"] = int(start_sample_to_use)
            if start_source != "stored" or "ephys_start_source" not in block_entry:
                block_entry["ephys_start_source"] = (
                    block_entry.get("ephys_start_source", "stored")
                    if start_source == "stored"
                    else start_source
                )
            # Precise per-block calibration from the shared RZ2 pulse train
            # ('rz_pulse' vDaq channel + 'p_in' TDT store), when recorded.
            # Computed once per block per run; falls back to the enable-rise
            # anchor + fleet-constant ratio when absent or failing gates.
            if block not in pulse_cal_cache:
                pulse_vdaq = load_pulse_channel(subject, exp, block)
                if pulse_vdaq is None:
                    pulse_cal_cache[block] = None
                else:
                    block_paths = wis.peri.ephys.get_block_paths(subject, exp)
                    pulse_cal_cache[block] = calibrate_block_from_pulses(
                        pulse_vdaq,
                        block_paths[block - 1],
                        coarse_anchor_idx=float(start_sample_to_use),
                    )
            cal = pulse_cal_cache[block]
            if cal is not None:
                block_entry.update(cal)
                anchor = cal["pulse_anchor_idx"]
                ratio = cal["pulse_clock_ratio"]
                method = "rz_pulse"
            else:
                anchor = float(start_sample_to_use)
                ratio = DEFS.VDAQ_TDT_CLOCK_RATIO
                method = "enable_rise"
            offset = float(
                compute_ephys_offset(
                    subject,
                    exp,
                    loc,
                    acq,
                    si,
                    scope[block],
                    ephys[block],
                    ephys_start_sample=anchor,
                    clock_ratio=ratio,
                )
            )
            # The offset is stored in TDT-clock seconds (raw vDaq measurement
            # / clock_ratio_applied). Keep the raw measurement alongside so
            # the ratio can be re-applied by pure arithmetic if re-measured.
            si[subject][exp]["acquisitions"][acq_id]["ephys_offset_uncorrected"] = (
                offset * ratio
            )
        else:
            offset = find_corrupted_sync_offsets(subject, exp, loc, acq, si)
            offset = float(offset) if offset is not None else None
            ratio = DEFS.VDAQ_TDT_CLOCK_RATIO
            method = "corrupt_fallback"
        si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = offset
        si[subject][exp]["acquisitions"][acq_id]["clock_ratio_applied"] = ratio
        si[subject][exp]["acquisitions"][acq_id]["offset_method"] = method

    # One final retry for any acquisitions still unresolved (offset is None),
    # which can happen when a corrupt acq was processed before any non-corrupt
    # reference was populated. By this pass, references are in place, so the
    # lookup usually succeeds. Anything still None after this means no
    # non-corrupt sync block exists anywhere in the experiment to anchor against.
    for acq_id in si[subject][exp]["acquisitions"].keys():
        loc, acq = acq_id.split("--")
        if si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] is None:
            ephys_offset = find_corrupted_sync_offsets(subject, exp, loc, acq, si)
            si[subject][exp]["acquisitions"][acq_id]["ephys_offset"] = (
                float(ephys_offset) if ephys_offset is not None else None
            )
            si[subject][exp]["acquisitions"][acq_id]["clock_ratio_applied"] = (
                DEFS.VDAQ_TDT_CLOCK_RATIO
            )

    # write the updated sync info to the yaml file
    with open(f"{DEFS.anmat_root}/sync_info.yaml", "w") as f:
        yaml.dump(si, f)
        f.close()
    return


# ==================================================================================
# Backfilling ephys_start_sample for blocks whose entries predate its persistence
# ==================================================================================
def _acq_onset_candidates(scope: np.ndarray, fs: int = 5000) -> np.ndarray:
    """Acquisition-signature onsets of the trigger channel (scope-UP windows as
    a fallback) — the same candidates the matcher uses."""
    seg = spy.utils.drec.generate_acq_segment_df(scope, fs=fs)
    starts = seg["start_idx"].to_numpy().astype(int)
    if starts.size == 0:
        win = spy.utils.drec.generate_scope_index_df(scope, fs=fs)
        starts = win["start_idx"].to_numpy().astype(int)
    return starts


def _vdaq_offset_s(acq_entry: dict) -> float | None:
    """The raw vDaq-seconds offset an acquisition entry encodes."""
    u = acq_entry.get("ephys_offset_uncorrected")
    if u is not None:
        return float(u)
    off = acq_entry.get("ephys_offset")
    if off is None:
        return None
    return float(off) * float(acq_entry.get("clock_ratio_applied", 1.0))


def back_calculate_ephys_start(
    scope: np.ndarray,
    ephys: np.ndarray,
    acq_entry: dict,
    ephys_start_ts,
    fs: int = 5000,
) -> dict:
    """The ephys-ON sample that one acquisition's stored offset encodes.

    ``ephys_offset_uncorrected = (acq_onset − ephys_start_sample) / fs``, so
    ``ephys_start_sample = acq_onset − round(offset · fs)`` with ``acq_onset``
    re-derived from the trigger channel (candidate nearest the wall-clock gap,
    anchored on the auto-detected ephys start when it fires — exactly the
    matcher's selection). Whatever value was used when the offset was computed
    (auto-detected or a manual override) falls out of this arithmetic.

    Returns a dict with ``value`` (int or None), ``onset``, ``detector`` (int
    or None), ``edge`` (bool: an ephys rise sits at ``value``), ``wall_gap_s``
    and ``reason`` (why ``value`` is None).
    """
    scope = np.asarray(scope)
    ephys = np.asarray(ephys)
    out = {"value": None, "onset": None, "detector": None, "edge": False, "reason": ""}
    try:
        out["detector"] = detect_ephys_start_sample(ephys, fs=fs)
    except ValueError:
        out["detector"] = None
    off = _vdaq_offset_s(acq_entry)
    if off is None:
        out["reason"] = "no ephys_offset stored"
        return out
    cands = _acq_onset_candidates(scope, fs=fs)
    if cands.size == 0:
        out["reason"] = "no acquisition onset / scope-UP window in the trigger"
        return out
    gap = (
        pd.Timestamp(acq_entry["dat_start"]) - pd.Timestamp(ephys_start_ts)
    ).total_seconds()
    out["wall_gap_s"] = float(gap)
    anchor = out["detector"] if out["detector"] is not None else 0
    i = int(np.argmin(np.abs((cands - anchor) / fs - gap)))
    out["onset"] = int(cands[i])
    out["value"] = int(cands[i]) - int(round(off * fs))
    out["edge"] = ephys_edge_near(ephys, out["value"])
    if not out["edge"]:
        out["reason"] = f"no ephys edge at back-calculated sample {out['value']}"
    return out


def backfill_ephys_start_samples(dry_run: bool = True, verbose: bool = True):
    """Fill ``sync_blocks[<b>].ephys_start_sample`` for blocks that lack it.

    For every non-corrupt block with at least one acquisition, back-calculate
    the ephys-ON sample from each acquisition's stored offset
    (:func:`back_calculate_ephys_start`). A block is written only when every
    acquisition agrees to within one sample AND an ephys rise sits at that
    sample. Stored values are never modified (mismatches are reported).
    Blocks without acquisitions are left untouched — nothing uses their ephys
    start and no override could ever have been supplied for them.

    Provenance goes to ``ephys_start_source``: ``"backfill (= detector)"`` when
    the auto-detector agrees, ``"backfill (override; detector=N|failed)"``
    otherwise.

    Parameters
    ----------
    dry_run
        When True (default) nothing is written; the returned table shows what
        would happen. When False, a timestamped backup of ``sync_info.yaml`` is
        written first and only rows with status ``write`` are applied.

    Returns
    -------
    polars.DataFrame
        One row per sync block: ``subject, exp, block, n_acqs, status, value,
        detector, source, stored, values, edge_ok, note``. ``status`` is one of
        ``written`` / ``write`` (dry run) / ``already_stored`` / ``manual_review``
        / ``no_acquisitions`` / ``corrupt``.
    """
    import polars as pl

    si = sync_info()
    rows: list[dict] = []
    for subject in si:
        for exp in si[subject]:
            d = si[subject][exp]
            for block, binfo in d["sync_blocks"].items():
                acqs = {
                    la: a
                    for la, a in d["acquisitions"].items()
                    if a.get("sync_block") == block
                }
                row = {
                    "subject": subject,
                    "exp": exp,
                    "block": int(block),
                    "n_acqs": len(acqs),
                    "status": "",
                    "value": None,
                    "detector": None,
                    "source": None,
                    "stored": binfo.get("ephys_start_sample"),
                    "values": "",
                    "edge_ok": None,
                    "note": "",
                }
                if not acqs:
                    row["status"] = "no_acquisitions"
                    rows.append(row)
                    continue
                scope, ephys = load_sync_block(subject, exp, int(block))
                if isinstance(scope, str):
                    row["status"] = "corrupt"
                    rows.append(row)
                    continue
                results = {
                    la: back_calculate_ephys_start(
                        scope, ephys, a, binfo["ephys_start"]
                    )
                    for la, a in acqs.items()
                }
                vals = [r["value"] for r in results.values()]
                row["values"] = ", ".join(
                    f"{la}: {r['value']}{'' if r['edge'] else ' (no edge)'}"
                    for la, r in results.items()
                )
                row["detector"] = next(iter(results.values()))["detector"]
                ok = [v for v in vals if v is not None]
                if len(ok) != len(vals) or max(ok) - min(ok) > 1:
                    row["status"] = "manual_review"
                    row["note"] = (
                        "; ".join(r["reason"] for r in results.values() if r["reason"])
                        or "acquisitions disagree on the ephys start sample"
                    )
                    rows.append(row)
                    continue
                value = int(round(float(np.median(ok))))
                row["value"] = value
                row["edge_ok"] = all(r["edge"] for r in results.values())
                if not row["edge_ok"]:
                    row["status"] = "manual_review"
                    row["note"] = f"no ephys edge at back-calculated sample {value}"
                    rows.append(row)
                    continue
                det = row["detector"]
                if det is not None and abs(det - value) <= 1:
                    row["source"] = "backfill (= detector)"
                else:
                    row["source"] = (
                        f"backfill (override; detector={det})"
                        if det is not None
                        else "backfill (override; detector failed)"
                    )
                if row["stored"] is not None:
                    row["status"] = "already_stored"
                    row["note"] = (
                        "matches"
                        if abs(int(row["stored"]) - value) <= 1
                        else f"MISMATCH: stored {row['stored']} vs "
                        f"back-calculated {value}"
                    )
                else:
                    row["status"] = "write"
                rows.append(row)
                if verbose:
                    print(
                        f"  {subject}/{exp} sync_block-{block}: {row['status']} "
                        f"value={value} detector={det} ({row['source']}) {row['note']}"
                    )

    if not dry_run:
        to_write = [r for r in rows if r["status"] == "write"]
        if to_write:
            import shutil
            from datetime import datetime

            path = f"{DEFS.anmat_root}/sync_info.yaml"
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.copy2(path, f"{path}.bak-{stamp}-pre-ephys-start-backfill")
            for r in to_write:
                b = si[r["subject"]][r["exp"]]["sync_blocks"][r["block"]]
                b["ephys_start_sample"] = int(r["value"])
                b["ephys_start_source"] = r["source"]
                r["status"] = "written"
            with open(path, "w") as f:
                yaml.dump(si, f)
            if verbose:
                print(
                    f"wrote {len(to_write)} block(s); backup at "
                    f"{path}.bak-{stamp}-pre-ephys-start-backfill"
                )
    return pl.DataFrame(
        rows,
        schema={
            "subject": pl.Utf8,
            "exp": pl.Utf8,
            "block": pl.Int64,
            "n_acqs": pl.Int64,
            "status": pl.Utf8,
            "value": pl.Int64,
            "detector": pl.Int64,
            "source": pl.Utf8,
            "stored": pl.Int64,
            "values": pl.Utf8,
            "edge_ok": pl.Boolean,
            "note": pl.Utf8,
        },
    )
