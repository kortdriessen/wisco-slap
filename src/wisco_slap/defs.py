from __future__ import annotations

import os

# -------- Defaults (edit these) --------
# raw data root, on the NAS
data_root: str = (
    "/run/user/1329238735/gvfs/smb-share:server=tononi-nas,share=slap_mi/slap_mi/data"
)
ssh_data_root = "slapmi@tononi-nas:/mnt/pool1/tononi_volume10/slap_mi/data"
# root of the analysis directory, local
anmat_root: str = "/data/slap_analysis/analysis_materials"
# local mirror of raw acquisition data (registered/downsampled TIFFs, .dat, etc.)
raw_mirror_root: str = "/data/raw_mirror"
# Source of truth for ExperimentSummary files (ExSum mirroring + prepro
# status). Normally this is data_root (the NAS). During the 2026-08
# channel-fix reprocessing reintegration it points at the local raw_mirror so
# the mirror pipeline pulls newly reprocessed ExSums without touching the NAS
# (whose copies are stale until it is comprehensively updated later). Only
# ExperimentSummary lookups use this; every other raw stream (SYNC h5, ephys,
# pupil, .dat files) still reads from data_root.
exsum_source_root: str = raw_mirror_root
# -------------------------------------------------------------

annotation_root = os.path.join(anmat_root, "annotation_materials")
# Recovered copies of corrupt raw SYNC HDF5 files (see
# slap2_py.utils.sync_recovery); wis.meta.sync.load_sync_block falls back to
# <sync_recovery_root>/<subject>/<exp>/SYNC_0000N.h5 when the raw file is corrupt.
sync_recovery_root = os.path.join(anmat_root, "sync_recovery")

exsum_mirror_root = os.path.join(anmat_root, "ExSum_mirrors")
plots_root = os.path.join(anmat_root, "plots")


dlc_root: str = "/data/slap_analysis/slap_mi_in_the_pupil"
dlc_proj_root: str = os.path.join(dlc_root, "dlc_slap_pupil-KD-2025-09-23")
dlc_env_py: str = os.path.join(dlc_root, ".venv", "bin", "python")
dlc_config_path: str = os.path.join(dlc_proj_root, "config.yaml")

# CASCADE (CascadeTorch) spike-inference model location, used by
# wisco_slap.scope.cascade. Env-overridable so the home-dir default isn't
# baked into pipeline logic.
cascade_root: str = os.environ.get("CASCADE_ROOT", "/home/driessen2/CascadeTorch")
cascade_model_folder: str = os.environ.get(
    "CASCADE_MODEL_FOLDER", os.path.join(cascade_root, "Pretrained_models")
)

# Microscope-vs-TDT clock-rate ratio: how many "microscope seconds" (vDaq
# SYNC-file samples / 5000, or equivalently SLAP2-FPGA seconds — the two were
# shown to share a timebase to +/-0.1 ppm in within_acq_shear_measurements.ipynb)
# elapse per true TDT-clock second. The microscope clock runs ~4 ppm slow.
# Any interval measured on a microscope clock must be DIVIDED by this ratio to
# express it in TDT seconds. Applied in three places: ephys_offset
# (sync.py match_scope_up_window), per-epoch FPGA deltas (epoch.py), and the
# scopex/shaft_ca trace time axes at load time (get/_get_scopex.py,
# get/_get_shaft_ca.py).
# Value locked 2026-08-31 by the 104-min shared-TTL recording test-block-2
# (12,292 pulses: -3.970 +/- ~0.05 ppm; linear to +/-0.4 ms over the block;
# see slap_mi_2_sleep/synchronization_validation/validate_on_test_blocks.ipynb),
# consistent with test-block-1 (-4.39 +/- 0.51) and the 46-block fleet fit
# (-6.5 +/- 3.0). Uncorrected values are preserved in sync_info.yaml /
# epoch_info.yaml, so re-application after any future re-measurement is pure
# arithmetic (see synchronization_validation/apply_clock_correction.py).
VDAQ_TDT_CLOCK_RATIO: float = 0.99999603

store_chans = {}
store_chans["EEGr"] = [1]
store_chans["EEG_"] = [1]
store_chans["loal"] = [1]
store_chans["Wav1"] = [1]

state_colors = {
    "Wake": "#62e273",
    "Brief-Arousal": "#90EE90",
    "Transition-to-NREM": "#87CEEB",
    "Transition-to-Wake": "#98FB98",
    "NREM": "#6495ED",
    "Transition-to-REM": "#DDA0DD",
    "REM": "#FF00FF",
    "Transition": "#808080",
    "Art": "#DC143C",
    "Wake-art": "#DC143C",
    "Unsure": "#FFFFFF",
    "Good": "#00FF00",
    # "Wake-Good": "turquoise",
    "Wake-Good": "#62e273",
    "Sort-Exclude": "#FF7F50",
    "unclear": "#FFFF00",
}

color_defs = {
    "whisk": "#0504aa",  # xkcd: royal blue
    "pupil": "#36013f",  # xkcd: deep purple
    "loal": "#411900",
    "glugreen_light": "#0cf000",
    "glugreen": "#2ec700",
    "jred": "#d60b1f",
    "deepgray": "#212121",
}
