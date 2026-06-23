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
# -------------------------------------------------------------

annotation_root = os.path.join(anmat_root, "annotation_materials")

exsum_mirror_root = os.path.join(anmat_root, "ExSum_mirrors")
plots_root = os.path.join(anmat_root, "plots")


dlc_root: str = "/data/slap_analysis/slap_mi_in_the_pupil"
dlc_proj_root: str = os.path.join(dlc_root, "dlc_slap_pupil-KD-2025-09-23")
dlc_env_py: str = os.path.join(dlc_root, ".venv", "bin", "python")
dlc_config_path: str = os.path.join(dlc_proj_root, "config.yaml")

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
