from __future__ import annotations

# -------- Defaults (edit these) --------
data_root: str = "/Volumes/slap_mi/slap_mi/data"
anmat_root: str = "/Volumes/slap_mi/slap_mi/analysis_materials"

dlc_root: str = "/Volumes/npx_nfs/slap/slap_mi_in_the_pupil"
dlc_proj_root: str = f"{dlc_root}/dlc_slap_pupil-KD-2025-09-23"
dlc_env_py: str = f"{dlc_root}/.venv/bin/python"
dlc_env_py: str = f"{dlc_root}/.venv/bin/python"
dlc_config_path: str = f"{dlc_proj_root}/config.yaml"

store_chans = {}
store_chans["EEGr"] = [1, 2]
store_chans["EEG_"] = [1]
store_chans["loal"] = [1]
store_chans["Wav1"] = [1]
