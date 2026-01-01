import h5py
import numpy as np
from pathlib import Path
from typing import Any
import h5py
from pathlib import Path
import os as os
import wisco_slap as wis
import wisco_slap.defs as DEFS
import polars as pl


def _decode_matlab_string(data: np.ndarray) -> str | np.ndarray:
    """
    Try to convert typical MATLAB char arrays to Python str.
    Falls back to original array if it doesn't look like text.
    """
    # MATLAB chars often come as 1D/2D uint16/uint8 arrays.
    if not isinstance(data, np.ndarray):
        return data

    if data.dtype.kind in ("u", "i") and data.ndim >= 1:
        # Flatten and drop zeros, then try UTF-8 / UTF-16
        flat = data.ravel()
        # Many MATLAB strings are stored as uint16 code units:
        try:
            # Heuristic: if max value < 256, treat as UTF-8 bytes
            if flat.max(initial=0) < 256:
                return (
                    bytes(flat.astype("uint8"))
                    .decode("utf-8", errors="ignore")
                    .strip("\x00")
                )
            else:
                return (
                    flat.astype("uint16")
                    .tobytes()
                    .decode("utf-16le", errors="ignore")
                    .strip("\x00")
                )
        except Exception:
            return data

    return data


def _h5_to_py(obj: h5py.Group | h5py.Dataset) -> Any:
    """
    Recursively convert an h5py Group/Dataset into Python structures.

    - Groups → dict of {name: value}
    - Datasets → numpy arrays / scalars / strings
    """
    if isinstance(obj, h5py.Group):
        out = {}
        for key, item in obj.items():
            out[key] = _h5_to_py(item)
        return out

    elif isinstance(obj, h5py.Dataset):
        data = obj[()]  # read the whole dataset

        # MATLAB cell arrays / objects often show up as object arrays or references;
        # here we only handle straightforward numeric / char data cleanly.
        if isinstance(data, np.ndarray):
            # Convert 0-d array to Python scalar
            if data.shape == ():
                data = data.item()

            # Try to turn MATLAB char arrays into strings
            matlab_class = obj.attrs.get("MATLAB_class", None)
            if matlab_class is not None:
                try:
                    matlab_class = (
                        matlab_class.decode()
                        if isinstance(matlab_class, (bytes, bytearray))
                        else matlab_class
                    )
                except Exception:
                    pass

            if matlab_class == "char":
                data = _decode_matlab_string(data)

        # If it's a bytes object, decode
        if isinstance(data, (bytes, bytearray)):
            try:
                data = data.decode("utf-8", errors="ignore")
            except Exception:
                pass

        return data

    else:
        # Shouldn't really happen with h5py, but just in case.
        return obj


def load_mat73_to_dict(path: Path, root: str | None = None) -> Any:
    """
    Load a MATLAB 7.3 (HDF5) .mat file into nested Python structures.

    Parameters
    ----------
    path : Path
        Path to the .mat file.
    root : str or None
        If None, convert the whole file.
        If a group name (e.g. 'exptSummary'), only that group is converted.

    Returns
    -------
    Any
        Usually a dict, but can be an array or scalar depending on root.
    """
    with h5py.File(path, "r") as f:
        if root is None:
            return _h5_to_py(f)
        else:
            if root not in f:
                raise KeyError(
                    f"Root group '{root}' not found in file. Available keys: {list(f.keys())}"
                )
            return _h5_to_py(f[root])


def deref_h5(ref, file_path: str | Path):
    """Given an h5py.Reference (e.g. data['meanIM'][0][0]), return the actual array."""
    with h5py.File(file_path, "r") as f:
        return f[ref][()]  # follow the reference and read the dataset


def _h5_to_py_simple(obj: Any) -> Any:
    """Minimal recursive converter: Dataset -> array, Group -> dict."""
    if isinstance(obj, h5py.Dataset):
        return obj[()]  # full array
    if isinstance(obj, h5py.Group):
        return {k: _h5_to_py_simple(v) for k, v in obj.items()}
    return obj


def deref_h5_any(ref: Any, file_path: str | Path) -> Any:
    """
    Given something like data['E'][0][0] (which may be a tuple/ndarray
    containing an HDF5 object reference), open the file and return the
    actual data (array or nested dict).
    """
    file_path = str(file_path)

    # 1. Unwrap containers (np.ndarray, list, tuple) until we get a reference or path
    while isinstance(ref, (np.ndarray, list, tuple)):
        if isinstance(ref, np.ndarray):
            if ref.shape == ():
                ref = ref.item()
            else:
                ref = ref.flat[0]
        else:  # list or tuple
            if len(ref) == 0:
                raise ValueError("Empty reference container")
            ref = ref[0]

    with h5py.File(file_path, "r") as f:
        # 2. Follow reference or path
        if isinstance(ref, h5py.Reference):
            obj = f[ref]
        elif isinstance(ref, (str, bytes)):
            obj = f[ref]
        else:
            raise TypeError(f"Unsupported reference type {type(ref)}: {ref!r}")

        # 3. Convert to Python structures
        return _h5_to_py_simple(obj)


def get_exsum_basics(exsum_path: Path | str) -> dict:
    data = load_mat73_to_dict(exsum_path, root="exptSummary")
    ntrials = data["E"].shape[1]
    fs = int(data["params"]["analyzeHz"][0][0])
    return data, ntrials, fs


def read_full_trial_data_dict(exsum_path):
    data, ntrials, fs = get_exsum_basics(exsum_path)
    ntrials = data["E"].shape[1]

    trial_data = {}
    for dmd in [1, 2]:
        dmd_idx = dmd - 1
        trial_data[dmd] = {}
        for trl in range(ntrials):
            try:
                tdat = deref_h5_any(data["E"][dmd_idx][trl], exsum_path)
                if type(tdat) is dict:
                    trial_data[dmd][trl] = tdat
                elif type(tdat) is np.ndarray:
                    if tdat.shape == (2,):
                        print(f"bad trial, zero-array, {dmd}-{trl}")
                        trial_data[dmd][trl] = "BAD_TRIAL"
                else:
                    print(f"unknown trial type, INVESTIGATE: {dmd}-{trl}")
                    trial_data[dmd][trl] = "BAD_TRIAL"
            except Exception as e:
                print(f"ERROR: {e} ---- trial {dmd}-{trl}")
                print("filling in missing_data")
                trial_data[dmd][trl] = "BAD_TRIAL"
                continue
    fs = int(data["params"]["analyzeHz"][0][0])
    return trial_data, data, fs, ntrials


def create_null_trial_data(clean):
    null_data = {}
    for key in clean.keys():
        if type(clean[key]) is np.ndarray:
            null_data[key] = np.zeros_like(clean[key]) * np.nan
        elif type(clean[key]) is dict:
            null_data[key] = {}
            for subkey in clean[key].keys():
                if type(clean[key][subkey]) is np.ndarray:
                    null_data[key][subkey] = np.zeros_like(clean[key][subkey]) * np.nan
                elif type(clean[key][subkey]) is dict:
                    null_data[key][subkey] = {}
                    for subsubkey in clean[key][subkey].keys():
                        if type(clean[key][subkey][subsubkey]) is np.ndarray:
                            null_data[key][subkey][subsubkey] = (
                                np.zeros_like(clean[key][subkey][subsubkey]) * np.nan
                            )
                        else:
                            print(
                                f"unknown subsubkey type, INVESTIGATE: {key}-{subkey}-{subsubkey}"
                            )
    return null_data


def assert_shape_match(clean, comparison_data):
    for key in clean.keys():
        if "ROI" in key:  # TODO: Fix this immedidately!!! Hack!
            continue
        if type(clean[key]) is np.ndarray:
            assert (
                comparison_data[key].shape == clean[key].shape
            ), f"Shape mismatch for {key}: {comparison_data[key].shape} vs {clean[key].shape}"
        elif type(clean[key]) is dict:
            for subkey in clean[key].keys():
                if type(clean[key][subkey]) is np.ndarray:
                    assert (
                        comparison_data[key][subkey].shape == clean[key][subkey].shape
                    ), f"Shape mismatch for {key}-{subkey}: {comparison_data[key][subkey].shape} vs {clean[key][subkey].shape}"
                elif type(clean[key][subkey]) is dict:
                    for subsubkey in clean[key][subkey].keys():
                        if type(clean[key][subkey][subsubkey]) is np.ndarray:
                            assert (
                                comparison_data[key][subkey][subsubkey].shape
                                == clean[key][subkey][subsubkey].shape
                            ), f"Shape mismatch for {key}-{subkey}-{subsubkey}: {comparison_data[key][subkey][subsubkey].shape} vs {clean[key][subkey][subsubkey].shape}"


def get_clean_trial_dict(trial_data):
    clean_trials = {}
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] != "BAD_TRIAL":
                clean_trials[dmd] = trial_data[dmd][trl]
                break
    return clean_trials


def replace_bad_trials_with_null_data(trial_data, clean_trials):
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] == "BAD_TRIAL":
                trial_data[dmd][trl] = create_null_trial_data(clean_trials[dmd])
    return trial_data


def check_all_trial_shapes_match(trial_data, clean_trials):
    for dmd in trial_data.keys():
        for trl in trial_data[dmd].keys():
            if trial_data[dmd][trl] == "BAD_TRIAL":
                raise ValueError(
                    f"NO TRIALS SHOULD BE BAD HERE, DMD {dmd}, TRIAL {trl}"
                )
            else:
                assert_shape_match(clean_trials[dmd], trial_data[dmd][trl])
    print("all trial shapes match")
    return


def build_atomic_syndf(trial_data, refdata, trace_group="dF", trace_type="matchFilt"):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for dmd in trial_data.keys():
        trial_arrays = [
            trial_data[dmd][trl][trace_group][trace_type] for trl in range(ntrials)
        ]
        full_array = np.concatenate(trial_arrays, axis=0)
        full_array = full_array.swapaxes(0, 1)
        n_sources = full_array.shape[0]
        time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
        data_flat = full_array.flatten()
        # Build the source column: 0,0,...0, 1,1,...1, etc.
        source_flat = np.repeat(np.arange(n_sources), len(time))

        # Build the time column: time[0], time[1], ... for each source
        time_flat = np.tile(time.reshape(-1), n_sources)

        single_trial_len = trial_arrays[0].shape[0]
        trials_flat_single_source = np.repeat(
            np.arange(1, ntrials + 1), single_trial_len
        )
        trials_flat = np.tile(trials_flat_single_source, n_sources)

        # Construct Polars DataFrame
        df = pl.DataFrame(
            {
                "source-ID": source_flat,
                "time": time_flat,
                "data": data_flat,
                "dmd": dmd,
                "trace_group": trace_group,
                "trace_type": trace_type,
                "trial": trials_flat,
                "channel": 2,
            }
        )
        data_frames.append(df)
    return pl.concat(data_frames)


def build_noise_est_df(trial_data, trace_group="dF", trace_type="matchFilt"):

    noise_est_dataframes = []
    ntrials = len(trial_data[1])
    for dmd in trial_data.keys():
        trial_arrays = [
            trial_data[dmd][trl]["noiseEst"][trace_group][trace_type].flatten()
            for trl in range(ntrials)
        ]
        n_sources = trial_arrays[0].shape[0]
        trial_label_arrays = [
            np.ones_like(trial_arrays[0], dtype=int) * (trl + 1)
            for trl in range(ntrials)
        ]
        source_arrays = np.tile(np.arange(n_sources), len(trial_arrays))
        trial_arrays = np.array(trial_arrays).flatten()
        trial_label_arrays = np.array(trial_label_arrays).flatten()
        source_arrays = np.array(source_arrays).flatten()
        df = pl.DataFrame(
            {
                "source-ID": source_arrays,
                "trial": trial_label_arrays,
                "noise": trial_arrays,
                "dmd": dmd,
                "trace_group": trace_group,
                "trace_type": trace_type,
            }
        )
        noise_est_dataframes.append(df)
    return pl.concat(noise_est_dataframes)


def create_full_syndf(trial_data, refdata, trace_group="dF"):
    dF_keys_to_construct = ["matchFilt", "denoised", "nonneg", "events"]
    syndfs = []
    for key in dF_keys_to_construct:
        syndfs.append(
            build_atomic_syndf(
                trial_data, refdata, trace_group=trace_group, trace_type=key
            )
        )
    syndf = pl.concat(syndfs)
    noise_dfs = []
    for key in dF_keys_to_construct:
        noise_dfs.append(
            build_noise_est_df(trial_data, trace_group=trace_group, trace_type=key)
        )
    noise_df = pl.concat(noise_dfs)
    syndf = syndf.join(
        noise_df,
        on=["source-ID", "trial", "dmd", "trace_group", "trace_type"],
        how="left",
    )
    return syndf


def create_roidf(soma_info, trial_data, refdata):
    fs = int(refdata["params"]["analyzeHz"][0][0])
    roidfs = []
    for trace_type in ["F", "Fsvd"]:
        for dmd_id in soma_info.keys():
            dmd = int(dmd_id.split("-")[-1])
            for soma_ix, soma in enumerate(soma_info[dmd_id]):
                ntrials = len(trial_data[dmd])
                alltrials = [
                    trial_data[dmd][trl]["ROIs"][trace_type][:, :, soma_ix]
                    for trl in range(ntrials)
                ]
                all_trials = np.concatenate(alltrials, axis=1)
                soma_flat = all_trials.flatten()
                channels = np.array([2, 1])
                channels_full = np.repeat(channels, all_trials.shape[1])
                time = np.arange(0, all_trials.shape[1] / fs, step=1 / fs)
                time_full = np.tile(time, all_trials.shape[0])
                df = pl.DataFrame(
                    {
                        "time": time_full,
                        "data": soma_flat,
                        "channel": channels_full,
                        "dmd": dmd,
                        "soma-ID": soma,
                        "trace_type": trace_type,
                    }
                )
                roidfs.append(df)
    return pl.concat(roidfs)


def create_lsdf(trial_data, refdata, trace_group="dF", trace_type="ls"):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for i, channel_label in enumerate([2, 1]):
        for dmd in trial_data.keys():
            trial_arrays = [
                trial_data[dmd][trl][trace_group][trace_type][i]
                for trl in range(ntrials)
            ]
            full_array = np.concatenate(trial_arrays, axis=0)
            full_array = full_array.swapaxes(0, 1)
            n_sources = full_array.shape[0]
            time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
            data_flat = full_array.flatten()
            # Build the source column: 0,0,...0, 1,1,...1, etc.
            source_flat = np.repeat(np.arange(n_sources), len(time))

            # Build the time column: time[0], time[1], ... for each source
            time_flat = np.tile(time.reshape(-1), n_sources)

            single_trial_len = trial_arrays[0].shape[0]
            trials_flat_single_source = np.repeat(
                np.arange(1, ntrials + 1), single_trial_len
            )
            trials_flat = np.tile(trials_flat_single_source, n_sources)

            # Construct Polars DataFrame
            df = pl.DataFrame(
                {
                    "source-ID": source_flat,
                    "time": time_flat,
                    "data": data_flat,
                    "dmd": dmd,
                    "trace_group": trace_group,
                    "trace_type": trace_type,
                    "trial": trials_flat,
                    "channel": channel_label,
                }
            )
            data_frames.append(df)
    return pl.concat(data_frames)


def create_fzerodf(trial_data, refdata):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for i, channel_label in enumerate([2, 1]):
        for dmd in trial_data.keys():
            trial_arrays = [trial_data[dmd][trl]["F0"][i] for trl in range(ntrials)]
            full_array = np.concatenate(trial_arrays, axis=0)
            full_array = full_array.swapaxes(0, 1)
            n_sources = full_array.shape[0]
            time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
            data_flat = full_array.flatten()
            # Build the source column: 0,0,...0, 1,1,...1, etc.
            source_flat = np.repeat(np.arange(n_sources), len(time))

            # Build the time column: time[0], time[1], ... for each source
            time_flat = np.tile(time.reshape(-1), n_sources)

            single_trial_len = trial_arrays[0].shape[0]
            trials_flat_single_source = np.repeat(
                np.arange(1, ntrials + 1), single_trial_len
            )
            trials_flat = np.tile(trials_flat_single_source, n_sources)

            # Construct Polars DataFrame
            df = pl.DataFrame(
                {
                    "source-ID": source_flat,
                    "time": time_flat,
                    "data": data_flat,
                    "dmd": dmd,
                    "trace_group": "baseline",
                    "trace_type": "F0",
                    "trial": trials_flat,
                    "channel": channel_label,
                }
            )
            data_frames.append(df)
    return pl.concat(data_frames)


def generate_and_save_all_activity_dfs(
    subject, exp, loc, acq, overwrite=False, return_raw=False
):

    act_dir = os.path.join(DEFS.anmat_root, subject, exp, "activity_data", loc, acq)
    wis.util.gen.check_dir(act_dir)

    if len(os.listdir(act_dir)) > 0:
        if not overwrite:
            print(f"{act_dir} already exists and something is in it, skipping")
            return
        else:
            print(f"{act_dir} already exists and something is in it, overwriting")
            for file in os.listdir(act_dir):
                os.system(f"rm -rf {os.path.join(act_dir, file)}")

    ep = wis.util.info.sub_esum_path(subject, exp, loc, acq)
    trial_data, refdata, fs, ntrials = read_full_trial_data_dict(ep)
    if return_raw:
        return trial_data, refdata, fs, ntrials
    clean_trials = get_clean_trial_dict(trial_data)
    trial_data = replace_bad_trials_with_null_data(trial_data, clean_trials)
    check_all_trial_shapes_match(trial_data, clean_trials)
    soma_info = wis.scope.anat.get_somas_by_dmd(subject, exp, loc, acq)

    syndf_df = create_full_syndf(trial_data, refdata, trace_group="dF")
    syndf_df.write_parquet(os.path.join(act_dir, "syn_dF.parquet"))

    syndf_df_dff = create_full_syndf(trial_data, refdata, trace_group="dFF")
    syndf_df_dff.write_parquet(os.path.join(act_dir, "syn_dFF.parquet"))

    roidf = create_roidf(soma_info, trial_data, refdata)
    roidf.write_parquet(os.path.join(act_dir, "roidf.parquet"))

    lsdf = create_lsdf(trial_data, refdata)
    lsdf.write_parquet(os.path.join(act_dir, "lsdf.parquet"))

    lsdf_dff = create_lsdf(trial_data, refdata, trace_group="dFF")
    lsdf_dff.write_parquet(os.path.join(act_dir, "lsdf_dFF.parquet"))

    fzdf = create_fzerodf(trial_data, refdata)
    fzdf.write_parquet(os.path.join(act_dir, "fzdf.parquet"))


def lsdff_all_subjects():
    si = wis.peri.sync.load_sync_info()
    for subject in si.keys():
        for exp in si[subject].keys():
            locacqs = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
            for la in locacqs:
                loc, acq = la.split("--")
                print(f"Working on {subject} {exp} {loc} {acq}")
                try:
                    quick_lsdff(subject, exp, loc, acq)
                except Exception as e:
                    print(
                        f"Error generating lsdff for {subject} {exp} {loc} {acq}: {e}"
                    )
                    continue
    return


def quick_lsdff(subject, exp, loc, acq):
    act_dir = os.path.join(DEFS.anmat_root, subject, exp, "activity_data", loc, acq)
    wis.util.gen.check_dir(act_dir)

    ep = wis.util.info.sub_esum_path(subject, exp, loc, acq)
    trial_data, refdata, fs, ntrials = read_full_trial_data_dict(ep)
    clean_trials = get_clean_trial_dict(trial_data)
    trial_data = replace_bad_trials_with_null_data(trial_data, clean_trials)
    check_all_trial_shapes_match(trial_data, clean_trials)

    lsdf_dff = create_lsdf(trial_data, refdata, trace_group="dFF")
    lsdf_dff.write_parquet(os.path.join(act_dir, "lsdf_dFF.parquet"))
    return
