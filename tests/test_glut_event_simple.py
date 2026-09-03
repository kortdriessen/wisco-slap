from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

from wisco_slap.scope._GLUT_EVENT_SIMPLE import _threshold_runs_2d, detect_basic_events

DT = 0.005  # 200 Hz


def _da(values: np.ndarray, dims: tuple[str, ...], **coords) -> xr.DataArray:
    """Build a scopex-like DataArray on a 200 Hz grid starting at t=10 s."""
    n_time = values.shape[-1]
    base = {"time": 10.0 + np.arange(n_time) * DT}
    return xr.DataArray(values, dims=dims, coords={**base, **coords})


# ---------------------------------------------------------------------------
# numpy core
# ---------------------------------------------------------------------------


def test_threshold_runs_2d_summarises_each_run() -> None:
    #            idx: 0    1    2    3    4    5    6    7
    v = np.array([[0.0, 2.0, 3.0, 0.0, 5.0, 1.5, 0.0, 4.0]])
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=1)
    assert runs["unit"].tolist() == [0, 0, 0]
    assert runs["start"].tolist() == [1, 4, 7]
    assert runs["end"].tolist() == [2, 5, 7]
    assert runs["peak"].tolist() == [2, 4, 7]
    assert runs["n_samples"].tolist() == [2, 2, 1]
    assert runs["peak_value"].tolist() == [3.0, 5.0, 4.0]
    assert runs["mean_value"] == pytest.approx([2.5, 3.25, 4.0])


def test_threshold_runs_2d_min_samples_drops_short_runs() -> None:
    v = np.array([[0.0, 2.0, 3.0, 0.0, 5.0, 1.5, 0.0, 4.0]])
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=2)
    assert runs["start"].tolist() == [1, 4]
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=3)
    assert runs["start"].tolist() == []
    assert runs["peak_value"].dtype == np.float64


def test_threshold_runs_2d_runs_do_not_cross_rows() -> None:
    # Row 0 is active through its last sample, row 1 from its first sample:
    # in the flattened array these are adjacent, but must be two runs.
    v = np.array([[0.0, 5.0, 5.0], [5.0, 5.0, 0.0]])
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=1)
    assert runs["unit"].tolist() == [0, 1]
    assert runs["start"].tolist() == [1, 0]
    assert runs["end"].tolist() == [2, 1]


def test_threshold_runs_2d_tie_picks_earliest_peak() -> None:
    v = np.array([[0.0, 3.0, 3.0, 3.0, 0.0]])
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=1)
    assert runs["peak"].tolist() == [1]


def test_threshold_runs_2d_empty_mask() -> None:
    v = np.zeros((3, 10))
    runs = _threshold_runs_2d(v > 1.0, v, min_samples=1)
    assert all(a.shape == (0,) for a in runs.values())


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def test_detect_basic_events_single_unit_columns_and_values() -> None:
    v = np.array([[0.0, 2.0, 3.0, 0.0, 5.0, 1.5, 0.0, 4.0]])
    da = _da(v, ("syn_id", "time"), syn_id=[7])
    df = detect_basic_events(da, threshold=1.0, min_samples=2)

    assert df.columns == [
        "syn_id",
        "start_time",
        "peak_time",
        "end_time",
        "duration",
        "n_samples",
        "peak_snr",
        "mean_snr",
        "start_idx",
        "peak_idx",
        "end_idx",
    ]
    assert df.height == 2
    assert df["syn_id"].to_list() == [7, 7]
    assert df["start_idx"].to_list() == [1, 4]
    assert df["peak_idx"].to_list() == [2, 4]
    assert df["end_idx"].to_list() == [2, 5]
    assert df["start_time"].to_list() == pytest.approx([10.005, 10.020], abs=1e-9)
    assert df["peak_time"].to_list() == pytest.approx([10.010, 10.020], abs=1e-9)
    assert df["end_time"].to_list() == pytest.approx([10.010, 10.025], abs=1e-9)
    assert df["n_samples"].to_list() == [2, 2]
    assert df["duration"].to_list() == pytest.approx([2 * DT, 2 * DT], abs=1e-9)
    assert df["peak_snr"].to_list() == [3.0, 5.0]
    assert df["mean_snr"].to_list() == pytest.approx([2.5, 3.25])


def test_detect_basic_events_threshold_is_strict() -> None:
    v = np.array([[1.0, 1.0, 1.0, 2.0, 2.0]])
    da = _da(v, ("syn_id", "time"), syn_id=[0])
    df = detect_basic_events(da, threshold=1.0, min_samples=1)
    assert df["start_idx"].to_list() == [3]
    assert df["end_idx"].to_list() == [4]


def test_detect_basic_events_nan_breaks_runs() -> None:
    v = np.array([[5.0, 5.0, np.nan, 5.0, 5.0]])
    da = _da(v, ("syn_id", "time"), syn_id=[0])
    df = detect_basic_events(da, threshold=1.0, min_samples=2)
    assert df["start_idx"].to_list() == [0, 3]
    assert df["end_idx"].to_list() == [1, 4]
    assert df["mean_snr"].to_list() == [5.0, 5.0]


def test_detect_basic_events_propagates_all_coords() -> None:
    # (channel, syn_id, time) with the usual mixed-dtype label coords on syn_id,
    # a scalar coord, and a coord that varies along time.
    v = np.zeros((2, 3, 10))
    v[0, 1, 2:5] = [2.0, 9.0, 3.0]  # ch0, syn 1
    v[1, 2, 7:9] = [4.0, 6.0]  # ch1, syn 2
    v[1, 0, 0:2] = [3.0, 3.0]  # ch1, syn 0
    da = _da(
        v,
        ("channel", "syn_id", "time"),
        channel=[0, 1],
        syn_id=[10, 11, 12],
        **{
            "soma-ID": (
                "syn_id",
                np.array(["soma1", None, "unidentifiable_soma"], dtype=object),
            ),
            "dend-ID": ("syn_id", np.array(["A-1", "A-1", None], dtype=object)),
            "pos": ("syn_id", np.array([np.nan, 43.0, 42.0])),
            "dmd-depth": ("syn_id", np.array([60, 60, 60], dtype=np.int32)),
            "trial": ("time", np.repeat([1, 2], 5)),
        },
    )
    da = da.assign_coords(acq="acq_1")
    df = detect_basic_events(da, threshold=1.0, min_samples=2)

    # row order: unit row-major (channel, then syn_id), then time
    assert df.select("channel", "syn_id").rows() == [(0, 11), (1, 10), (1, 12)]
    assert df["soma-ID"].to_list() == [None, "soma1", "unidentifiable_soma"]
    assert df["dend-ID"].to_list() == ["A-1", "A-1", None]
    assert df["pos"].to_list() == [43.0, None, 42.0]
    assert df["pos"].dtype == pl.Float64
    assert df["dmd-depth"].to_list() == [60, 60, 60]
    assert df["dmd-depth"].dtype == pl.Int32
    assert df["acq"].to_list() == ["acq_1"] * 3
    # time-varying coord is evaluated at the peak sample (peaks at idx 3, 0, 8)
    assert df["peak_idx"].to_list() == [3, 0, 8]
    assert df["trial"].to_list() == [1, 1, 2]
    assert df["peak_snr"].to_list() == [9.0, 3.0, 6.0]
    assert df["mean_snr"].to_list() == pytest.approx([14 / 3, 3.0, 5.0])


def test_detect_basic_events_empty_result_keeps_schema() -> None:
    v = np.zeros((2, 5))
    da = _da(
        v,
        ("syn_id", "time"),
        syn_id=[0, 1],
        **{"dend-ID": ("syn_id", np.array(["A-1", None], dtype=object))},
    )
    df = detect_basic_events(da, threshold=1.0)
    assert df.height == 0
    assert df.columns[:2] == ["syn_id", "dend-ID"]
    assert df.schema["dend-ID"] == pl.Utf8
    assert df.schema["n_samples"] == pl.Int64
    assert df.schema["peak_snr"] == pl.Float64


def test_detect_basic_events_time_only_array() -> None:
    v = np.array([0.0, 2.0, 2.0, 0.0])
    da = _da(v, ("time",))
    df = detect_basic_events(da, threshold=1.0)
    assert df.height == 1
    assert df["start_idx"].to_list() == [1]
    assert df["end_idx"].to_list() == [2]


def test_detect_basic_events_value_name() -> None:
    da = _da(np.array([[0.0, 2.0, 2.0]]), ("syn_id", "time"), syn_id=[0])
    df = detect_basic_events(da, threshold=1.0, value_name="dff")
    assert "peak_dff" in df.columns
    assert "mean_dff" in df.columns
    assert "peak_snr" not in df.columns


def test_detect_basic_events_per_unit_threshold() -> None:
    v = np.array([[3.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    da = _da(v, ("syn_id", "time"), syn_id=[0, 1])
    thr = xr.DataArray([1.0, 5.0], dims=("syn_id",), coords={"syn_id": [0, 1]})
    df = detect_basic_events(da, threshold=thr)
    assert df["syn_id"].to_list() == [0]


def test_detect_basic_events_per_unit_threshold_must_align_exactly() -> None:
    v = np.array([[3.0, 3.0, 0.0], [3.0, 3.0, 0.0]])
    da = _da(v, ("syn_id", "time"), syn_id=[0, 1])
    thr = xr.DataArray([1.0], dims=("syn_id",), coords={"syn_id": [0]})
    with pytest.raises(ValueError):
        detect_basic_events(da, threshold=thr)


def test_detect_basic_events_dict_input_adds_dmd_column() -> None:
    v1 = np.array([[0.0, 2.0, 2.0, 0.0]])
    v2 = np.array([[2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 2.0]])
    d1 = _da(v1, ("syn_id", "time"), syn_id=[0], pos=("syn_id", np.array([1.5])))
    d2 = _da(
        v2,
        ("syn_id", "time"),
        syn_id=[0, 1],
        pos=("syn_id", np.array([3, 4], dtype=np.int32)),  # dtype differs by DMD
    )
    df = detect_basic_events({"dmd_1": d1, "dmd_2": d2}, threshold=1.0)
    assert df.columns[0] == "dmd"
    assert df["dmd"].to_list() == [1, 2, 2]
    assert df["syn_id"].to_list() == [0, 0, 1]
    assert df["pos"].to_list() == [1.5, 3.0, 4.0]


def test_detect_basic_events_dict_input_respects_existing_dmd_coord() -> None:
    v = np.array([[0.0, 2.0, 2.0, 0.0]])
    d1 = _da(v, ("syn_id", "time"), syn_id=[0], dmd=("syn_id", np.array([1])))
    df = detect_basic_events({"dmd_1": d1}, threshold=1.0)
    assert df.columns.count("dmd") == 1
    assert df["dmd"].to_list() == [1]


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"threshold": 1.0, "min_samples": 0}, ValueError),
        ({"threshold": float("nan")}, ValueError),
        ({"threshold": 1.0, "time_dim": "t"}, ValueError),
        ({"threshold": 1.0, "value_name": "x"}, ValueError),  # coord clash
    ],
)
def test_detect_basic_events_invalid_inputs(kwargs: dict, exc: type) -> None:
    da = _da(
        np.array([[0.0, 2.0, 2.0]]),
        ("syn_id", "time"),
        syn_id=[0],
        peak_x=("syn_id", np.array([0])),
    )
    with pytest.raises(exc):
        detect_basic_events(da, **kwargs)


def test_detect_basic_events_rejects_non_array() -> None:
    with pytest.raises(TypeError):
        detect_basic_events([1, 2, 3], threshold=1.0)  # type: ignore[arg-type]
