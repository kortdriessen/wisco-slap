"""Synthetic contract checks for the coordinated glutamate packet detector.

These fixtures check known signal construction, boundaries and reported units.
They do not establish specificity or biological validity on recorded data.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict

import numpy as np
import polars as pl
import pytest
import xarray as xr

from wisco_slap.scope.rings import (
    LEVEL_DEFAULTS,
    RING_SCHEMA,
    RingParams,
    default_params,
    detect_rings,
)

DT = 0.005

# The fixtures below are 20-sample (0.1 s) pulses built for the detector's
# original settings. The level-dependent review presets that are now the
# defaults (0.3 s minimum duration, 0.2 s smoothing) would reject them, so
# every contract check pins these settings and overrides them as needed.
SYNTHETIC: dict[str, float] = {
    "activity_threshold": 4.0,
    "min_prominence": 0.20,
    "smooth_s": 0.05,
    "min_duration_s": 0.04,
}


def detect(a: object, **kwargs: object) -> object:
    """``detect_rings`` with the synthetic-fixture settings unless overridden."""
    return detect_rings(a, **{**SYNTHETIC, **kwargs})


def array(values: np.ndarray, **coords: object) -> xr.DataArray:
    """Attach explicit IDs, seconds and fluorescence units to a fixture."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("syn_id", "time"),
        coords={
            "syn_id": [f"s{i}" for i in range(len(values))],
            "time": np.arange(values.shape[1]) * DT,
            **coords,
        },
        attrs={"units": "test fluorescence"},
    )


def pulse(
    n: int = 6,
    length: int = 240,
    start: int = 50,
    stop: int = 70,
    participants: int = 4,
) -> xr.DataArray:
    values = np.zeros((n, length))
    values[:participants, start:stop] = 6.0
    return array(values)


@pytest.mark.parametrize("value", [0.0, 1.0, 1e6])
@pytest.mark.parametrize("normalization", ["none", "robust"])
def test_constant_has_no_arrival_or_phasic_peak(
    value: float, normalization: str
) -> None:
    result = detect(array(np.full((8, 400), value)), normalization=normalization)
    assert result.is_empty()
    assert result.schema == RING_SCHEMA
    assert result.schema["participant_ids"] == pl.List(pl.String)
    assert result.schema["signal_integral"] == pl.Float64


def test_independent_unit_noise_is_not_a_coordinated_packet() -> None:
    noise = np.random.default_rng(847).normal(size=(24, 4000))
    assert detect(array(noise)).is_empty()


def test_one_giant_roi_cannot_meet_distinct_roi_requirement() -> None:
    a = pulse(participants=1)
    a.values[0] *= 1e6
    assert detect(a).is_empty()


def test_shared_transient_is_detected_and_transpose_is_identical() -> None:
    a = pulse()
    expected = detect(a)
    assert expected.height == 1
    assert expected.row(0, named=True)["participant_ids"] == ["s0", "s1", "s2", "s3"]
    assert expected.equals(detect(a.transpose("time", "syn_id")))


def test_broad_shared_pulse_uses_arrivals_at_its_leading_edge() -> None:
    # The plateau midpoint is far beyond the 60 ms arrival-count window.
    # Its well-supported leading edge must still qualify the 500 ms packet.
    result = detect(pulse(length=400, stop=150))
    assert result.height == 1
    assert result["n_newly_active"][0] == 4


def test_two_nearby_arrivals_remain_separate() -> None:
    a = pulse()
    a.values[:4, 82:102] = 6
    result = detect(a)
    assert result.height == 2
    assert result["end_s"][0] <= result["start_s"][1]
    assert result["ring_id"].to_list() == [0, 1]
    assert result["n_newly_active"].to_list() == [4, 4]


def test_overlapping_widths_split_at_trough_and_recompute_participation() -> None:
    values = np.zeros((12, 240))
    values[:4, 40:160] = 6
    values[4:8, 60:80] = 6
    values[8:12, 100:120] = 6
    result = detect(array(values), smooth_s=0)
    assert result.height == 2
    assert result["end_index"][0] <= result["start_index"][1]
    assert result["participant_ids"][0].to_list() == [f"s{i}" for i in range(8)]
    assert result["participant_ids"][1].to_list() == [
        f"s{i}" for i in [0, 1, 2, 3, 8, 9, 10, 11]
    ]
    for row in result.iter_rows(named=True):
        assert row["start_s"] <= row["peak_s"] < row["end_s"]
        assert row["rise_s"] >= 0 and row["decay_s"] > 0


def test_split_cannot_borrow_arrivals_from_the_previous_packet() -> None:
    values = np.zeros((12, 240))
    values[:4, 40:160] = 6
    values[4:8, 60:80] = 6
    values[4:8, 80:100] = 3  # Below activity, but above the rearming threshold.
    values[4:8, 100:120] = 6
    result = detect(array(values), smooth_s=0)
    assert result.height == 1
    assert result["n_newly_active"][0] == 8


@pytest.mark.parametrize("gap", ["nan", "valid_time", "valid_roi", "time_jump"])
def test_missing_samples_and_time_jumps_never_join_packets(gap: str) -> None:
    a = pulse()
    a.values[:4, 140:160] = 6
    if gap == "nan":
        a.values[0, 95:105] = np.nan
    elif gap == "valid_time":
        valid = np.ones(a.sizes["time"], dtype=bool)
        valid[95:105] = False
        a = a.assign_coords(is_valid=("time", valid))
    elif gap == "valid_roi":
        valid = np.ones(a.shape, dtype=bool)
        valid[0, 95:105] = False
        # Exercise the supported transposed coordinate order.
        a = a.assign_coords(is_valid=(("time", "syn_id"), valid.T))
    else:
        times = a.time.values.copy()
        times[100:] += 10
        a = a.assign_coords(time=times)
    result, diagnostics = detect(a, return_diagnostics=True)
    assert result.height == 2
    assert result["valid_segment"].to_list() == [0, 1]
    assert result["end_index"][0] < 95
    assert result["start_index"][1] > 105
    d = diagnostics["population"]
    assert d.attrs["n_valid_segments"] == 2
    if gap != "time_jump":
        assert not d.valid.values[95:105].any()
        assert np.isnan(d.envelope.values[95:105]).all()
        assert d.attrs["valid_exposure_s"] == pytest.approx(230 * DT)
    else:
        assert d.valid.values.all()
        assert d.attrs["valid_exposure_s"] == pytest.approx(240 * DT)


@pytest.mark.parametrize("position", ["left_crop", "right_crop", "after_gap"])
def test_crop_or_gap_boundary_does_not_invent_an_arrival(position: str) -> None:
    if position == "left_crop":
        a = pulse().isel(time=slice(55, 120))
    elif position == "right_crop":
        a = pulse().isel(time=slice(0, 65))
    else:
        a = pulse()
        a.values[:, 45:55] = np.nan
    assert detect(a).is_empty()


def test_group_coordinate_preserves_scope_and_local_validity() -> None:
    values = np.zeros((6, 240))
    values[:3, 50:70] = 6
    values[3:, 140:160] = 6
    values[0, 140:160] = np.nan  # Missing group A must not invalidate B.
    a = array(values, dendrite=("syn_id", ["A"] * 3 + ["B"] * 3))
    result, diag = detect(a, level="dendrite", return_diagnostics=True)
    assert result["group"].to_list() == ["A", "B"]
    assert result["level"].to_list() == ["dendrite", "dendrite"]
    assert result["n_synapses"].to_list() == [3, 3]
    assert result["fraction_participating"].to_list() == [1.0, 1.0]
    assert not diag["A"].valid.values[150]
    assert diag["B"].valid.values[150]


def test_missing_group_annotations_warn_and_small_group_cannot_qualify() -> None:
    a = pulse().assign_coords(dendrite=("syn_id", ["A", "A", None, "", np.nan, "B"]))
    with pytest.warns(UserWarning, match="Excluded 3 synapses"):
        result, diag = detect(a, level="dendrite", return_diagnostics=True)
    assert result.is_empty()
    assert set(diag) == {"A", "B"}
    assert diag["A"].sizes["syn_id"] == 2


def test_integrals_counts_and_signal_shares_match_constructed_input() -> None:
    values = np.zeros((6, 240))
    values[:3, 50:70] = 6
    values[3, 50:70] = 1  # Positive contribution without participation.
    values[4, 50:70] = -2  # Must affect signed integral only.
    result = detect(array(values), smooth_s=0)
    assert result.height == 1
    r = result.row(0, named=True)
    assert (r["start_index"], r["end_index"]) == (50, 70)
    assert r["start_s"] == pytest.approx(0.25)
    assert r["end_s"] == pytest.approx(0.35)
    assert r["duration_s"] == pytest.approx(0.10)
    assert r["n_participating"] == r["n_newly_active"] == r["peak_active_count"] == 3
    assert r["participant_ids"] == r["newly_active_ids"] == ["s0", "s1", "s2"]
    for k in [
        "fraction_participating",
        "peak_active_fraction",
        "mean_active_fraction",
        "peak_new_fraction",
    ]:
        assert r[k] == pytest.approx(0.5)
    assert r["signal_sum"] == pytest.approx(340)
    assert r["signal_integral"] == pytest.approx(1.7)
    assert r["positive_signal_integral"] == pytest.approx(1.9)
    assert r["excess_integral"] == pytest.approx(0.6)
    assert r["peak_signal_sum"] == pytest.approx(17)
    assert r["participant_signal_fraction"] == pytest.approx(18 / 19)
    assert r["largest_synapse_signal_fraction"] == pytest.approx(6 / 19)
    assert r["onset_spread_s"] == 0
    assert r["sample_interval_s"] == pytest.approx(DT)
    assert r["signal_units"] == "test fluorescence"


def test_robust_normalization_preserves_original_integral_units() -> None:
    a = pulse()
    # Per-ROI offsets and scales are deliberately different.
    a.values[:] = 10 + a.values * 2
    result, diag = detect(a, normalization="robust", return_diagnostics=True)
    assert result.height == 1
    r = result.row(0, named=True)
    selected = a.values[:, r["start_index"] : r["end_index"]]
    threshold = diag["population"].input_threshold.values[:, None]
    assert r["signal_integral"] == pytest.approx(selected.sum() * DT)
    assert r["excess_integral"] == pytest.approx(
        np.maximum(selected - threshold, 0).sum() * DT
    )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_too_short_input_returns_typed_empty_table(size: int) -> None:
    result, diagnostics = detect(
        pulse().isel(time=slice(size)), return_diagnostics=True
    )
    assert result.is_empty()
    assert result.schema["end_index"] == pl.Int64
    assert diagnostics == {}


def test_diagnostics_record_parameters_actually_used() -> None:
    _, diag = detect(pulse(), activity_threshold=5.5, return_diagnostics=True)
    params = diag["population"].attrs["params"]
    assert params["activity_threshold"] == 5.5
    assert params["normalization"] == "none"
    assert set(params) == set(RingParams.__dataclass_fields__)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activity_threshold": 0},
        {"activity_threshold": np.nan},
        {"min_active_synapses": 0},
        {"min_active_synapses": 2.5},
        {"min_active_synapses": True},
        {"min_peak_fraction": -0.1},
        {"min_peak_fraction": 1.1},
        {"min_onset_fraction": 1.1},
        {"min_prominence": 0},
        {"min_prominence": 1.1},
        {"smooth_s": -1},
        {"min_separation_s": -1},
        {"boundary_fraction": 0},
        {"boundary_fraction": 1},
        {"rearm_fraction": 0},
        {"rearm_fraction": 1},
        {"min_duration_s": 0},
        {"min_duration_s": 3, "max_duration_s": 2},
        {"onset_window_s": 0},
        {"prominence_window_s": 0},
        {"normalization": "snr"},
    ],
)
def test_invalid_parameters_fail_explicitly(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RingParams(**kwargs)
    with pytest.raises(ValueError):
        detect(pulse(), **kwargs)


@pytest.mark.parametrize(
    "kind",
    [
        "not_array",
        "extra_dimension",
        "reversed_time",
        "nan_time",
        "duplicate_time",
        "jitter",
        "duplicate_ids",
        "missing_level",
        "wrong_valid_dim",
    ],
)
def test_invalid_data_contract_fails_explicitly(kind: str) -> None:
    a = pulse()
    options: dict[str, str] = {}
    if kind == "not_array":
        a = a.values
    elif kind == "extra_dimension":
        a = a.expand_dims(channel=[1])
    elif kind == "reversed_time":
        a = a.isel(time=slice(None, None, -1))
    elif kind in ["nan_time", "duplicate_time", "jitter"]:
        times = a.time.values.copy()
        times[10] = {
            "nan_time": np.nan,
            "duplicate_time": times[9],
            "jitter": times[10] + DT * 0.15,
        }[kind]
        a = a.assign_coords(time=times)
    elif kind == "duplicate_ids":
        a = a.assign_coords(syn_id=["same"] * 6)
    elif kind == "missing_level":
        options["level"] = "missing"
    else:
        a = a.assign_coords(is_valid=("syn_id", [True] * 6))
    with pytest.raises(ValueError):
        detect(a, **options)


def test_defaults_are_the_level_dependent_review_presets() -> None:
    grouped = pulse().assign_coords(dendrite=("syn_id", ["A"] * 6))
    _, pop_diag = detect_rings(pulse(), return_diagnostics=True)
    _, grp_diag = detect_rings(grouped, level="dendrite", return_diagnostics=True)
    pop = pop_diag["population"].attrs["params"]
    grp = grp_diag["A"].attrs["params"]
    assert pop == asdict(RingParams()) == asdict(default_params("syn_id"))
    assert grp == asdict(default_params("dendrite"))
    for level, params in [("syn_id", pop), ("dend-ID", grp)]:
        for name, value in LEVEL_DEFAULTS[level].items():
            assert params[name] == value
    shared = set(pop) - set(LEVEL_DEFAULTS["syn_id"])
    assert {k: pop[k] for k in shared} == {k: grp[k] for k in shared}
    assert (pop["smooth_s"], pop["min_duration_s"]) == (0.20, 0.30)


def test_explicit_settings_override_level_defaults() -> None:
    grouped = pulse().assign_coords(dendrite=("syn_id", ["A"] * 6))
    _, diag = detect_rings(
        grouped, level="dendrite", activity_threshold=4.0, return_diagnostics=True
    )
    params = diag["A"].attrs["params"]
    assert params["activity_threshold"] == 4.0
    assert params["min_prominence"] == LEVEL_DEFAULTS["dend-ID"]["min_prominence"]
    assert default_params("dendrite", min_prominence=0.3).min_prominence == 0.3


def test_signature_defaults_match_ring_params_fields() -> None:
    signature = inspect.signature(detect_rings)
    for name, field in RingParams.__dataclass_fields__.items():
        default = signature.parameters[name].default
        if name in LEVEL_DEFAULTS["syn_id"]:
            assert default is None
        else:
            assert default == field.default


@pytest.mark.parametrize("level", ["syn_id", "dendrite"])
def test_default_presets_detect_a_packet_longer_than_the_minimum(level: str) -> None:
    # 80 samples = 0.4 s, above the 0.3 s default minimum duration.
    a = pulse(length=400, start=50, stop=130)
    if level != "syn_id":
        a = a.assign_coords(dendrite=("syn_id", ["A"] * 6))
    result = detect_rings(a, level=level)
    assert result.height == 1
    assert result["n_newly_active"][0] == 4
    assert result["activity_threshold"][0] == default_params(level).activity_threshold
