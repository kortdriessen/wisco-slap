from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import wisco_slap as wis  # noqa: F401 — registers the `.sx` accessor


def _make_da() -> xr.DataArray:
    # syn 0 is complete; syn 1 has a NaN at t=2, so t=2 is invalid.
    data = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, np.nan, 4.0, 5.0],
        ]
    )
    return xr.DataArray(
        data,
        dims=("syn_id", "time"),
        coords={
            "syn_id": [0, 1],
            "time": np.arange(5.0),
            "dend-ID": ("syn_id", ["A-1", "A-2"]),
        },
        attrs={"units": "dF/F"},
    )


def test_unify_valid_samples_nans_all_syns_at_invalid_times() -> None:
    da = _make_da()
    out = da.sx.unify_valid_samples()

    assert bool(np.isnan(out.sel(syn_id=0, time=2.0)))
    assert bool(np.isnan(out.sel(syn_id=1, time=2.0)))
    valid_t = [0.0, 1.0, 3.0, 4.0]
    np.testing.assert_array_equal(
        out.sel(time=valid_t).values, da.sel(time=valid_t).values
    )
    # Every syn shares one NaN pattern, and is_valid matches it.
    np.testing.assert_array_equal(out["is_valid"].values, [1, 1, 0, 1, 1])
    assert (np.isnan(out.values) == ~out["is_valid"].values[None, :]).all()


def test_unify_valid_samples_ignores_stale_is_valid_coord() -> None:
    # A stale (all-True) coord must not mask the real NaN at t=2.
    da = _make_da().assign_coords(is_valid=("time", np.ones(5, dtype=bool)))
    out = da.sx.unify_valid_samples()
    assert bool(np.isnan(out.sel(syn_id=0, time=2.0)))
    assert not out["is_valid"].values[2]


def test_unify_valid_samples_collapses_channel_dim() -> None:
    da = _make_da().expand_dims(channel=[0, 1]).copy()
    da[1, 0, 4] = np.nan  # channel 1, syn 0, t=4
    out = da.sx.unify_valid_samples()
    assert np.isnan(out.sel(time=[2.0, 4.0]).values).all()
    np.testing.assert_array_equal(out["is_valid"].values, [1, 1, 0, 1, 0])


def test_unify_valid_samples_preserves_coords_and_attrs() -> None:
    da = _make_da()
    out = da.sx.unify_valid_samples()
    assert "dend-ID" in out.coords
    assert out.attrs == da.attrs
    assert not np.isnan(da.sel(syn_id=0, time=2.0))  # not modified in place


def test_unify_valid_samples_requires_time_dim() -> None:
    da = xr.DataArray(np.zeros((2, 3)), dims=("syn_id", "x"))
    with pytest.raises(ValueError, match="time"):
        da.sx.unify_valid_samples()
