from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl
import pytest

matplotlib.use("Agg")

import wisco_slap as wis  # noqa: E402
from wisco_slap.pns.overview_plot_gen import (  # noqa: E402
    ID_COLUMNS,
    error_path,
    generate_plot_set,
    set_members,
)
from wisco_slap.viz.overview_and_summary_plots import (  # noqa: E402
    PlotSpec,
    acq_level,
    discover_specs,
    plot_specs,
)


def test_discover_specs_finds_registered_acq_level_plots() -> None:
    names = [s.name for s in discover_specs(acq_level)]
    assert "basic_sync_summary" in names
    assert names == sorted(names)
    assert plot_specs("acq_level") == discover_specs(acq_level)


def test_plot_specs_unknown_set_raises() -> None:
    with pytest.raises(ValueError, match="Unknown plot set"):
        plot_specs("nope")


class _FakePlot:
    """A plot spec whose failures can be toggled per member (any id length)."""

    def __init__(self, root: Path, failing: set[str]) -> None:
        self.root = root
        self.failing = failing
        self.calls: list[tuple[str, ...]] = []

    def path(self, *ids: str) -> Path:
        return self.root / "fake" / ("--".join(ids) + ".png")

    def plot(self, *ids: str, overwrite: bool):
        self.calls.append(ids)
        tag = "--".join(ids[2:])
        if tag in self.failing:
            raise RuntimeError(f"boom for {tag}")
        out = self.path(*ids)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        fig.savefig(out, dpi=20)
        return fig, ax

    def spec(self) -> PlotSpec:
        return PlotSpec(name="fake", plot=self.plot, path=self.path)


def _install(monkeypatch: pytest.MonkeyPatch, fake: _FakePlot) -> None:
    monkeypatch.setattr(
        wis.meta.get,
        "acq_master",
        lambda: {"m": {"exp_1": ["loc_A--acq_1", "loc_B--acq_1"]}},
    )
    monkeypatch.setattr(
        wis.pns.overview_plot_gen,
        "_single_soma_members",
        lambda: [("m", "exp_1", "loc_A", "acq_1", "soma1"), ("m", "exp_1", "loc_B", "acq_1", "soma2")],
    )
    monkeypatch.setattr(
        wis.viz.overview_and_summary_plots, "plot_specs", lambda name: [fake.spec()]
    )


def test_generate_plot_set_writes_blank_plot_and_error_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakePlot(tmp_path, failing={"loc_B--acq_1"})
    _install(monkeypatch, fake)
    plt.close("all")  # other test modules may leave figures open

    df = generate_plot_set(set="acq_level", overwrite=False)
    assert isinstance(df, pl.DataFrame)
    assert df.sort("loc")["status"].to_list() == ["generated", "error"]

    good = fake.path("m", "exp_1", "loc_A", "acq_1")
    bad = fake.path("m", "exp_1", "loc_B", "acq_1")
    assert good.exists() and not error_path(good).exists()
    assert bad.exists(), "a blank placeholder plot must be written on failure"
    err = error_path(bad)
    assert err.name == "m--exp_1--loc_B--acq_1--ERROR.txt"
    text = err.read_text()
    assert "RuntimeError: boom for loc_B--acq_1" in text
    assert "Traceback" in text
    assert df.filter(pl.col("loc") == "loc_B")["error"][0].startswith("RuntimeError")
    # No figures leaked from either the success or the failure path.
    assert plt.get_fignums() == []


def test_generate_plot_set_skips_existing_but_retries_errored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakePlot(tmp_path, failing={"loc_B--acq_1"})
    _install(monkeypatch, fake)
    generate_plot_set(overwrite=False)
    fake.calls.clear()

    # Second run: the good plot is skipped, the errored one is retried.
    fake.failing.clear()
    df = generate_plot_set(overwrite=False)
    assert fake.calls == [("m", "exp_1", "loc_B", "acq_1")]
    assert df.sort("loc")["status"].to_list() == ["skipped", "generated"]
    assert not error_path(fake.path("m", "exp_1", "loc_B", "acq_1")).exists()

    # overwrite=True rebuilds everything.
    fake.calls.clear()
    df = generate_plot_set(overwrite=True)
    assert len(fake.calls) == 2
    assert df["status"].to_list() == ["generated", "generated"]


def test_generate_plot_set_rejects_unknown_set() -> None:
    with pytest.raises(ValueError, match="Unknown plot set 'exp_level'"):
        generate_plot_set(set="exp_level")
    with pytest.raises(ValueError, match="Unknown plot set"):
        set_members("exp_level")


def test_set_members_id_lengths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install(monkeypatch, _FakePlot(tmp_path, failing=set()))
    assert set_members("acq_level") == [
        ("m", "exp_1", "loc_A", "acq_1"),
        ("m", "exp_1", "loc_B", "acq_1"),
    ]
    assert all(len(ids) == 5 for ids in set_members("single_soma"))


def test_generate_plot_set_single_soma_iterates_subgroup_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakePlot(tmp_path, failing={"loc_B--acq_1--soma2"})
    _install(monkeypatch, fake)
    plt.close("all")

    df = generate_plot_set(set="single_soma", overwrite=False)
    assert fake.calls == [
        ("m", "exp_1", "loc_A", "acq_1", "soma1"),
        ("m", "exp_1", "loc_B", "acq_1", "soma2"),
    ]
    assert list(df.columns) == ["plot", *ID_COLUMNS, "status", "path", "error"]
    assert df["soma_id"].to_list() == ["soma1", "soma2"]
    assert df["status"].to_list() == ["generated", "error"]
    bad = fake.path("m", "exp_1", "loc_B", "acq_1", "soma2")
    assert bad.exists()
    assert error_path(bad).name == "m--exp_1--loc_B--acq_1--soma2--ERROR.txt"
    assert "boom for loc_B--acq_1--soma2" in error_path(bad).read_text()
    # acq_level rows leave the soma column empty
    df = generate_plot_set(set="acq_level", overwrite=False)
    assert df["soma_id"].to_list() == ["", ""]
    assert plt.get_fignums() == []
