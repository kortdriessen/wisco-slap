"""Tests for loading subgroup sets and splitting member IDs into a DataFrame."""

import polars as pl
import pytest

from wisco_slap.meta.subgroups import (
    list_subgroup_sets,
    load_subgroup_set,
    members_to_df,
)
from wisco_slap.meta.subgroups.single_soma import single_soma


def test_single_soma_ids_split_into_soma_id_column():
    ids = ["kaus--exp_1--loc_E--acq_1--soma1", "avior--exp_2--loc_C--acq_2--soma3"]
    df = members_to_df(ids, "single_soma")
    assert df.columns == ["subject", "exp", "loc", "acq", "soma_id"]
    assert df.row(0) == ("kaus", "exp_1", "loc_E", "acq_1", "soma1")
    assert df.get_column("soma_id").to_list() == ["soma1", "soma3"]


def test_unknown_set_uses_unit_column_and_drops_it_when_absent():
    with_suffix = members_to_df(["a--exp_1--loc_A--acq_1--x--y"], "other")
    assert with_suffix.columns == ["subject", "exp", "loc", "acq", "unit"]
    assert with_suffix.item(0, "unit") == "x--y"

    no_suffix = members_to_df(["a--exp_1--loc_A--acq_1"], "other")
    assert no_suffix.columns == ["subject", "exp", "loc", "acq"]


def test_empty_members_gives_empty_typed_frame():
    df = members_to_df([], "single_soma")
    assert df.shape == (0, 4)
    assert df.schema["subject"] == pl.String


def test_malformed_id_raises():
    with pytest.raises(ValueError, match="Malformed member ID"):
        members_to_df(["kaus--exp_1"], "single_soma")


def test_list_subgroup_sets_finds_same_name_generator_scripts():
    assert "single_soma" in list_subgroup_sets()


def test_load_subgroup_set_calls_the_generators_members_function(monkeypatch):
    """The loader must go through ``members()``, not any cached file."""
    calls: list[bool] = []

    def fake_members(verbose: bool = True) -> list[str]:
        calls.append(verbose)
        return ["kaus--exp_1--loc_E--acq_1--soma1", "avior--exp_2--loc_C--acq_2--soma3"]

    monkeypatch.setattr(single_soma, "members", fake_members)

    ids = load_subgroup_set("single_soma", verbose=False)
    assert ids == [
        "kaus--exp_1--loc_E--acq_1--soma1",
        "avior--exp_2--loc_C--acq_2--soma3",
    ]
    assert calls == [False]

    df = load_subgroup_set("single_soma", verbose=False, return_df=True)
    assert df.columns == ["subject", "exp", "loc", "acq", "soma_id"]
    assert df.height == 2


def test_verbose_false_prints_nothing_even_if_the_generator_does(monkeypatch, capsys):
    def chatty_members(verbose: bool = True) -> list[str]:
        print("noise from a helper that ignores verbose")
        return ["kaus--exp_1--loc_E--acq_1--soma1"]

    monkeypatch.setattr(single_soma, "members", chatty_members)
    load_subgroup_set("single_soma", verbose=False)
    assert capsys.readouterr().out == ""


def test_unknown_set_lists_the_available_ones():
    with pytest.raises(FileNotFoundError, match="No subgroup set 'not_a_set'"):
        load_subgroup_set("not_a_set")


def test_generator_without_members_function_is_rejected(monkeypatch):
    monkeypatch.delattr(single_soma, "members")
    with pytest.raises(AttributeError, match="does not define"):
        load_subgroup_set("single_soma", verbose=False)
