"""Prevent unintended cohort exclusions and non-NREM waves in batch plots."""

from types import SimpleNamespace
import warnings

import polars as pl

from wisco_slap.pns.single_soma_sw_locking import existing_single_soma_members
from wisco_slap.viz.overview_and_summary_plots.single_soma import _single_soma_sw_locking as sw


def test_detect_calibrates_on_sws_and_drops_other_stages(monkeypatch):
    class Waves:
        def __init__(self, frame):
            self.df = frame

        def keep_stage(self, stage):
            return Waves(self.df.filter(pl.col("stage") == stage))

    def detect(**kwargs):
        assert kwargs["compute_on_stages"] == ["NREM SWS"]
        assert kwargs["polarity"] == "negative"
        assert kwargs["dur_neg"] == (0.15, 1.0)
        assert kwargs["amp_threshold_params"] == {"k": 2.5}
        return Waves(pl.DataFrame({"stage": ["NREM SWS", "NREM sigma", "Quiet wake"],
                                   "trough_time": [1., 2., 3.]}))

    monkeypatch.setattr(sw.epy.wavedet, "detect_slow_waves", detect)
    hypno = SimpleNamespace(df=pl.DataFrame({"state": ["NREM SWS"]}))
    result = sw.detect_stage_waves(None, hypno)
    assert result.df["trough_time"].to_list() == [1.]


def test_absent_sws_never_calibrates_on_other_stages(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("must not detect without calibration stage")

    monkeypatch.setattr(sw.epy.wavedet, "detect_slow_waves", forbidden)
    hypno = SimpleNamespace(df=pl.DataFrame({"state": ["NREM sigma", "Quiet wake"]}))
    assert sw.detect_stage_waves(None, hypno) is None


def test_sws_outside_eeg_support_rejects_detector_fallback(monkeypatch):
    def fallback(**kwargs):
        warnings.warn("compute_on_stages=['NREM SWS'] matched no samples in the hypnogram; "
                      "falling back to whole-recording parameter estimation.", stacklevel=2)
        raise AssertionError("fallback should have been interrupted")

    monkeypatch.setattr(sw.epy.wavedet, "detect_slow_waves", fallback)
    hypno = SimpleNamespace(df=pl.DataFrame({"state": ["NREM SWS"]}))
    assert sw.detect_stage_waves(None, hypno) is None


def test_membership_is_union_of_source_folders_and_excludes_own_output(tmp_path):
    first = "m--exp_1--loc_A--acq_1--soma1"
    second = "m--exp_1--loc_A--acq_1--soma2"
    generated_only = "m--exp_2--loc_B--acq_1--soma1"
    for folder, members in [("single_soma_overview", [first]),
                            ("single_soma_corr_overview", [first, second]),
                            ("single_soma_sw_locking", [generated_only])]:
        (tmp_path / folder).mkdir()
        for member in members:
            (tmp_path / folder / f"{member}.png").touch()
    assert existing_single_soma_members(tmp_path) == [tuple(x.split("--")) for x in [first, second]]
