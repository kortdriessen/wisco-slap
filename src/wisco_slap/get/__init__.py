"""Convenience loading functions, accessible as ``wis.get.<function>(...)``."""

from ._annotations import hypno_csv as hypno_csv
from ._get_event_detection import denoised_noise_std as denoised_noise_std
from ._get_event_detection import glu_events_basic as glu_events_basic
from ._get_event_detection import (
    glu_events_basic_denoised as glu_events_basic_denoised,
)
from ._get_event_detection import matchFilt_noise_std as matchFilt_noise_std
from ._get_event_detection import matchFilt_traces as matchFilt_traces
from ._get_scopex import merge_syn_info_to_scopex as merge_syn_info_to_scopex
from ._get_scopex import roi_F as roi_F
from ._get_scopex import syn_dF as syn_dF
from ._get_scopex import syn_F0 as syn_F0
from ._get_syn_info import synid_labels as synid_labels
from ._get_sync_block_dat import ephys as ephys
from ._get_sync_block_dat import eye_metrics as eye_metrics
from ._get_sync_block_dat import eye_metrics_xa as eye_metrics_xa
from ._get_sync_block_dat import frame_times as frame_times
from ._get_sync_block_dat import whisk_df as whisk_df
from ._get_sync_block_dat import whisk_xa as whisk_xa

__all__ = [
    "denoised_noise_std",
    "ephys",
    "eye_metrics",
    "eye_metrics_xa",
    "frame_times",
    "glu_events_basic",
    "glu_events_basic_denoised",
    "matchFilt_noise_std",
    "matchFilt_traces",
    "merge_syn_info_to_scopex",
    "roi_F",
    "syn_dF",
    "syn_F0",
    "synid_labels",
    "whisk_df",
    "whisk_xa",
    "hypno_csv",
]
