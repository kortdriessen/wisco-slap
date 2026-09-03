"""Convenience loading functions, accessible as ``wis.get.<function>(...)``."""

from ._get_hypnos import acq_sleep_coverage as acq_sleep_coverage
from ._get_hypnos import acq_sleep_table as acq_sleep_table
from ._get_hypnos import acq_sleep_table_markdown as acq_sleep_table_markdown
from ._get_hypnos import bout_hypno as bout_hypno
from ._get_hypnos import hypno_csv as hypno_csv
from ._get_hypnos import sb_bout_hypno as sb_bout_hypno
from ._get_hypnos import write_acq_sleep_table as write_acq_sleep_table
from ._get_scopex import merge_syn_info_to_scopex as merge_syn_info_to_scopex
from ._get_scopex import roi_F as roi_F
from ._get_scopex import syn_dF as syn_dF
from ._get_scopex import syn_F0 as syn_F0
from ._get_shaft_ca import shaft_ca as shaft_ca
from ._get_syn_info import soma_SynIDs as soma_SynIDs
from ._get_syn_info import synid_labels as synid_labels
from ._get_sync_block_dat import ephys as ephys
from ._get_sync_block_dat import eye_metrics as eye_metrics
from ._get_sync_block_dat import eye_metrics_xa as eye_metrics_xa
from ._get_sync_block_dat import frame_times as frame_times
from ._get_sync_block_dat import whisk_df as whisk_df
from ._get_sync_block_dat import whisk_xa as whisk_xa
from ._standard_loading_routines import fetch_data_pipe as fetch_data_pipe
from ._utils import combine_scopex_arrays as combine_scopex_arrays

__all__ = [
    "ephys",
    "shaft_ca",
    "eye_metrics",
    "eye_metrics_xa",
    "frame_times",
    "merge_syn_info_to_scopex",
    "roi_F",
    "syn_dF",
    "syn_F0",
    "synid_labels",
    "whisk_df",
    "whisk_xa",
    "hypno_csv",
    "bout_hypno",
    "sb_bout_hypno",
    "acq_sleep_coverage",
    "acq_sleep_table",
    "acq_sleep_table_markdown",
    "write_acq_sleep_table",
    "combine_scopex_arrays",
    "fetch_data_pipe",
    "soma_SynIDs",
]
