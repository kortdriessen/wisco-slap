"""
subpackage for finalized pipelines for SLAP2 experiments
In general, pipelines should be functions that do things across subjects,
and _underscore functions that help those functions do their thing.
"""

from . import exp_info as exp_info
from . import annotation_materials as annotation_materials
from . import sleepscore as sleepscore
from . import videos as videos
from ._auto_slapscore_model_code import train_sleep_hsmm, infer_sleep_hsmm
from . import score_mi as score_mi
from . import syn_id as syn_id
from . import trace_gen as trace_gen

__all__ = [
    "exp_info",
    "annotation_materials",
    "sleepscore",
    "videos",
    "train_sleep_hsmm",
    "infer_sleep_hsmm",
    "score_mi",
    "syn_id",
    "trace_gen",
]
