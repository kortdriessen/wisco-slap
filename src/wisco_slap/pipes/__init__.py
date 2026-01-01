"""
subpackage for finalized pipelines for SLAP2 experiments
In general, pipelines should be functions that do things across subjects,
and _underscore functions that help those functions do their thing.
"""

from . import annotation_materials as annotation_materials
from . import bayglutev as bayglutev
from . import exp_info as exp_info
from . import score_mi as score_mi
from . import sleepscore as sleepscore
from . import syn_id as syn_id
from . import synmovies as synmovies
from . import trace_gen as trace_gen
from . import videos as videos
from ._auto_slapscore_model_code import infer_sleep_hsmm, train_sleep_hsmm

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
    "synmovies",
    "bayglutev",
]
