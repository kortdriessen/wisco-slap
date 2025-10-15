"""
subpackage for finalized pipelines for SLAP2 experiments
In general, pipelines should be functions that do things across subjects,
and _underscore functions that help those functions do their thing.
"""

from . import exp_info as exp_info
from . import ref_images as ref_images
from . import sleepscore as sleepscore
from . import traces as traces
from . import videos as videos

__all__ = ["traces", "exp_info", "ref_images", "sleepscore", "videos"]
