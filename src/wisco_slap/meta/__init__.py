"""
subpackage for dealing with peripherals (ephys, video, syncrhonization, etc.)
"""

from . import dmd_info as dmd_info
from . import epoch as epoch
from . import exsum_mirror as exsum_mirror
from . import get as get
from . import prepro_info as prepro_info
from . import status as status
from . import sync as sync
from ._update import update as update
from .get import acq_timing as get_acq_timing

__all__ = [
    "sync",
    "epoch",
    "prepro_info",
    "dmd_info",
    "exsum_mirror",
    "get",
    "status",
    "update",
    "get_acq_timing",
]
