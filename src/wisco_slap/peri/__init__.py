"""
subpackage for dealing with peripherals (ephys, video, syncrhonization, etc.)
"""

from . import (
    anno as anno,
)
from . import (
    ephys as ephys,
)
from . import (
    vid as vid,
)
from .anno import load_auto_hypno as load_auto_hypno

__all__ = ["ephys", "vid", "anno", "load_auto_hypno"]
