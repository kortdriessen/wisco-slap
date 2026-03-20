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

__all__ = ["ephys", "vid", "anno"]
