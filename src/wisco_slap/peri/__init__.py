"""
subpackage for dealing with peripherals (ephys, video, syncrhonization, etc.)
"""

from . import (
    ephys as ephys,
)
from . import (
    sync as sync,
)
from . import (
    vid as vid,
)

from . import (
    anno as anno,
)

__all__ = ["ephys", "sync", "vid", "anno"]
