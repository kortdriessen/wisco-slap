"""
Utilities for working with SLAP2 data
"""

from . import (
    checks as checks,
)
from . import (
    matchfilt as matchfilt,
)
from . import (
    plot as plot,
)
from . import (
    snr as snr,
)
from . import (
    validity as validity,
)
from .core import *  # noqa: F403
from .snr import compute_snr as compute_snr
from .snr import lower_half_mad_sigma as lower_half_mad_sigma

__all__ = [
    "checks",
    "matchfilt",
    "plot",
    "snr",
    "validity",
    "compute_snr",
    "lower_half_mad_sigma",
]
