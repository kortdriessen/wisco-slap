"""
subpackage for dealing with data from SLAP2 microscope
"""

from . import _GLUT_PURSUIT_PIPELINE as pursuit
from . import anat as anat
from . import cascade as cascade
from . import corr as corr
from . import pro as pro
from . import rates as rates
from . import synfo as synfo
from . import viz as viz
from ._GLUT_EVENT_MAIN import master_event_detector
from ._GLUT_EVENT_SIMPLE import detect_basic_events

__all__ = [
    "anat",
    "cascade",
    "corr",
    "rates",
    "synfo",
    "pro",
    "viz",
    "pursuit",
    "master_event_detector",
    "detect_basic_events",
]
