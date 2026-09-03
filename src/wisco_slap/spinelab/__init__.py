"""Spine-labelling dataset generation and measurement.

Isolated, untracked pipeline around the standalone ``spine-labeller`` app:

- :func:`generate_spine_labelling_dataset` turns one acquisition's mirrored
  ExperimentSummary into a self-contained dataset folder that the labeller
  can open on any machine (by default at :func:`spinelab_dataset_dir`, i.e.
  ``<anmat_root>/<subject>/<exp>/spinelab/<loc>--<acq>``);
- :func:`measure_dataset` runs the Wright-style inactive-frame spine-size
  measurement on such a folder once it contains saved annotations for at
  least one DMD (needs this machine's raw TIFFs and ExperimentSummary).

No staleness flags or audit rows — regenerate/remeasure explicitly.

Design note: ``F0_IMAGE_NOTE.md`` (next to this file) records why the
measurement uses inactive-frame mean projections rather than the
preprocessing pipeline's per-pixel F0 baseline.
"""

from . import activity as activity
from . import measure as measure
from . import qc as qc
from .datasets import (
    generate_spine_labelling_dataset as generate_spine_labelling_dataset,
)
from .datasets import spinelab_dataset_dir as spinelab_dataset_dir
from .measure import measure_dataset as measure_dataset

__all__ = [
    "activity",
    "generate_spine_labelling_dataset",
    "measure",
    "measure_dataset",
    "qc",
    "spinelab_dataset_dir",
]
