"""NaN-aware analysis utilities.

Carries validity (per-timepoint, per-syn) through analyses that span scopex
DataArrays, hypnograms, and event arrays.

The motivating problem is that NaNs in scopex traces represent dead time
(synapse not being scanned, sample rejected, gap between trials) — and most
analyses need to either (a) exclude NaN timepoints from a denominator, or
(b) avoid spanning long NaN gaps when picking out windows of "real" data.
This submodule provides a single canonical way to do both.

Public API:

- :func:`validity_mask` — derive a boolean mask from a scopex DataArray.
- :func:`attach_validity` — stash a mask on the array as an ``is_valid`` coord.
- :func:`resolve_mask` — internal mask-resolution policy used by consumers.
- :func:`valid_duration` — seconds of valid time in a window.
- :func:`validity_intervals` — contiguous-True runs of a mask.
- :func:`valid_state_intervals` — runs of (hypno-state ∧ valid).
- :func:`valid_state_epochs` — NaN-aware fixed-length-epoch tiling.
- :func:`valid_duration_per_state` — per-state valid duration.
- :func:`event_rate_per_state` — NaN-aware event rate per state.

Default policy: when collapsing a 2-D ``(syn_id, time)`` array to a 1-D
validity mask, a timepoint is considered valid only if **every** synapse is
non-NaN (``mode='all'``). Override with ``mode='any'`` or ``mode='per_syn'``.
For most population-level analyses, ``'all'`` is the right default — it
ensures every synapse contributes equally to whatever statistic is being
pooled. Pre-filter the array to your analysis subset (e.g. one DMD, one
soma, one set of dendrites) *before* calling :func:`validity_mask` so the
mask reflects the syns you care about.

See also the longer-form rationale in
``/home/driessen2/.claude/plans/warm-doodling-horizon.md`` (the design plan
that produced this module).
"""

from __future__ import annotations

from .events import (
    event_rate_per_state as event_rate_per_state,
    valid_duration_per_state as valid_duration_per_state,
)
from .hypno import (
    add_valid_duration as add_valid_duration,
    valid_state_epochs as valid_state_epochs,
    valid_state_intervals as valid_state_intervals,
)
from .mask import (
    attach_validity as attach_validity,
    resolve_mask as resolve_mask,
    valid_duration as valid_duration,
    validity_intervals as validity_intervals,
    validity_mask as validity_mask,
)

__all__ = [
    "add_valid_duration",
    "attach_validity",
    "event_rate_per_state",
    "resolve_mask",
    "valid_duration",
    "valid_duration_per_state",
    "valid_state_epochs",
    "valid_state_intervals",
    "validity_intervals",
    "validity_mask",
]
