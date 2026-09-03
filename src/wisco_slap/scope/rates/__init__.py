"""Per-synapse event rates from a ``master_event_detector`` ``peaks_df``.

The rate counterpart to :mod:`wisco_slap.scope.corr`: where ``corr`` answers
"how do these synapses co-vary?", ``rates`` answers "how often does each
synapse fire, and does that depend on sleep state / time / bout?".

Where :func:`wisco_slap.util.validity.event_rate_per_state` computes a rate
from the *boolean event array* (``final_peaks``), the functions here compute
rates from the *event table* (``peaks_df``) — the polars DataFrame returned by
:func:`wisco_slap.scope.master_event_detector`. Working from ``peaks_df`` has
two advantages:

1. **Filtering is respected.** Any subset the caller takes of ``peaks_df``
   (an SNR floor, a single dendrite, a synapse-type, an amplitude band) flows
   straight into the numerator — the count is simply the surviving rows.
2. **Metadata comes for free.** ``peaks_df`` carries every per-synapse
   coordinate; the outputs merge it back in so each row is self-describing.

The one thing ``peaks_df`` cannot supply is the **denominator**: how many
seconds of *valid* (non-NaN) data the detector actually saw. That has to come
from the ``is_valid`` coordinate on the scopex array the detection ran on —
which is exactly what ``final_peaks`` carries. So every function here takes
both ``peaks_df`` (numerator + metadata) and ``final_peaks`` (validity mask +
sample grid + the complete synapse universe, including synapses with zero
surviving events). The NaN-aware primitives underneath all of this live in
:mod:`wisco_slap.util.validity`.

**The denominator is always valid time, never wall-clock time.** A 20 s bout
that was only scanned for 10 s yields ``events / 10 s``. Numerator and
denominator are tied to the *same* set of samples: an event is counted only if
it lands on a sample that also counts toward the denominator, so the two can
never disagree.

Per-bout results also carry the hypnogram's own extra columns. Whatever you
attached per bout before calling — bandpower, elapsed time, valid duration,
bout-level annotations — is joined into the output of ``compute_rate`` with
``by='bout'`` by default, so a rate row is directly regressable against its
bout's context without a follow-up merge. Turn it off with
``join_hypno_context=False``. See :func:`compute_rate` for the collision rules.

Time-axis contract
------------------
``peaks_df['peak_time']``, ``final_peaks['time']`` and the hypnogram must all
be on the **same clock**. In this project that is TDT-block-relative seconds:
``wis.get.syn_dF`` applies ``ephys_offset`` by default, and ``wis.get.bout_hypno``
/ ``wis.get.acq_sleep_coverage`` return hypnograms on that same axis. As long
as ``peaks_df`` and ``final_peaks`` come from the *same* ``master_event_detector``
call and the hypnogram was loaded for the same acquisition, they line up. The
functions validate the ``peaks_df`` ↔ ``final_peaks`` pairing and raise if the
sample indices don't match.

Public API
----------
- :func:`compute_rate` — per-syn rate ``by='bout'`` or ``by='state'`` (the core).
- :func:`compute_rate_by_bout` / :func:`compute_rate_by_state` — thin named
  wrappers around :func:`compute_rate`.
- :func:`compute_rate_rolling` — sliding-window rate over time (for time courses).
- :func:`compute_rate_by_fraction` — rate in the first/middle/last N-th of each
  state or bout (equal *valid*-time bins).
- :func:`compute_rate_by_epoch` — rate per fixed-valid-duration epoch, tiled
  NaN-aware from each state's bouts (controls for unequal bout/valid durations).
"""

from __future__ import annotations

from .core import (
    compute_rate,
    compute_rate_by_bout,
    compute_rate_by_state,
)
from .epoch import compute_rate_by_epoch
from .fraction import compute_rate_by_fraction
from .rolling import compute_rate_rolling

__all__ = [
    # Core: by state / by bout
    "compute_rate",
    "compute_rate_by_bout",
    "compute_rate_by_state",
    # Time course
    "compute_rate_rolling",
    # Within-container fractions (early/middle/late)
    "compute_rate_by_fraction",
    # Fixed-valid-duration epochs
    "compute_rate_by_epoch",
]
