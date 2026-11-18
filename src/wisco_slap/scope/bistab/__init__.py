"""Population ON/OFF detection and synaptic-bistability analyses.

Moved here on 2026-10-29 from
``slap_mi_2_sleep/clean_analysis_threads/synaptic_bistability/bistab.py``
(that thread's ``README.md`` holds the hypothesis, plan and results log; its
``detector_first_pass.qmd`` is the plain-language description of the detector).

The detector in one paragraph
-----------------------------
Glutamate events (pursuit catalog, ``z_shot >= z_min``) of every synapse of one
soma are binned at 10 ms. In a sliding window of ``window_s`` we count the
population's events (``N``) and compare that count with what we would expect
if every synapse kept firing at its *own* recent rate, *independently of the
others* (``E``, with the spread ``V`` that independent synapses would produce,
both estimated from each synapse's own activity over ``baseline_s``). The
surprise of a window is ``-log10 p`` of seeing so few (OFF) or so many (ON)
events under that independence model (negative-binomial approximation). Runs
of windows whose surprise exceeds ``edge`` and that contain at least one
window above ``seed`` become an OFF or ON period; the period spans the union
of its windows. ON and OFF are detected the same way and nothing in the
detector prefers alternation. A moment is only tested when every synapse of
the soma (both DMDs) is observable for the whole window (``min_valid = 1``).

Typical use
-----------
>>> import wisco_slap as wis
>>> bs = wis.scope.bistab
>>> acq = bs.load_acq("alnair", "exp_1", "loc_D", "acq_1", "soma1")  # cached
>>> det = bs.Detector(acq)
>>> periods = det.periods()                  # ON/OFF table at DEFAULTS
>>> bs.state_table(det)                      # rates / durations per state
>>> app, P = bs.build_view(det, eeg=bs.load_eeg(acq))   # Loupe + live Tuner
>>> bs.cohort("alternation_test", {"a1": ("alnair", "exp_1", "loc_D", "acq_1", "soma1")})

Every analysis takes a :class:`Detector` (or, for the dendrite analyses, an
:class:`AcqData`) plus detector settings as keyword arguments
(``z_min``, ``window_s``, ``baseline_s``, ``seed``, ``edge``; defaults in
:data:`DEFAULTS`). Random-number seeds of the null models are ``rng_seed``
(``seed`` is always the detector threshold). Rates always use *testable* time
as the denominator. All times are ephys (TDT-block) seconds.

Modules
-------
- :mod:`.core` — :class:`AcqData`, :class:`Detector`, :func:`refine_edges`, constants.
- :mod:`.data` — :func:`load_acq` (+ cache), :func:`load_eeg`, :func:`load_dff` / :func:`load_ca`.
- :mod:`.summary` — per-state tables, per-period measures, time trends.
- :mod:`.nulls` — independent-synapse shuffles, alternation / chain tests, jittered-period nulls, episode coupling.
- :mod:`.compartments` — dendrite participation, local (thinned) detection, depression check.
- :mod:`.calcium` — crosstalk-corrected synaptic calcium around glutamate events.
- :mod:`.cohort` — run any analysis over many somas with a parquet result cache.
- :mod:`.viz` — Loupe viewer (:func:`build_view`) and :func:`state_dots`.
"""

from .calcium import CA_POST, CA_PRE, XTALK_LAGS, ca_responses, ca_trigger_average
from .cohort import ANALYSES, ANALYSIS_VERSION, FN_VERSIONS, cohort
from .compartments import (
    dendrite_groups,
    depression_check,
    group_rate,
    local_detection,
    participation,
)
from .core import (
    BIN_S,
    CA_CHANNEL,
    DEFAULTS,
    GLU_CHANNEL,
    GLU_TAU_S,
    STATE_ORDER,
    STATE_SHORT,
    THIRDS,
    AcqData,
    Detector,
    refine_edges,
    state_at,
    third_of,
)
from .data import (
    CACHE_VERSION,
    acq_cache_dir,
    build_acq_cache,
    clear_acq_cache,
    load_acq,
    load_ca,
    load_dff,
    load_eeg,
)
from .nulls import (
    alternation_test,
    chain_test,
    chains,
    episode_coupling,
    jitter_periods,
    shuffle_rates,
)
from .summary import (
    SENSITIVITY,
    TREND_METRICS,
    event_rates,
    period_stats,
    state_table,
    summarize,
    testable_by_bin,
    testable_minutes,
    time_trends,
)
from .viz import SPAN_COLORS, build_view, state_dots

__all__ = [
    # data + detector
    "AcqData",
    "Detector",
    "load_acq",
    "build_acq_cache",
    "clear_acq_cache",
    "acq_cache_dir",
    "load_eeg",
    "load_dff",
    "load_ca",
    "refine_edges",
    "state_at",
    "third_of",
    # constants
    "DEFAULTS",
    "BIN_S",
    "GLU_CHANNEL",
    "CA_CHANNEL",
    "GLU_TAU_S",
    "STATE_ORDER",
    "STATE_SHORT",
    "THIRDS",
    "CACHE_VERSION",
    # summaries
    "summarize",
    "state_table",
    "testable_minutes",
    "testable_by_bin",
    "event_rates",
    "period_stats",
    "time_trends",
    "SENSITIVITY",
    "TREND_METRICS",
    # nulls / alternation
    "shuffle_rates",
    "alternation_test",
    "chains",
    "chain_test",
    "jitter_periods",
    "episode_coupling",
    # compartments
    "dendrite_groups",
    "participation",
    "group_rate",
    "local_detection",
    "depression_check",
    # calcium
    "ca_responses",
    "ca_trigger_average",
    "CA_PRE",
    "CA_POST",
    "XTALK_LAGS",
    # cohort runner
    "cohort",
    "ANALYSES",
    "ANALYSIS_VERSION",
    "FN_VERSIONS",
    # viewer + plots
    "build_view",
    "state_dots",
    "SPAN_COLORS",
]
