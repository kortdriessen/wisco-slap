"""State-conditioned pairwise correlation analysis for SLAP2 synapses.

The original three public functions remain at this package's top level:

- :func:`pairwise_pearson_corr` — pairwise-complete-observations Pearson r
- :func:`pairwise_pearson_corr_by_bouts` — per-bout r + simple mean
- :func:`plot_slap2_corr_matrix` — synapse-by-synapse correlation matrix plot

New: a long-form, sleep-state-aware pipeline. The typical entry points are
:func:`build_state_corr_table` (single soma) /
:func:`build_state_corr_table_multi` (many somas), then
:func:`stratified_summaries` and :func:`mixed_state_test` for inference.

Temporal-dynamics extensions live in :mod:`temporal`, :mod:`ensemble`, and
:mod:`lagged`. They power the temporal_corr notebook and answer questions
about within-bout drift, cumulative-state-time dependence, ensemble
stability, and lagged co-activation.
"""

from .aggregate import (
    aggregate_fisher_z,
    aggregate_pooled_sums,
    aggregate_simple_mean,
)
from ...util.validity.hypno import merge_brief_breaking_bouts
from .bouts import (
    all_segments_in_state,
    fixed_valid_bouts,
    random_subsample_bouts,
    state_hypno_bouts,
)
from .core import (
    _to_combined,
    pairwise_pearson_corr,
    pairwise_pearson_corr_by_bouts,
)
from .ensemble import (
    ensemble_subspace_similarity,
    pair_rank_stability,
    per_bout_pair_r_vectors,
    per_bout_pair_r_vectors_multi,
    per_bout_pca,
    per_bout_pca_multi,
)
from .lagged import lagged_pairwise_corr, lagged_pairwise_corr_multi
from .outliers import bout_level_synchrony, outlier_pairs
from .plot import (
    LabelOpt,
    _resolve_label,
    _safe_str,
    plot_bout_synchrony_timeline,
    plot_outlier_pairs,
    plot_pair_distribution,
    plot_slap2_corr_matrix,
    plot_state_clock_trace,
    plot_state_onset_aligned,
    plot_state_pair_matrices,
    plot_stratum_paired,
    plot_within_bout_drift,
)
from .state_compare import (
    StateCorrTableBundle,
    _load_dn,
    build_state_corr_table,
    build_state_corr_table_multi,
    stratified_summaries,
)
from .stats import (
    cluster_bootstrap_state,
    fisher_z_meta_state_test,
    mixed_state_test,
    paired_state_test_per_pair,
    subject_level_state_test,
)
from .temporal import (
    annotate_bout_temporal_context,
    head_tail_bout_drift,
    head_tail_bout_drift_multi,
    sliding_window_corr_in_bout,
    state_clock_table,
    state_clock_table_multi,
    state_onset_aligned_synchrony,
    state_onset_aligned_synchrony_multi,
    within_bout_correlation_timeline,
    within_bout_correlation_timeline_multi,
)
from .time_evolution import (
    build_state_corr_table_periods,
    build_state_corr_table_periods_multi,
    split_bouts_into_periods,
)

__all__ = [
    # Original
    "pairwise_pearson_corr",
    "pairwise_pearson_corr_by_bouts",
    "plot_slap2_corr_matrix",
    # Bouts
    "fixed_valid_bouts",
    "all_segments_in_state",
    "state_hypno_bouts",
    "random_subsample_bouts",
    "merge_brief_breaking_bouts",
    # Aggregation
    "aggregate_simple_mean",
    "aggregate_fisher_z",
    "aggregate_pooled_sums",
    # State comparison
    "StateCorrTableBundle",
    "build_state_corr_table",
    "build_state_corr_table_multi",
    "stratified_summaries",
    # Time evolution (early/middle/late periods)
    "build_state_corr_table_periods",
    "build_state_corr_table_periods_multi",
    "split_bouts_into_periods",
    # Stats
    "paired_state_test_per_pair",
    "mixed_state_test",
    "fisher_z_meta_state_test",
    "cluster_bootstrap_state",
    "subject_level_state_test",
    # Outliers
    "outlier_pairs",
    "bout_level_synchrony",
    # Plots
    "plot_state_pair_matrices",
    "plot_pair_distribution",
    "plot_stratum_paired",
    "plot_outlier_pairs",
    "plot_bout_synchrony_timeline",
    "plot_within_bout_drift",
    "plot_state_clock_trace",
    "plot_state_onset_aligned",
    # Temporal dynamics
    "annotate_bout_temporal_context",
    "sliding_window_corr_in_bout",
    "within_bout_correlation_timeline",
    "within_bout_correlation_timeline_multi",
    "state_clock_table",
    "state_clock_table_multi",
    "state_onset_aligned_synchrony",
    "state_onset_aligned_synchrony_multi",
    "head_tail_bout_drift",
    "head_tail_bout_drift_multi",
    # Ensemble & rank stability
    "per_bout_pair_r_vectors",
    "per_bout_pair_r_vectors_multi",
    "pair_rank_stability",
    "per_bout_pca",
    "per_bout_pca_multi",
    "ensemble_subspace_similarity",
    # Lagged
    "lagged_pairwise_corr",
    "lagged_pairwise_corr_multi",
    # Internal but commonly used in notebooks
    "_to_combined",
    "_load_dn",
    "LabelOpt",
    "_resolve_label",
    "_safe_str",
]
