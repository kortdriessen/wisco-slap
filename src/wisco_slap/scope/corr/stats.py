"""Statistical tests on the long-form state-correlation table.

Four entry points, each consuming the output of
:func:`wisco_slap.scope.corr.build_state_corr_table` /
:func:`build_state_corr_table_multi`:

- :func:`paired_state_test_per_pair` — per-pair Δz with a sample-size-aware
  two-sample Fisher-z Z-test. Useful for ranking pairs and feeding the
  outlier-pair plot.
- :func:`mixed_state_test` — pair-level linear mixed-effects model via
  pymer4 (R/lme4). The primary group-level inference tool — keeps every
  pair as an observation and absorbs nesting via random intercepts.
- :func:`fisher_z_meta_state_test` — fixed/random-effects meta-analysis
  across clusters (default: per (subject, soma)). Quick parametric backstop
  to the mixed-effects model; doesn't require R/pymer4.
- :func:`cluster_bootstrap_state` — hierarchical bootstrap (subjects →
  somas → pairs). Distribution-free robustness check.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl
from scipy import stats as _sstats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pivot_pairs(table: pl.DataFrame, states: Sequence[str] = ("NREM", "Wake")) -> pl.DataFrame:
    """Wide-format per-pair table with z and se_z per state.

    One row per (cell_id, syn_i, syn_j), with columns
    ``z_<state>``, ``se_z_<state>``, ``n_<state>`` for each state plus
    pair metadata (``dmd_i, dmd_j, dend_i, dend_j, dend_type_i, dend_type_j,
    pair_type, same_type, dend_pair``). ``cell_id`` is the globally-unique
    cell key (``subject|exp|loc|acq|soma_id``); ``soma_id`` alone is *not*
    unique across recordings of the same animal.
    """
    keep_meta = [
        "cell_id", "recording_id",
        "subject", "exp", "loc", "acq", "soma_id",
        "syn_i", "syn_j",
        "dmd_i", "dmd_j",
        "dend_i", "dend_j",
        "dend_type_i", "dend_type_j",
        "pair_type", "same_type", "dend_pair",
    ]

    parts: list[pl.DataFrame] = []
    for state in states:
        sub = table.filter(pl.col("state") == state).select(
            *keep_meta,
            pl.col("z").alias(f"z_{state}"),
            pl.col("se_z").alias(f"se_z_{state}"),
            pl.col("n_samples").alias(f"n_{state}"),
        )
        parts.append(sub)

    if not parts:
        raise ValueError("No matching rows in table for any of the requested states.")
    out = parts[0]
    for sub in parts[1:]:
        out = out.join(sub, on=keep_meta, how="inner")
    return out


# ---------------------------------------------------------------------------
# Per-pair test
# ---------------------------------------------------------------------------


def paired_state_test_per_pair(
    table: pl.DataFrame,
    *,
    states: tuple[str, str] = ("NREM", "Wake"),
) -> pl.DataFrame:
    """Per-pair two-sample Fisher-z Z-test of Δz between two states.

    For each pair, computes
    ``Δz = z_<state0> − z_<state1>`` and
    ``SE(Δz) = sqrt(se_z_<state0>² + se_z_<state1>²)``,
    then a two-sided Z-test on ``Δz / SE(Δz)``.

    Requires ``se_z`` to be populated (use ``aggregation="fisher_z_weighted"``
    or ``"pooled_sums"`` when building the table). If ``se_z`` is missing
    (e.g. ``aggregation="simple_mean"``), the test columns are NaN.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form state-correlation table.
    states : tuple of two str
        Order of the contrast: ``Δz = z_states[0] − z_states[1]``. Default
        ``('NREM', 'Wake')`` so positive ``Δz`` means "more correlated in
        NREM."

    Returns
    -------
    pl.DataFrame
        One row per pair with columns: pair metadata, ``z_<state0>``,
        ``z_<state1>``, ``delta_z``, ``se_delta_z``, ``Z``, ``p_two_sided``.
    """
    s0, s1 = states
    wide = _pivot_pairs(table, states=states)

    delta_z = wide[f"z_{s0}"] - wide[f"z_{s1}"]
    se = (wide[f"se_z_{s0}"] ** 2 + wide[f"se_z_{s1}"] ** 2).sqrt()
    Z = delta_z / se
    # 2 * (1 - normal_cdf(|Z|))
    p = 2.0 * (1.0 - _sstats.norm.cdf(np.abs(Z.to_numpy())))

    return wide.with_columns(
        delta_z.alias("delta_z"),
        se.alias("se_delta_z"),
        Z.alias("Z"),
        pl.Series("p_two_sided", p, dtype=pl.Float64),
    ).sort("delta_z", descending=True)


# ---------------------------------------------------------------------------
# Mixed-effects test (pymer4)
# ---------------------------------------------------------------------------


def mixed_state_test(
    table: pl.DataFrame,
    *,
    stratum_filter: str | None = None,
    formula: str | None = None,
    states: tuple[str, ...] = ("NREM", "Wake"),
    use_precision_weights: bool = False,
    drop_within_dend: bool = True,
    reference_state: str = "Wake",
    verbose: bool = True,
    fit_kwargs: dict | None = None,
) -> dict:
    """Pair-level linear mixed-effects model via pymer4 (R/lme4).

    Default formula:
    ``z ~ is_NREM + (1|subject) + (1|cell_id) + (1|cell_id:dend_pair)``,
    where ``is_NREM`` is a 0/1 indicator (1 = NREM, 0 = Wake by default)
    and ``cell_id`` is the globally-unique cell identifier
    (``subject|exp|loc|acq|soma_id``). The fixed-effect coefficient on
    ``is_NREM`` is the NREM − Wake shift in Fisher-z.

    **Important**: ``cell_id`` is used (not ``soma_id``) because the same
    ``soma_id`` string (e.g. ``"soma1"``) can refer to different physical
    cells across acquisitions of the same animal. The random-intercept
    ``(1|cell_id)`` correctly groups the synapses imaged together as a
    single neuron in a single recording.

    Pass an explicit ``formula`` to interact with ``pair_type`` or
    ``same_type``, e.g.
    ``"z ~ is_NREM * pair_type + (1|subject) + (1|cell_id) +
    (1|cell_id:dend_pair)"``.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form table.
    stratum_filter : str | None
        If given, filter to rows where ``pair_type == stratum_filter`` OR
        ``same_type == stratum_filter`` before fitting.
    formula : str | None
        lme4 formula. ``None`` → default above. Use ``is_NREM`` (0/1) for
        the state contrast — the column is added before fitting.
    states : tuple of str
        Subset of states to include. Default ``('NREM', 'Wake')``.
    use_precision_weights : bool
        If True, pass ``weights = 1 / se_z**2`` (precision-weighted model).
        Requires ``se_z`` populated.
    drop_within_dend : bool
        Default True — drop ``pair_type == "within_dend"`` rows when no
        ``stratum_filter`` is given.
    reference_state : str
        Which state encodes ``is_NREM == 0``. Default ``"Wake"``.
    verbose : bool
        If True, print the fitted-model summary.
    fit_kwargs : dict | None
        Extra kwargs passed to ``model.fit()``. Useful values:
        ``conf_method="wald"`` or ``"boot"``, ``nboot=...``.

    Returns
    -------
    dict
        Keys: ``model`` (the fitted ``lmer`` instance), ``coefs`` (polars
        DataFrame of fixed effects with ``term, estimate, std_error,
        conf_low, conf_high, t_stat, df, p_value``), ``ranef_var``
        (polars DataFrame), ``data`` (the polars DataFrame fitted).
    """
    try:
        from pymer4.models import lmer
    except ImportError as e:
        raise ImportError(
            "pymer4 is required for mixed_state_test. Install with "
            "`uv add pymer4` in the slap_mi_2_sleep venv (R + lme4 also required)."
        ) from e

    df = table.filter(pl.col("state").is_in(list(states)))
    if stratum_filter is not None:
        df = df.filter(
            (pl.col("pair_type") == stratum_filter) | (pl.col("same_type") == stratum_filter)
        )
    elif drop_within_dend:
        df = df.filter(pl.col("pair_type") != "within_dend")

    df = df.filter(pl.col("z").is_not_null() & pl.col("z").is_finite())
    if df.height == 0:
        raise ValueError("Filter removed all rows — nothing to fit.")

    df = df.with_columns(
        (pl.col("state") != reference_state).cast(pl.Int8).alias("is_NREM")
    )

    if formula is None:
        formula = (
            "z ~ is_NREM + (1|subject) + (1|cell_id) + "
            "(1|cell_id:dend_pair)"
        )

    fit_kw = dict(summary=False)
    if fit_kwargs:
        fit_kw.update(fit_kwargs)

    if use_precision_weights:
        if df["se_z"].is_null().all():
            raise ValueError(
                "use_precision_weights=True but se_z is missing/NaN — "
                "use a Fisher-z or pooled-sums aggregation when building the table."
            )
        df = df.with_columns(
            (1.0 / pl.col("se_z") ** 2).alias("w")
        ).filter(
            pl.col("w").is_finite() & pl.col("w").is_not_null()
        )
        model = lmer(formula, data=df, weights="w")
    else:
        model = lmer(formula, data=df)

    model.fit(**fit_kw)
    if verbose:
        try:
            print(model.summary())
        except Exception:  # noqa: BLE001
            print(model.result_fit)

    return {
        "model": model,
        "coefs": model.result_fit,
        "ranef_var": model.ranef_var,
        "data": df,
    }


# ---------------------------------------------------------------------------
# Fisher-z meta-analysis
# ---------------------------------------------------------------------------


def fisher_z_meta_state_test(
    table: pl.DataFrame,
    *,
    stratum_filter: str | None = None,
    group_by: tuple[str, ...] = ("cell_id",),
    states: tuple[str, str] = ("NREM", "Wake"),
    drop_within_dend: bool = True,
) -> dict:
    """Meta-analysis of per-cluster mean Δz across groups.

    For each ``group_by`` cluster (default = one per ``cell_id``):
    1. compute per-pair ``Δz = z_state0 − z_state1`` and
       ``SE(Δz) = sqrt(se_z²(state0) + se_z²(state1))``;
    2. fixed-effect inverse-variance pool to a per-cluster mean Δz with SE.

    Then meta-analyse across clusters two ways:
    - **fixed-effect**: inverse-variance pooled mean Δz across clusters.
    - **random-effects** (DerSimonian–Laird): adds between-cluster variance
      ``τ²`` to the weights. More appropriate when clusters genuinely differ.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form table.
    stratum_filter : str | None
        See :func:`mixed_state_test`.
    group_by : tuple of str
        Cluster definition. Default ``("cell_id",)`` — one cluster per
        unique cell (``subject|exp|loc|acq|soma_id``). Use ``("subject",)``
        for the per-mouse test (which pools across cells from the same
        animal). Do **not** use ``("subject", "soma_id")`` — ``soma_id``
        is not unique across recordings.
    states : tuple of two str
        Default ``("NREM", "Wake")``.
    drop_within_dend : bool
        If True (default) and no ``stratum_filter`` given, drop within-dend
        pairs.

    Returns
    -------
    dict
        ``per_cluster``: pl.DataFrame with one row per cluster
        (``mean_delta_z``, ``se``, ``n_pairs``).
        ``fixed_effect``: dict with ``mean, se, Z, p, ci_low, ci_high, k``.
        ``random_effects``: dict with same keys plus ``tau2``.
    """
    s0, s1 = states
    df = table
    if stratum_filter is not None:
        df = df.filter(
            (pl.col("pair_type") == stratum_filter) | (pl.col("same_type") == stratum_filter)
        )
    elif drop_within_dend:
        df = df.filter(pl.col("pair_type") != "within_dend")

    wide = _pivot_pairs(df, states=states)
    wide = wide.with_columns(
        (pl.col(f"z_{s0}") - pl.col(f"z_{s1}")).alias("delta_z"),
        (pl.col(f"se_z_{s0}") ** 2 + pl.col(f"se_z_{s1}") ** 2).sqrt().alias("se_delta_z"),
    ).filter(
        pl.col("delta_z").is_not_null()
        & pl.col("delta_z").is_finite()
        & (pl.col("se_delta_z") > 0)
    )
    if wide.height == 0:
        raise ValueError("No usable rows after filtering — table missing se_z?")

    # Per-cluster fixed-effect pool: weights = 1 / se².
    per_cluster = (
        wide.with_columns((1.0 / pl.col("se_delta_z") ** 2).alias("w"))
        .group_by(list(group_by))
        .agg(
            ((pl.col("delta_z") * pl.col("w")).sum() / pl.col("w").sum()).alias("mean_delta_z"),
            (1.0 / pl.col("w").sum()).sqrt().alias("se"),
            pl.len().alias("n_pairs"),
        )
        .sort(*group_by)
    )

    yi = per_cluster["mean_delta_z"].to_numpy()
    se = per_cluster["se"].to_numpy()
    vi = se**2
    wi_fe = 1.0 / vi
    k = len(yi)

    mean_fe = float(np.sum(wi_fe * yi) / np.sum(wi_fe))
    se_fe = float(1.0 / np.sqrt(np.sum(wi_fe)))
    Z_fe = mean_fe / se_fe
    p_fe = float(2.0 * (1.0 - _sstats.norm.cdf(abs(Z_fe))))
    ci_lo_fe = mean_fe - 1.96 * se_fe
    ci_hi_fe = mean_fe + 1.96 * se_fe

    # Random-effects (DerSimonian–Laird).
    Q = float(np.sum(wi_fe * (yi - mean_fe) ** 2))
    df_dl = k - 1
    sum_w = float(np.sum(wi_fe))
    sum_w2 = float(np.sum(wi_fe**2))
    if df_dl > 0:
        tau2 = max(0.0, (Q - df_dl) / (sum_w - sum_w2 / sum_w))
    else:
        tau2 = 0.0
    wi_re = 1.0 / (vi + tau2)
    mean_re = float(np.sum(wi_re * yi) / np.sum(wi_re))
    se_re = float(1.0 / np.sqrt(np.sum(wi_re)))
    Z_re = mean_re / se_re
    p_re = float(2.0 * (1.0 - _sstats.norm.cdf(abs(Z_re))))
    ci_lo_re = mean_re - 1.96 * se_re
    ci_hi_re = mean_re + 1.96 * se_re

    return {
        "per_cluster": per_cluster,
        "fixed_effect": {
            "mean": mean_fe, "se": se_fe, "Z": Z_fe, "p": p_fe,
            "ci_low": ci_lo_fe, "ci_high": ci_hi_fe, "k": k,
        },
        "random_effects": {
            "mean": mean_re, "se": se_re, "Z": Z_re, "p": p_re,
            "ci_low": ci_lo_re, "ci_high": ci_hi_re, "k": k, "tau2": tau2,
        },
    }


# ---------------------------------------------------------------------------
# Hierarchical bootstrap
# ---------------------------------------------------------------------------


def cluster_bootstrap_state(
    table: pl.DataFrame,
    *,
    stratum_filter: str | None = None,
    states: tuple[str, str] = ("NREM", "Wake"),
    drop_within_dend: bool = True,
    n: int = 10_000,
    seed: int = 0,
) -> dict:
    """Hierarchical bootstrap of mean Δz across (subject → cell → pair).

    For each bootstrap iteration:
    1. Resample subjects with replacement.
    2. Within each, resample cells (``cell_id``) with replacement from that
       subject's cells. ``cell_id`` is unique per recording, so this
       respects the fact that one ``soma_id`` can be different cells
       across acquisitions.
    3. Within each, resample pairs (rows) with replacement.
    4. Compute the mean Δz across the resampled pairs.

    Returns the bootstrap distribution and a 95% percentile CI plus the
    observed point estimate.

    Parameters
    ----------
    table : pl.DataFrame
    stratum_filter : str | None
    states : tuple of two str
    drop_within_dend : bool
    n : int
        Number of bootstrap iterations. Default 10_000.
    seed : int

    Returns
    -------
    dict with keys ``observed, mean, ci_low, ci_high, frac_pos, samples``.
    """
    s0, s1 = states
    df = table
    if stratum_filter is not None:
        df = df.filter(
            (pl.col("pair_type") == stratum_filter) | (pl.col("same_type") == stratum_filter)
        )
    elif drop_within_dend:
        df = df.filter(pl.col("pair_type") != "within_dend")

    wide = _pivot_pairs(df, states=states)
    wide = wide.with_columns(
        (pl.col(f"z_{s0}") - pl.col(f"z_{s1}")).alias("delta_z"),
    ).filter(
        pl.col("delta_z").is_not_null() & pl.col("delta_z").is_finite()
    )
    if wide.height == 0:
        raise ValueError("No usable rows after filtering.")

    # Build subject → list[cell_id] → array(delta_z) lookup.
    # cell_id (= subject|exp|loc|acq|soma_id) is the proper "soma" key:
    # the same soma_id string can refer to different cells across recordings.
    subjects = wide["subject"].unique().sort().to_list()
    subj_to_cells: dict[str, list[str]] = {s: [] for s in subjects}
    cell_to_dz: dict[str, np.ndarray] = {}
    for grp in (
        wide.group_by(["subject", "cell_id"])
        .agg(pl.col("delta_z"))
        .iter_rows(named=True)
    ):
        cell = grp["cell_id"]
        cell_to_dz[cell] = np.asarray(grp["delta_z"], dtype=float)
        subj_to_cells[grp["subject"]].append(cell)

    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    for it in range(n):
        # Resample subjects.
        chosen_subjs = rng.choice(subjects, size=len(subjects), replace=True)
        deltas: list[float] = []
        for s in chosen_subjs:
            cells = subj_to_cells[s]
            if not cells:
                continue
            chosen_idx = rng.integers(0, len(cells), size=len(cells))
            for k in chosen_idx:
                arr = cell_to_dz[cells[k]]
                if arr.size == 0:
                    continue
                pair_idx = rng.integers(0, arr.size, size=arr.size)
                deltas.append(float(arr[pair_idx].mean()))
        samples[it] = float(np.mean(deltas)) if deltas else np.nan

    samples_clean = samples[~np.isnan(samples)]
    observed = float(wide["delta_z"].mean())
    return {
        "observed": observed,
        "mean": float(np.mean(samples_clean)),
        "ci_low": float(np.percentile(samples_clean, 2.5)),
        "ci_high": float(np.percentile(samples_clean, 97.5)),
        "frac_pos": float(np.mean(samples_clean > 0)),
        "samples": samples_clean,
    }


# ---------------------------------------------------------------------------
# Subject- / cell-level tests (the reviewer-friendly p-values)
# ---------------------------------------------------------------------------


def subject_level_state_test(
    table: pl.DataFrame,
    *,
    summarise_by: str = "cell_id",
    summary: str = "median",
    stratum_filter: str | None = None,
    states: tuple[str, str] = ("NREM", "Wake"),
    drop_within_dend: bool = True,
) -> dict:
    """Collapse pairs to one Δz per cluster, then run paired t and Wilcoxon.

    The "boring but defensible" headline test for animal experiments. Per
    cluster (default: per cell, set ``summarise_by="subject"`` for the
    per-mouse version), summarise pair Δz by mean or median, then run a
    paired t-test and a Wilcoxon signed-rank on the cluster-level Δz.

    Parameters
    ----------
    table : pl.DataFrame
        Long-form correlation table.
    summarise_by : str
        Column to group on. ``"cell_id"`` (default) collapses each cell
        to one Δz; ``"subject"`` collapses each animal to one Δz.
    summary : str
        ``"median"`` (default — robust to outlier pairs) or ``"mean"``.
    stratum_filter : str | None
        If set, filter rows where ``pair_type == stratum_filter`` OR
        ``same_type == stratum_filter`` OR ``soma_type_class ==
        stratum_filter`` first.
    states : tuple of two str
        ``Δz = z_states[0] - z_states[1]``. Default ``('NREM', 'Wake')``.
    drop_within_dend : bool
        Default True, ignored when ``stratum_filter`` is set.

    Returns
    -------
    dict with:
        ``per_cluster``  pl.DataFrame with one row per cluster:
            ``<summarise_by>``, NREM, Wake, ``delta_z``, ``n_pairs``.
        ``mean_delta_z``  mean across clusters
        ``sd_delta_z``    cluster-level SD
        ``n_clusters``    N
        ``t_stat``, ``t_df``, ``t_p``  paired t-test on the cluster Δz
        ``w_stat``, ``w_p``  Wilcoxon signed-rank
        ``ci_low``, ``ci_high``  95% t-distribution CI on the cluster mean
    """
    s0, s1 = states
    df = table
    if stratum_filter is not None:
        cols = ["pair_type", "same_type"]
        if "soma_type_class" in df.columns:
            cols.append("soma_type_class")
        expr = pl.col(cols[0]) == stratum_filter
        for c in cols[1:]:
            expr = expr | (pl.col(c) == stratum_filter)
        df = df.filter(expr)
    elif drop_within_dend:
        df = df.filter(pl.col("pair_type") != "within_dend")

    df = df.filter(pl.col("z").is_not_null() & pl.col("z").is_finite())

    # Pivot per pair, compute Δz per pair, then aggregate per cluster.
    keep = [summarise_by, "syn_i", "syn_j"]
    nrem = (
        df.filter(pl.col("state") == s0)
        .select(*keep, pl.col("z").alias(f"z_{s0}"))
    )
    wake = (
        df.filter(pl.col("state") == s1)
        .select(*keep, pl.col("z").alias(f"z_{s1}"))
    )
    pairs_wide = nrem.join(wake, on=keep, how="inner").with_columns(
        (pl.col(f"z_{s0}") - pl.col(f"z_{s1}")).alias("delta_z")
    )
    if pairs_wide.height == 0:
        raise ValueError("No usable pairs after filtering and joining states.")

    summary_expr = (
        pl.col("delta_z").median() if summary == "median" else pl.col("delta_z").mean()
    )
    nrem_summary = (
        pl.col(f"z_{s0}").median() if summary == "median" else pl.col(f"z_{s0}").mean()
    )
    wake_summary = (
        pl.col(f"z_{s1}").median() if summary == "median" else pl.col(f"z_{s1}").mean()
    )
    per_cluster = (
        pairs_wide.group_by(summarise_by)
        .agg(
            nrem_summary.alias(s0),
            wake_summary.alias(s1),
            summary_expr.alias("delta_z"),
            pl.len().alias("n_pairs"),
        )
        .sort(summarise_by)
    )

    deltas = per_cluster["delta_z"].to_numpy()
    n_clusters = len(deltas)
    if n_clusters < 2:
        raise ValueError(
            f"Need ≥2 clusters for a paired test, got {n_clusters} from "
            f"summarise_by={summarise_by!r}."
        )

    t_stat, t_p = _sstats.ttest_1samp(deltas, 0.0)
    t_df = n_clusters - 1
    sd = float(np.std(deltas, ddof=1))
    se = sd / np.sqrt(n_clusters)
    t_crit = float(_sstats.t.ppf(0.975, t_df))
    mean = float(np.mean(deltas))
    ci_low, ci_high = mean - t_crit * se, mean + t_crit * se

    # Wilcoxon — use 'wilcox' mode for exact small-sample p when possible.
    try:
        w_res = _sstats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
        w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "per_cluster": per_cluster,
        "summarise_by": summarise_by,
        "summary": summary,
        "mean_delta_z": mean,
        "sd_delta_z": sd,
        "n_clusters": n_clusters,
        "t_stat": float(t_stat),
        "t_df": t_df,
        "t_p": float(t_p),
        "w_stat": w_stat,
        "w_p": w_p,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
