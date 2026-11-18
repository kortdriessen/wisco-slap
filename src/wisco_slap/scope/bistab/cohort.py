"""Run one bistab analysis over a set of somas, with an on-disk result cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import polars as pl

from ... import defs
from .calcium import ca_responses, ca_trigger_average
from .compartments import depression_check, local_detection, participation
from .core import DEFAULTS, AcqData, Detector, refine_edges
from .data import load_acq
from .nulls import alternation_test, chain_test, episode_coupling, shuffle_rates
from .summary import event_rates, period_stats, testable_by_bin

#: Bump when an analysis function changes, so every cached cohort result rebuilds.
ANALYSIS_VERSION = 1
#: Per-analysis revision numbers (bump one to rebuild just that analysis).
#: 2026-10-09: refine_edges searches no further than the midpoint to a neighbour.
FN_VERSIONS = {"refine_edges": 2, "period_stats": 2}

#: Analyses :func:`cohort` can run: name -> (function, runs on AcqData rather than Detector).
ANALYSES: dict[str, tuple[Callable[..., pl.DataFrame], bool]] = {
    "shuffle_rates": (shuffle_rates, False),
    "alternation_test": (alternation_test, False),
    "chain_test": (chain_test, False),
    "episode_coupling": (episode_coupling, False),
    "participation": (participation, True),
    "local_detection": (local_detection, True),
    "event_rates": (event_rates, False),
    "ca_responses": (ca_responses, False),
    "ca_trigger_average": (ca_trigger_average, False),
    "refine_edges": (refine_edges, False),
    "depression_check": (depression_check, False),
    "period_stats": (period_stats, False),
    "testable_by_bin": (testable_by_bin, False),
}


def cohort(
    analysis: str,
    acqs: Mapping[str, Sequence[str]],
    *,
    duration_s: float | None = None,
    loader: Callable[[str], AcqData] | None = None,
    cache_dir: str | Path | None = None,
    identity: Mapping | None = None,
    rebuild: bool = False,
    **kwargs,
) -> pl.DataFrame:
    """Run one analysis on every soma in ``acqs``; cached on disk.

    Parameters
    ----------
    analysis
        A name in :data:`ANALYSES` (e.g. ``"alternation_test"``, ``"participation"``).
    acqs
        ``{key: (subject, exp, loc, acq, soma)}``. ``key`` becomes the ``acq``
        column of the result.
    duration_s
        Analyse the first ``duration_s`` seconds of each acquisition
        (``None`` = all); passed to :func:`load_acq`.
    loader
        ``key -> AcqData``; default :func:`load_acq` with ``duration_s``.
    cache_dir
        Where results are cached; default ``<bistab_cache_root>/_cohorts``.
    identity
        What identifies the set of somas in the cache key; default the
        ``acqs`` mapping and ``duration_s``. Only override to keep an existing
        cache valid (the synaptic_bistability thread does).
    rebuild
        Recompute even when a cached result exists.
    **kwargs
        Passed to the analysis (detector settings default to :data:`DEFAULTS`).

    The cache key is the analysis, its arguments, :data:`DEFAULTS`,
    :data:`ANALYSIS_VERSION` / :data:`FN_VERSIONS` and ``identity``. It does
    not see the per-soma input caches: after rebuilding those (new ExSum,
    relabelled synapses), pass ``rebuild=True`` here too.
    """
    if analysis not in ANALYSES:
        raise KeyError(f"unknown analysis {analysis!r}; choose from {sorted(ANALYSES)}")
    fn, on_acq = ANALYSES[analysis]
    if identity is None:
        identity = {
            "acqs": {k: list(v) for k, v in acqs.items()},
            "duration_s": duration_s,
        }
    if loader is None:

        def loader(key: str) -> AcqData:
            return load_acq(*acqs[key], duration_s=duration_s, key=key)

    spec_d = dict(a=analysis, kw=kwargs, d=DEFAULTS, v=ANALYSIS_VERSION, **identity)
    if analysis in FN_VERSIONS:  # per-analysis revisions (only rebuilds that analysis)
        spec_d["fv"] = FN_VERSIONS[analysis]
    spec = json.dumps(spec_d, sort_keys=True, default=str)
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(defs.bistab_cache_root) / "_cohorts"
    )
    path = root / f"{analysis}__{hashlib.sha1(spec.encode()).hexdigest()[:10]}.parquet"
    if path.exists() and not rebuild:
        return pl.read_parquet(path)
    parts = []
    for key in acqs:
        acq = loader(key)
        obj = acq if on_acq else Detector(acq)
        parts.append(fn(obj, **kwargs).with_columns(pl.lit(key).alias("acq")))
        del obj, acq
    df = pl.concat(parts, how="diagonal_relaxed")
    root.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    tmp.replace(path)
    return df
