"""Generate a whole set of overview plots across every member of the set.

``generate_plot_set(set)`` loops over the members of a plot set and, for each
plot registered in the matching subpackage of
``wis.viz.overview_and_summary_plots``, calls its plot function:

- ``"acq_level"`` — one plot per acquisition listed in
  ``acquisition_master.yaml``; member ids are ``(subject, exp, loc, acq)``.
- ``"single_soma"`` — one plot per member of the ``single_soma`` subgroup
  (``wis.meta.subgroups.single_soma``, i.e. the inclusion criteria applied to
  the current valid acquisitions); member ids are
  ``(subject, exp, loc, acq, soma_id)``.

Failure handling: if a plot raises for a member, a **blank** figure is
written at the plot's normal output path so the gap is visible when browsing,
and the full traceback is written beside it as
``<subject>--<exp>--<loc>--<acq>[--<soma_id>]--ERROR.txt``. An ``--ERROR.txt``
sidecar marks that plot as "still broken": it is retried on every run
regardless of ``overwrite``, and deleted as soon as the plot succeeds.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import wisco_slap as wis

ERROR_SUFFIX = "--ERROR.txt"

#: Plot sets known to the generator, in the order they are documented.
SET_NAMES: tuple[str, ...] = ("acq_level", "single_soma")
#: Columns of the returned table that hold the member id, in id order. Sets
#: whose ids are shorter (``acq_level`` has no soma) leave the tail empty.
ID_COLUMNS: tuple[str, ...] = ("subject", "exp", "loc", "acq", "soma_id")

_STATUS_GENERATED = "generated"
_STATUS_SKIPPED = "skipped"
_STATUS_ERROR = "error"


def error_path(plot_path: Path) -> Path:
    """Return the ``--ERROR.txt`` sidecar path for a plot output path."""
    return plot_path.with_name(f"{plot_path.stem}{ERROR_SUFFIX}")


def _all_acquisitions() -> list[tuple[str, ...]]:
    """Every ``(subject, exp, loc, acq)`` in ``acquisition_master.yaml``."""
    acqs: list[tuple[str, ...]] = []
    for subject, exps in wis.meta.get.acq_master().items():
        for exp, loc_acqs in exps.items():
            for loc_acq in loc_acqs:
                loc, acq = loc_acq.split("--")
                acqs.append((subject, exp, loc, acq))
    return acqs


def _single_soma_members() -> list[tuple[str, ...]]:
    """Every ``(subject, exp, loc, acq, soma_id)`` of the single-soma subgroup.

    Goes through ``wis.meta.subgroups.load_subgroup_set``, which recomputes the
    subgroup's inclusion criteria against the current data on every call — so
    the plot set and any analysis loading the subgroup always see the same
    members.
    """
    # imported lazily: the subgroup generator imports wisco_slap at module level
    from wisco_slap.meta.subgroups import load_subgroup_set

    return [tuple(m.split("--")) for m in load_subgroup_set("single_soma")]


def set_members(set: str) -> list[tuple[str, ...]]:
    """Return the member ids the plot set ``set`` is generated over.

    Raises
    ------
    ValueError
        If ``set`` is not one of :data:`SET_NAMES`.
    """
    if set == "acq_level":
        return _all_acquisitions()
    if set == "single_soma":
        return _single_soma_members()
    raise ValueError(f"Unknown plot set {set!r}; known sets: {list(SET_NAMES)}")


def _write_blank_plot(out_path: Path, title: str, err_path: Path) -> None:
    """Save a figure with no axes at ``out_path`` (a visible placeholder)."""
    fig = plt.figure(figsize=(28, 14))
    fig.suptitle(f"{title}\nplot generation FAILED — see {err_path.name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=50)
    plt.close(fig)


def _close_new_figures(fignums_before: list[int]) -> None:
    for num in plt.get_fignums():
        if num not in fignums_before:
            plt.close(num)


def _generate_one(
    spec: wis.viz.overview_and_summary_plots.PlotSpec,
    ids: tuple[str, ...],
    overwrite: bool,
) -> dict[str, str]:
    """Generate (or skip) one plot for one member; never raises."""
    out_path = spec.path(*ids)
    err_path = error_path(out_path)
    row = {"plot": spec.name}
    row.update(dict(zip(ID_COLUMNS, (*ids, *[""] * (len(ID_COLUMNS) - len(ids))))))
    row.update({"path": str(out_path), "error": ""})

    if not overwrite and out_path.exists() and not err_path.exists():
        return {**row, "status": _STATUS_SKIPPED}

    fignums_before = plt.get_fignums()
    try:
        # The generator has already decided to (re)build, so force the save.
        spec.plot(*ids, overwrite=True)
    except Exception as exc:
        _close_new_figures(fignums_before)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(traceback.format_exc())
        _write_blank_plot(out_path, " / ".join(ids), err_path)
        return {**row, "status": _STATUS_ERROR, "error": f"{type(exc).__name__}: {exc}"}

    _close_new_figures(fignums_before)
    err_path.unlink(missing_ok=True)
    return {**row, "status": _STATUS_GENERATED}


def generate_plot_set(set: str = "acq_level", overwrite: bool = False) -> pl.DataFrame:
    """Generate every plot in a plot set for every member of that set.

    Parameters
    ----------
    set
        Which plot set to generate: ``"acq_level"`` (every acquisition in the
        master list) or ``"single_soma"`` (every member of the single-soma
        subgroup). See :func:`set_members`.
    overwrite
        If ``False`` (default), plots whose PNG already exists are skipped —
        unless an ``--ERROR.txt`` sidecar sits beside it, in which case the
        plot is retried. If ``True``, every plot is rebuilt.

    Returns
    -------
    pl.DataFrame
        One row per (plot, member) with columns ``plot, subject, exp, loc,
        acq, soma_id, status, path, error``; ``status`` is ``generated``,
        ``skipped`` or ``error``; ``soma_id`` is empty for ``acq_level``. A
        summary is also printed.

    Raises
    ------
    ValueError
        If ``set`` is not one of :data:`SET_NAMES`.
    """
    members = set_members(set)
    specs = wis.viz.overview_and_summary_plots.plot_specs(set)
    print(
        f"Generating {len(specs)} '{set}' plot(s) x {len(members)} members "
        f"(overwrite={overwrite}): {[s.name for s in specs]}"
    )

    rows: list[dict[str, str]] = []
    with plt.ioff():
        for spec in specs:
            for ids in members:
                row = _generate_one(spec, ids, overwrite)
                rows.append(row)
                tag = "--".join(ids)
                if row["status"] == _STATUS_ERROR:
                    print(f"  [{spec.name}] {tag}: ERROR — {row['error']}")
                elif row["status"] == _STATUS_GENERATED:
                    print(f"  [{spec.name}] {tag}: generated")

    df = pl.DataFrame(
        rows,
        schema={
            "plot": pl.Utf8,
            **{col: pl.Utf8 for col in ID_COLUMNS},
            "status": pl.Utf8,
            "path": pl.Utf8,
            "error": pl.Utf8,
        },
    )
    counts = {
        status: int((df["status"] == status).sum())
        for status in (_STATUS_GENERATED, _STATUS_SKIPPED, _STATUS_ERROR)
    }
    print(
        f"Done: {counts[_STATUS_GENERATED]} generated, "
        f"{counts[_STATUS_SKIPPED]} skipped, {counts[_STATUS_ERROR]} errors."
    )
    if counts[_STATUS_ERROR]:
        print("Errors (see the --ERROR.txt beside each plot):")
        for r in df.filter(pl.col("status") == _STATUS_ERROR).iter_rows(named=True):
            tag = "--".join(r[col] for col in ID_COLUMNS if r[col])
            print(f"  [{r['plot']}] {tag}: {r['error']}")
    return df
