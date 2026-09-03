"""Generator for the ``single_soma`` subgroup (see ``../subgroup_info.md``).

``single_soma_acqs`` applies the inclusion criteria to the current valid
acquisitions and returns the member ids; running this file as a script writes
``candidate_set/candidate_set.json`` and fills ``candidate_set/main_overview/``
with one PNG per candidate. The overview PNGs are the ``single_soma_overview``
plot of the ``single_soma`` plot set (``wis.viz.overview_and_summary_plots``),
generated through ``wis.pns.overview_plot_gen.generate_plot_set`` (which also
writes a blank placeholder + ``--ERROR.txt`` sidecar when a plot fails) and
copied here so the review set stays complete.
"""

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

import wisco_slap as wis


def write_main_overviews(
    candidates: list[str],
    overview_dir: Path,
    *,
    replace: bool = False,
) -> list[Path]:
    """Populate ``overview_dir`` with the single-soma overview of each candidate.

    Runs ``wis.pns.overview_plot_gen.generate_plot_set("single_soma",
    overwrite=replace)`` (idempotent: existing, non-errored plots are skipped
    unless ``replace``) and then copies each candidate's PNG from the plot
    set's output directory into ``overview_dir``. Failed plots arrive as the
    generator's blank placeholder, so there is always exactly one PNG per
    candidate; candidates the generator did not produce (it iterates the same
    ``single_soma_acqs`` selection, so this only happens if the data changed
    between the two calls) are reported and skipped.

    Returns
    -------
    list[Path]
        The PNG written (or refreshed) for each candidate, in candidate order.
    """
    wis.pns.overview_plot_gen.generate_plot_set("single_soma", overwrite=replace)
    overview_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for candidate in candidates:
        ids = candidate.split("--")
        src = wis.viz.overview_and_summary_plots.single_soma_overview_path(*ids)
        dst = overview_dir / f"{candidate}.png"
        if not src.exists():
            print(f"No overview was generated for {candidate}; expected {src}")
            continue
        shutil.copy2(src, dst)
        written.append(dst)
    print(f"Copied {len(written)} / {len(candidates)} overview PNGs to {overview_dir}")
    return written


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for overview generation."""
    parser = argparse.ArgumentParser(description="Generate single-soma overviews.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing overview PNGs instead of skipping them.",
    )
    return parser.parse_args()


def _soma_syn_stats(syn_info: pl.DataFrame, soma_id: str) -> tuple[int, int, int]:
    """Summarize the dendrite-assigned synapses of one soma.

    Parameters
    ----------
    syn_info : pl.DataFrame
        Output of :func:`wis.get.synid_labels` for one acquisition.
    soma_id : str
        Soma label (``"soma1"``, ``"soma2"``, ...).

    Returns
    -------
    tuple[int, int, int]
        ``(n_dmds, max_syns_on_one_dendrite, n_synapses)`` — the number of
        DMDs on which the soma has assigned synapses, the synapse count of its
        most populated dendrite, and its total assigned-synapse count. This is
        exactly the priority order used to break ties between somas.
    """
    soma_syns = syn_info.filter(
        (pl.col("soma-ID") == soma_id) & pl.col("dend-ID").is_not_null()
    )
    if soma_syns.is_empty():
        return 0, 0, 0
    n_dmds = soma_syns["dmd"].n_unique()
    max_dend = soma_syns.group_by("dend-ID").len()["len"].max()
    return n_dmds, int(max_dend), len(soma_syns)


def single_soma_acqs(
    min_n_synapses: int = 40, min_duration_seconds: float = 1200
) -> list[str]:
    """Return member IDs for somas passing the single-soma inclusion criteria.

    At most one soma is taken per acquisition. When several somas in the same
    acquisition pass the synapse-count threshold, the winner is chosen by, in
    order: (1) synapses on both DMDs beats synapses on only one DMD, (2) the
    largest number of synapses on a single dendrite, (3) the largest total
    number of synapses.

    Parameters
    ----------
    min_n_synapses : int
        Minimum number of dendrite-assigned synapses on the soma.
    min_duration_seconds : float
        Minimum duration of the acquisition in seconds.

    Returns
    -------
    list[str]
        Member IDs formatted as ``"subject--exp--loc--acq--soma#"``.
    """
    valid_acqs = wis.meta.get.valid_acquisitions()
    valid_soma_acqs = []
    tot_n_syns = 0
    # Recorded somas use the convention ``soma#``; there have never been more
    # than nine in one FOV.
    valid_soma_ids = [f"soma{i}" for i in range(1, 10)]
    for v in valid_acqs:
        subject, exp, loc, acq = v.split("--")
        _hypno, acq_dur, _wake, _nrem, _rem = wis.get.acq_sleep_coverage(
            subject, exp, loc, acq
        )
        if acq_dur < min_duration_seconds:
            continue

        syn_info = wis.get.synid_labels(subject, exp, loc, acq)
        candidates: list[tuple[tuple[int, int, int], str]] = []
        for soma_id in syn_info["soma-ID"].unique():
            if soma_id not in valid_soma_ids:
                continue
            stats = _soma_syn_stats(syn_info, soma_id)
            if stats[2] >= min_n_synapses:
                candidates.append((stats, soma_id))
        if not candidates:
            continue
        # Tuple comparison implements the tiebreak priority order directly.
        best_stats, best_soma = max(candidates)
        if len(candidates) > 1:
            losers = ", ".join(s for _, s in candidates if s != best_soma)
            print(
                f"{v}: {len(candidates)} eligible somas; keeping {best_soma} "
                f"(n_dmds={best_stats[0]}, max_dend={best_stats[1]}, "
                f"n_syns={best_stats[2]}), dropping {losers}"
            )
        tot_n_syns += best_stats[2]
        valid_soma_acqs.append(f"{subject}--{exp}--{loc}--{acq}--{best_soma}")
    print(
        f"{len(valid_soma_acqs)} somas with at least {min_n_synapses} synapses"
        f" | total synapses across all somas: {tot_n_syns}"
    )
    return valid_soma_acqs


if __name__ == "__main__":
    args = _parse_args()
    acqs = single_soma_acqs()

    script_dir = Path(__file__).resolve().parent
    candidate_set_path = script_dir / "candidate_set" / "candidate_set.json"
    candidate_set_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_set_path.write_text(json.dumps(acqs, indent=2) + "\n")
    overview_dir = candidate_set_path.parent / "main_overview"
    write_main_overviews(acqs, overview_dir, replace=args.replace)
