"""Generator for the ``single_soma`` subgroup (see ``../subgroup_info.md``).

``single_soma_acqs`` applies the inclusion criteria to the current valid
acquisitions and returns the member ids; ``members`` is the subgroup-set entry
point that ``wis.meta.subgroups.load_subgroup_set`` calls. Membership is always
computed live, so running this file as a script just prints the current set.
"""

import polars as pl

import wisco_slap as wis


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
    min_n_synapses: int = 40,
    min_duration_seconds: float = 1200,
    verbose: bool = True,
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
    verbose : bool
        Print the per-acquisition tie-break decisions and the final count.

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
        # Inclusion needs acquisition duration only. acq_sleep_coverage uses
        # this same timing value but also scores the full-block hypnogram.
        _start, _stop, acq_dur, _sb = wis.meta.get_acq_timing(subject, exp, loc, acq)
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
        if verbose and len(candidates) > 1:
            losers = ", ".join(s for _, s in candidates if s != best_soma)
            print(
                f"{v}: {len(candidates)} eligible somas; keeping {best_soma} "
                f"(n_dmds={best_stats[0]}, max_dend={best_stats[1]}, "
                f"n_syns={best_stats[2]}), dropping {losers}"
            )
        tot_n_syns += best_stats[2]
        valid_soma_acqs.append(f"{subject}--{exp}--{loc}--{acq}--{best_soma}")
    if verbose:
        print(
            f"{len(valid_soma_acqs)} somas with at least {min_n_synapses} synapses"
            f" | total synapses across all somas: {tot_n_syns}"
        )
    return valid_soma_acqs


def members(verbose: bool = True) -> list[str]:
    """The current members of the ``single_soma`` subgroup.

    The subgroup-set entry point required by the folder contract (see
    ``../subgroup_info.md``): ``wis.meta.subgroups.load_subgroup_set`` calls
    this on every load, so membership is always recomputed from the current
    data. Applies :func:`single_soma_acqs` with the standard criteria.

    Parameters
    ----------
    verbose : bool
        Forwarded to :func:`single_soma_acqs`.

    Returns
    -------
    list[str]
        Member IDs formatted as ``"subject--exp--loc--acq--soma#"``.
    """
    return single_soma_acqs(verbose=verbose)


if __name__ == "__main__":
    for member in members():
        print(member)
