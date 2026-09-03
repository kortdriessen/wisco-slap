import slap2_py as spy
import wisco_slap as wis


def _piped_synapses(
    subject,
    exp,
    loc,
    acq,
    soma_id=None,
    trace="ls",
    channel=1,
    dend_only=True,
    valid_mask=True,
):
    if trace != "f0":
        scopx_dict = wis.get.syn_dF(
            subject,
            exp,
            loc,
            acq,
            trace=trace,
            channels=channel,
        )
    else:
        scopx_dict = wis.get.syn_F0(
            subject,
            exp,
            loc,
            acq,
            channels=channel,
        )
    # this gives us a dictionary where each key is a dmd name, and the value is the scopex array for each dmd.
    # Therefore, we can conveniently combine everything into a single scopex array, where the syn_id becomes a string of the form '{dmd-number}-{syn-ID}', e.g. '1-17' .
    scopx = wis.get.combine_scopex_arrays(scopx_dict)

    if soma_id is not None:
        # optionally, if we want to filter out only the synapses that belong to our soma, we can do that:
        scopx = scopx.sx.somaid(soma_id)

    if dend_only:
        # still further, if we want to filiter out only the synapses that actually have an
        # associated dendrite ID (i.e. aren't just synapses onto the soma or some other non-dendritic location):
        scopx = scopx.sx.dendid("any")
    if valid_mask:
        # once we have a scopex array for the data we want to analyze, we should make sure that we know which timepoints have valid data (i.e. not-NaN) for all synapses
        # we can do this with the wisco_slap validity module as follows:
        # this function will attach a coordinate on the time dimension called 'is_valid', which is true for any timepoint where all synapses are non-NaN, and false otherwise.
        scopx = wis.validity.attach_validity(scopx)

    return scopx


def _piped_soma(
    subject,
    exp,
    loc,
    acq,
    soma_id=None,
    trace="Fraw",
    channel=0,
    return_dFF=True,
):
    soma_dmd = None
    dmd_info = wis.meta.get.dmd_info()
    if soma_id is None:
        soma_ids = None
    else:
        soma_ids = [soma_id]

    for dmd in dmd_info[subject][exp][loc][acq].keys():
        if soma_id in dmd_info[subject][exp][loc][acq][dmd]["somas"]:
            soma_dmd = dmd
            break
    if soma_dmd is None:
        raise ValueError(f"No DMD found for soma {soma_id!r}")

    dmd_num = int(soma_dmd.split("-")[1])
    soma_dmd = f"dmd_{dmd_num}"
    roi = wis.get.roi_F(
        subject,
        exp,
        loc,
        acq,
        trace=trace,
        dmd=dmd_num,
        channels=channel,
        soma_ids=soma_ids,
        return_dFF=return_dFF,
    )
    roi = roi[soma_dmd]
    soma_roi = roi.sel(soma_id=soma_id)
    soma_roi = wis.validity.attach_validity(soma_roi)
    spike_prob, spikes = wis.scope.cascade.run_cascade_detection(soma_roi)
    return soma_roi, spike_prob, spikes


def _piped_footprints(subject, exp, loc, acq, fp_threshold=0.01):
    esum_p = wis.meta.get.esum_mirror_path(subject, exp, loc, acq)
    mim = spy.xsum.get_meanIM(esum_p)
    smap, fpvals = spy.xsum.get_fp_info(esum_p, threshold=fp_threshold)
    return mim, smap, fpvals


def _piped_peripherals(subject, exp, loc, acq, stores=["EEG_", "loal"]):
    sb = wis.meta.sync.get_acq_sync_block(subject, exp, loc, acq)
    ephys = wis.get.ephys(subject, exp, sb, stores=stores)
    # eeg = ephys['EEG_']
    # loal = ephys['loal']
    eye = wis.get.eye_metrics(subject, exp, sb)
    whisk = wis.get.whisk_df(subject, exp, sb)
    hypno_acq, acq_dur, frw, frn, frr = wis.get.acq_sleep_coverage(
        subject, exp, loc, acq
    )
    hypno_full = wis.get.bout_hypno(subject, exp, loc, acq)
    return ephys, eye, whisk, hypno_acq, hypno_full, acq_dur


_PIPELINES = {
    "soma": _piped_soma,
    "synapse": _piped_synapses,
    "footprints": _piped_footprints,
    "peripherals": _piped_peripherals,
}


def fetch_data_pipe(pipeline, *args, **kwargs):
    """Dispatch to a named data-loading pipeline.

    Parameters
    ----------
    pipeline : str
        Which loading routine to run. One of the keys of ``_PIPELINES``
        (currently ``"soma"``, ``"synapse"``, ``"footprints"``, or ``"peripherals"``).
    *args, **kwargs
        Forwarded verbatim to the selected pipeline function.

    Returns
    -------
    Whatever the selected pipeline returns (a scopex array for
    ``"synapse"``; see the individual ``_piped_*`` functions).

    Raises
    ------
    ValueError
        If ``pipeline`` is not a registered pipeline name.
    """
    try:
        fn = _PIPELINES[pipeline]
    except KeyError:
        options = ", ".join(sorted(_PIPELINES))
        raise ValueError(
            f"Unknown pipeline: {pipeline!r}. Available pipelines: {options}."
        ) from None
    return fn(*args, **kwargs)
