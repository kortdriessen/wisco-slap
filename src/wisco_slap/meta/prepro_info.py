# ==============================================
# Meta File that this handles: prepro_info.yaml
# ==============================================
import glob
import os

import yaml

from wisco_slap.defs import anmat_root, exsum_mirror_root, exsum_source_root
from wisco_slap.meta.get import acq_master


def determine_processing_done(subject, exp, loc, acq):
    """Determine if the processing has been done for a given acquisition.

    Checks the ExperimentSummary source of truth (``defs.exsum_source_root``).

    Parameters
    ----------
    subject : str
        The subject ID.
    exp : str
        The experiment ID.
    loc : str
        The location ID.
    acq : str
        The acquisition ID.
    """
    data_dir = os.path.join(
        exsum_source_root, subject, exp, loc, acq, "ExperimentSummary"
    )
    exp_summary_files = glob.glob(os.path.join(data_dir, "*Summary*"))
    if len(exp_summary_files) > 0:
        return exp_summary_files[0]
    else:
        return "NO"


def update_prepro_info_acqid(subject, exp, loc, acq, value=None):
    """Record one acq's preprocessing version in ``prepro_info.yaml``.

    With ``value=None`` the version is determined from the ExSum source of
    truth; if the source has no ExSum but the mirror already holds one (an
    acq awaiting reprocessing), the mirror's version is recorded instead of
    ``"NO"`` — consistent with :func:`wis.meta.exsum_mirror.mirror_one_acq`.
    """
    acq_id = f"{loc}--{acq}"
    prepro_path = os.path.join(anmat_root, "prepro_info.yaml")
    with open(prepro_path) as f:
        prepro_info = yaml.safe_load(f)
    if subject not in prepro_info:
        prepro_info[subject] = {}
    if exp not in prepro_info[subject]:
        prepro_info[subject][exp] = {}
    if value is None:
        value = determine_processing_done(subject, exp, loc, acq)
        if value != "NO":
            value = os.path.basename(value).split(".mat")[0]
        else:
            mirror_dir = os.path.join(exsum_mirror_root, subject, exp, acq_id)
            mirror_mats = (
                [f for f in os.listdir(mirror_dir) if f.endswith(".mat")]
                if os.path.isdir(mirror_dir)
                else []
            )
            if mirror_mats:
                value = mirror_mats[0].split(".mat")[0]
    prepro_info[subject][exp][acq_id] = value
    with open(prepro_path, "w") as f:
        yaml.dump(prepro_info, f)
    return


def update_prepro_info():
    macq = acq_master()
    for subject in macq.keys():
        for exp in macq[subject].keys():
            for acq_id in macq[subject][exp]:
                loc, acq = acq_id.split("--")
                update_prepro_info_acqid(subject, exp, loc, acq)
    return
