# ==============================================
# Meta File that this handles: prepro_info.yaml
# ==============================================
import glob
import os

import yaml

from wisco_slap.defs import anmat_root, data_root
from wisco_slap.meta.get import acq_master


def determine_processing_done(subject, exp, loc, acq):
    """Determine if the processing has been done for a given acquisition.

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
    data_dir = os.path.join(data_root, subject, exp, loc, acq, "ExperimentSummary")
    exp_summary_files = glob.glob(os.path.join(data_dir, "*Summary*"))
    if len(exp_summary_files) > 0:
        return exp_summary_files[0]
    else:
        return "NO"


def update_prepro_info_acqid(subject, exp, loc, acq, value=None):
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
