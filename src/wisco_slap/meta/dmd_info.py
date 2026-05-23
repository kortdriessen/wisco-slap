# ============================================
# Meta File that this handles: dmd_info.yaml
# ============================================
import yaml
from slap2_py.core.xsum import get_roi_list

import wisco_slap.defs as DEFS
from wisco_slap.meta.get import esum_mirror_path


def _update_dmd_info(subject, exp, loc, acq):
    dmd_info_path = f"{DEFS.anmat_root}/dmd_info.yaml"
    esum_path = esum_mirror_path(subject, exp, loc, acq)
    # esum_mirror_path can return either None or the sentinel string
    # "NO_ESUM_MIRROR" when no .mat has been mirrored yet. In both
    # cases we have nothing to read, so this acq is a no-op.
    if esum_path is None or esum_path == "NO_ESUM_MIRROR":
        print(f"{subject} {exp} {loc} {acq} has no esum path")
        return None
    with open(dmd_info_path) as f:
        dmd_info = yaml.safe_load(f)
    if subject not in dmd_info:
        dmd_info[subject] = {}
    if exp not in dmd_info[subject]:
        dmd_info[subject][exp] = {}
    if loc not in dmd_info[subject][exp]:
        dmd_info[subject][exp][loc] = {}
    if acq not in dmd_info[subject][exp][loc]:
        dmd_info[subject][exp][loc][acq] = {}
    if "dmd-1" not in dmd_info[subject][exp][loc][acq]:
        dmd_info[subject][exp][loc][acq]["dmd-1"] = {}
        dmd_info[subject][exp][loc][acq]["dmd-1"]["depth"] = -1
    if "dmd-2" not in dmd_info[subject][exp][loc][acq]:
        dmd_info[subject][exp][loc][acq]["dmd-2"] = {}
        dmd_info[subject][exp][loc][acq]["dmd-2"]["depth"] = -1
    for dmd in [1, 2]:
        roi = get_roi_list(esum_path, dmd)
        if roi == []:
            dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"] = []
        elif roi[0] == 0:
            dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"] = []
        else:
            roi_names = []
            for r in roi:
                roi_names.append(r["Label"])
            dmd_info[subject][exp][loc][acq][f"dmd-{dmd}"]["somas"] = roi_names
    with open(dmd_info_path, "w") as f:
        yaml.dump(dmd_info, f)
