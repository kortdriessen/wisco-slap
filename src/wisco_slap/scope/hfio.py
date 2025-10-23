import slap2_py as spy
import wisco_slap as wis
import wisco_slap.defs as DEFS


def load_mean_ims(subject, exp, loc, acq):
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    mean_ims = {}
    for dmd in [1, 2]:
        meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
        print(meanim.shape)
        mean_ims[dmd] = meanim.swapaxes(1, 2)
    return mean_ims


def load_fprts(subject, exp, loc, acq):
    fps = {}
    esum_path = wis.util.io.sub_esum_path(subject, exp, loc, acq)
    for dmd in [1, 2]:
        fp = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][0]['footprints']")
        fp = fp.swapaxes(1, 2)
        fps[dmd] = fp
    return fps
