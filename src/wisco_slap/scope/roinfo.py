import os

import polars as pl
import tifffile as tiff

from wisco_slap.defs import anmat_root


def load_roi_info_df(subject, exp, loc, acq):
    return pl.DataFrame(
        schema={
            "soma-ID": pl.Int32,
            "dmd": pl.Int32,
            "dmd_depth": pl.Float32,
            "centroid_x": pl.Float32,
            "centroid_y": pl.Float32,
        }
    )


def load_roi_map(subject, exp, loc, acq, dmd):
    path = os.path.join(
        anmat_root,
        "annotation_materials",
        subject,
        exp,
        loc,
        acq,
        "roi_locations",
        f"roi_locs_dmd{dmd}_mask.tif",
    )
    if not os.path.exists(path):
        print(f"{path} does not exist!")
        return None
    return tiff.imread(path)


def load_dual_roi_map(subject, exp, loc, acq):
    roi_map = {}
    for dmd in [1, 2]:
        roi_map[dmd] = load_roi_map(subject, exp, loc, acq, dmd)
    return roi_map
