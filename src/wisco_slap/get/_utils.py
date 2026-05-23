import numpy as np
import xarray as xr


def combine_scopex_arrays(sxs: dict[str, xr.DataArray]) -> xr.DataArray:
    """Combine multiple scopexrays into one, adding a 'dmd' coordinate, and renaming the syn_id coordinate to include the dmd."""

    # make a dmd coordinate (equal to 1) on syn_id dimension:
    sxs["dmd_1"] = sxs["dmd_1"].assign_coords(
        dmd=("syn_id", np.ones(sxs["dmd_1"].sizes["syn_id"], dtype=int))
    )
    sxs["dmd_2"] = sxs["dmd_2"].assign_coords(
        dmd=("syn_id", np.ones(sxs["dmd_2"].sizes["syn_id"], dtype=int) * 2)
    )
    for dmd in [1, 2]:
        new_ids = []
        sids = sxs[f"dmd_{dmd}"]["syn_id"].values
        for sid in sids:
            new_ids.append(f"{dmd}-{sid}")
        sxs[f"dmd_{dmd}"] = sxs[f"dmd_{dmd}"].assign_coords(syn_id=new_ids)
    mfs = xr.concat([sxs["dmd_1"], sxs["dmd_2"]], dim="syn_id")
    return mfs
