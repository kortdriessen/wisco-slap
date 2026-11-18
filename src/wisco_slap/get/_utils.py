import numpy as np
import xarray as xr


def combine_scopex_arrays(sxs: dict[str, xr.DataArray]) -> xr.DataArray:
    """Combine per-DMD scopex arrays into one array along ``syn_id``.

    Parameters
    ----------
    sxs : dict[str, xr.DataArray]
        Mapping of ``"dmd_<n>"`` keys (as returned by :func:`wisco_slap.get.syn_dF`
        and friends) to arrays with a ``syn_id`` dimension. Any subset of DMDs is
        accepted, so acquisitions or selections that only have one DMD work too.

    Returns
    -------
    xr.DataArray
        Concatenated along ``syn_id`` in ascending DMD order, with a ``dmd``
        coordinate on ``syn_id``, ``syn_id`` renamed to ``"<dmd>-<id>"`` and an
        integer ``syn_id_number`` coordinate holding the original id.
    """

    if not sxs:
        raise ValueError("combine_scopex_arrays received no arrays")
    dmds = sorted(int(key.split("_")[1]) for key in sxs)
    pieces = []
    for dmd in dmds:
        array = sxs[f"dmd_{dmd}"]
        sids = array["syn_id"].values
        pieces.append(
            array.assign_coords(
                dmd=("syn_id", np.full(array.sizes["syn_id"], dmd, dtype=int)),
                syn_id=[f"{dmd}-{sid}" for sid in sids],
            )
        )
    mfs = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="syn_id")
    syn_id_numbers = [int(x.split("-")[1]) for x in mfs["syn_id"].values]
    return mfs.assign_coords(syn_id_number=("syn_id", syn_id_numbers))
