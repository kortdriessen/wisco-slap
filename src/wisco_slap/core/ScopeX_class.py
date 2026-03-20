"""
ScopeX xarray accessor — registers `da.sx.*` convenience methods on all DataArrays.

Usage:
    import wisco_slap  # auto-registers the accessor
    da.sx.ts(0, 15)
    da.sx.dendid('G-1')
    da.sx.glut()
"""

from __future__ import annotations

import xarray as xr

TimeLike = int | float


@xr.register_dataarray_accessor("sx")
class ScopeXAccessor:
    """SLAP2 ScopeX convenience methods for xarray DataArrays.

    Provides shorthand filtering/selection methods for the standard ScopeX
    coordinate system: (channel, syn_id | soma_id, time) with non-dimension
    coordinates like dend-ID, soma-ID, source-ID, etc.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    # ── Time ──────────────────────────────────────────────────────────

    def ts(self, t1: TimeLike, t2: TimeLike) -> xr.DataArray:
        """Select a time window (inclusive on both ends).

        Parameters
        ----------
        t1, t2 : int or float
            Start and end times in seconds.

        Returns
        -------
        xr.DataArray
            Filtered to time >= t1 and time <= t2.
        """
        return self._obj.sel(time=slice(t1, t2))

    # ── Channel selection ─────────────────────────────────────────────

    def ch(self, n: int) -> xr.DataArray:
        """Select a single channel by index.

        Parameters
        ----------
        n : int
            Channel index (0 = iGluSnFR4f, 1 = jRGECO1a).
        """
        return self._obj.sel(channel=n)

    def glut(self) -> xr.DataArray:
        """Select the iGluSnFR4f channel (channel 0)."""
        return self._obj.sel(channel=0)

    def calc(self) -> xr.DataArray:
        """Select the jRGECO1a calcium channel (channel 1)."""
        return self._obj.sel(channel=1)

    # ── Synapse / source selection ────────────────────────────────────

    def sid(self, *ids: int) -> xr.DataArray:
        """Select synapse(s) by syn_id.

        Parameters
        ----------
        *ids : int
            One or more synapse IDs. A single ID returns that synapse;
            multiple IDs return all of them.

        Examples
        --------
        >>> da.sx.sid(3)
        >>> da.sx.sid(0, 3, 7)
        """
        if len(ids) == 1:
            return self._obj.sel(syn_id=ids[0])
        return self._obj.sel(syn_id=list(ids))

    def source(self, source_id: int) -> xr.DataArray:
        """Select synapse(s) by source-ID coordinate.

        Parameters
        ----------
        source_id : int
            The source-ID value to filter on.
        """
        return self._obj.sel(syn_id=self._obj["source-ID"] == source_id)

    # ── Metadata coordinate filters ──────────────────────────────────

    def dendid(self, dend_id: str) -> xr.DataArray:
        """Select synapse(s) belonging to a specific dendrite.

        Parameters
        ----------
        dend_id : str
            Dendrite identifier (e.g. 'G-1', 'B-2').
        """
        return self._obj.sel(syn_id=self._obj["dend-ID"] == dend_id)

    def somaid(self, soma_id: str) -> xr.DataArray:
        """Select synapse(s) belonging to a specific soma.

        Parameters
        ----------
        soma_id : str
            Soma identifier.
        """
        return self._obj.sel(syn_id=self._obj["soma-ID"] == soma_id)

    def dendtype(self, dend_type: str) -> xr.DataArray:
        """Select synapse(s) by dendrite type.

        Parameters
        ----------
        dend_type : str
            Dendrite type label.
        """
        return self._obj.sel(syn_id=self._obj["dend-type"] == dend_type)

    def syntype(self, synapse_type: str) -> xr.DataArray:
        """Select synapse(s) by synapse type.

        Parameters
        ----------
        synapse_type : str
            Synapse type label.
        """
        return self._obj.sel(syn_id=self._obj["synapse-type"] == synapse_type)

    # ── Soma DataArray helpers ────────────────────────────────────────

    def soma(self, soma_id: str) -> xr.DataArray:
        """Select a single soma trace by soma_id (for ROI DataArrays).

        Parameters
        ----------
        soma_id : str
            Soma identifier (e.g. 'soma1').
        """
        return self._obj.sel(soma_id=soma_id)
