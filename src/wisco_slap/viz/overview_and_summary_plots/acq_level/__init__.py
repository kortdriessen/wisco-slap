"""Acquisition-level overview plots — one module per plot.

Convention for this directory: each module builds exactly one figure for one
``(subject, exp, loc, acq)`` and provides

- ``<name>_plot(subject, exp, loc, acq, *, overwrite=False, ...)`` — builds,
  saves and returns ``(fig, axes)``;
- ``<name>_path(subject, exp, loc, acq) -> Path`` — where the PNG is saved;
- ``PLOT_SPEC = PlotSpec(name=..., plot=<name>_plot, path=<name>_path)`` so the
  module is picked up by ``wis.pns.overview_plot_gen.generate_plot_set``.
"""

from ._acq_within_exp import acq_within_exp_path as acq_within_exp_path
from ._acq_within_exp import acq_within_exp_plot as acq_within_exp_plot
from ._sync_overview import basic_sync_summary_path as basic_sync_summary_path
from ._sync_overview import basic_sync_summary_plot as basic_sync_summary_plot

__all__ = [
    "acq_within_exp_path",
    "acq_within_exp_plot",
    "basic_sync_summary_path",
    "basic_sync_summary_plot",
]
