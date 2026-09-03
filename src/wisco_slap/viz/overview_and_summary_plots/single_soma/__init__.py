"""Single-soma overview plots — one module per plot.

Convention for this directory: each module builds exactly one figure for one
``(subject, exp, loc, acq, soma_id)`` — a member of the ``single_soma``
subgroup (``wis.meta.subgroups.single_soma``) — and provides

- ``<name>_plot(subject, exp, loc, acq, soma_id, *, overwrite=False, ...)`` —
  builds, saves and returns ``(fig, axes)``;
- ``<name>_path(subject, exp, loc, acq, soma_id) -> Path`` — where the PNG is
  saved (``<annotation_root>/OVERVIEW_FIGURES/single_soma/<name>/``);
- ``PLOT_SPEC = PlotSpec(name=..., plot=<name>_plot, path=<name>_path)`` so the
  module is picked up by ``wis.pns.overview_plot_gen.generate_plot_set``.
"""

from ._single_soma_corr_overview import (
    compute_single_soma_corr_differences as compute_single_soma_corr_differences,
)
from ._single_soma_corr_overview import (
    compute_single_soma_corr_matrices as compute_single_soma_corr_matrices,
)
from ._single_soma_corr_overview import (
    corr_difference_figure as corr_difference_figure,
)
from ._single_soma_corr_overview import corr_overview_figure as corr_overview_figure
from ._single_soma_corr_overview import (
    load_single_soma_corr_signals as load_single_soma_corr_signals,
)
from ._single_soma_corr_overview import (
    single_soma_corr_overview_path as single_soma_corr_overview_path,
)
from ._single_soma_corr_overview import (
    single_soma_corr_overview_plot as single_soma_corr_overview_plot,
)
from ._single_soma_overview import load_roi_polys as load_roi_polys
from ._single_soma_overview import load_single_soma_data as load_single_soma_data
from ._single_soma_overview import (
    single_soma_overview_path as single_soma_overview_path,
)
from ._single_soma_overview import (
    single_soma_overview_plot as single_soma_overview_plot,
)

__all__ = [
    "compute_single_soma_corr_differences",
    "compute_single_soma_corr_matrices",
    "corr_difference_figure",
    "corr_overview_figure",
    "load_roi_polys",
    "load_single_soma_corr_signals",
    "single_soma_corr_overview_path",
    "single_soma_corr_overview_plot",
    "load_single_soma_data",
    "single_soma_overview_path",
    "single_soma_overview_plot",
]
