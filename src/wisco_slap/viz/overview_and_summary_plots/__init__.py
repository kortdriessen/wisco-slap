"""Overview and summary figures, grouped into plot *sets*.

Each set is a subpackage whose modules each generate one plot (see
``acq_level/__init__.py`` / ``single_soma/__init__.py`` for the convention).
``plot_specs(set_name)`` returns every registered plot in a set;
``wis.pns.overview_plot_gen`` uses it to generate a whole set across the
set's members (every acquisition for ``acq_level``, every member of the
``single_soma`` subgroup for ``single_soma``).
"""

from . import acq_level as acq_level
from . import single_soma as single_soma
from ._spec import PlotSpec as PlotSpec
from ._spec import discover_specs as discover_specs
from .acq_level import acq_within_exp_path as acq_within_exp_path
from .acq_level import acq_within_exp_plot as acq_within_exp_plot
from .acq_level import basic_sync_summary_path as basic_sync_summary_path
from .acq_level import basic_sync_summary_plot as basic_sync_summary_plot
from .single_soma import load_single_soma_data as load_single_soma_data
from .single_soma import (
    single_soma_corr_overview_path as single_soma_corr_overview_path,
)
from .single_soma import (
    single_soma_corr_overview_plot as single_soma_corr_overview_plot,
)
from .single_soma import single_soma_overview_path as single_soma_overview_path
from .single_soma import single_soma_overview_plot as single_soma_overview_plot

#: Plot-set name -> subpackage holding one module per plot.
PLOT_SETS = {
    "acq_level": acq_level,
    "single_soma": single_soma,
}


def plot_specs(set_name: str) -> list[PlotSpec]:
    """Return every ``PlotSpec`` registered in the plot set ``set_name``."""
    try:
        package = PLOT_SETS[set_name]
    except KeyError:
        raise ValueError(
            f"Unknown plot set {set_name!r}; known sets: {sorted(PLOT_SETS)}"
        ) from None
    return discover_specs(package)


__all__ = [
    "PLOT_SETS",
    "PlotSpec",
    "acq_level",
    "acq_within_exp_path",
    "acq_within_exp_plot",
    "basic_sync_summary_path",
    "basic_sync_summary_plot",
    "discover_specs",
    "load_single_soma_data",
    "plot_specs",
    "single_soma",
    "single_soma_corr_overview_path",
    "single_soma_corr_overview_plot",
    "single_soma_overview_path",
    "single_soma_overview_plot",
]
