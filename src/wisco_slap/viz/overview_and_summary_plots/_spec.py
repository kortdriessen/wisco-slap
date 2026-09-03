"""Registration convention for overview / summary plot modules.

Every module inside a plot-set subpackage (e.g. ``acq_level/``) generates one
plot and must expose a module-level ``PLOT_SPEC = PlotSpec(...)`` naming its
plot function and the path function that says where that plot is saved.
:func:`discover_specs` walks a subpackage and collects them, which is how
``wis.pns.overview_plot_gen`` finds every plot in a set without a hand-kept
registry.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

#: Attribute name each plot module must define.
SPEC_ATTR = "PLOT_SPEC"


@dataclass(frozen=True)
class PlotSpec:
    """One plot in a plot set: how to build it and where it is saved.

    Attributes
    ----------
    name
        Short identifier, e.g. ``"basic_sync_summary"``. Used in generator
        summaries; should match the plot's output subdirectory name.
    plot
        Callable ``plot(*ids, overwrite: bool = ...)`` that builds, saves, and
        returns the figure. For the ``acq_level`` set ``ids`` is
        ``(subject, exp, loc, acq)``.
    path
        Callable ``path(*ids) -> Path`` giving the PNG location for ``ids``,
        without building anything.
    """

    name: str
    plot: Callable[..., Any]
    path: Callable[..., Path]


def discover_specs(package: ModuleType) -> list[PlotSpec]:
    """Import every module in ``package`` and collect their ``PLOT_SPEC``.

    Modules without a ``PLOT_SPEC`` attribute (helpers, ``__init__``) are
    ignored. Results are sorted by spec name so iteration order is stable.
    """
    specs: list[PlotSpec] = []
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        spec = getattr(module, SPEC_ATTR, None)
        if spec is None:
            continue
        if not isinstance(spec, PlotSpec):
            raise TypeError(
                f"{module.__name__}.{SPEC_ATTR} must be a PlotSpec, "
                f"got {type(spec).__name__}"
            )
        specs.append(spec)
    return sorted(specs, key=lambda s: s.name)
