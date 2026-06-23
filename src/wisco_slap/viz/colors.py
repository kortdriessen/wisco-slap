"""Tiny colormap utilities for the SLAP2 viz layer."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgba


def two_color_cmap(
    color_1: Any,
    color_2: Any,
    *,
    name: str = "two_color",
    n: int = 256,
    show: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.0, 0.6),
) -> Colormap:
    """Make a colormap that linearly interpolates ``color_1`` → ``color_2``.

    Drop-in for ``wis.viz.plot_synapto_overlays(..., cmap=cmap)`` and the
    ``overlay_kwargs={'cmap': cmap}`` field on ``synapto_anim.SceneConfig``.

    Parameters
    ----------
    color_1, color_2 : matplotlib color spec
        Anything ``matplotlib.colors.to_rgba`` accepts: ``'#e8ff16'``,
        ``'red'``, ``(1, 0, 0)``, ``(0.5, 0.5, 0.5, 1.0)``.
    name : str
        Colormap registry name.
    n : int
        Number of discrete sample points (256 is plenty for smooth output).
    show : bool
        If True (default), display a horizontal swatch of the colormap so
        you can see what you got before plugging it in elsewhere.
    ax : matplotlib Axes | None
        Axes to draw the swatch on. If None and ``show`` is True, a small
        new figure is created.
    figsize : (w, h)
        Size of the auto-created swatch figure (ignored when ``ax`` is given).

    Returns
    -------
    matplotlib.colors.Colormap
        A ``LinearSegmentedColormap``. ``cmap(0.0)`` returns ``color_1``,
        ``cmap(1.0)`` returns ``color_2``.
    """
    rgba_1 = to_rgba(color_1)
    rgba_2 = to_rgba(color_2)
    cmap = LinearSegmentedColormap.from_list(name, [rgba_1, rgba_2], N=int(n))

    if show:
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        gradient = np.linspace(0.0, 1.0, int(n)).reshape(1, -1)
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_xticks([0, int(n) - 1])
        ax.set_xticklabels(["color_1", "color_2"])
        ax.set_yticks([])
        ax.set_title(name)
    return cmap
