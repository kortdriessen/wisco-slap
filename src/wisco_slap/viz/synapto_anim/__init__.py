"""Synapse float-off animation v2 — config-driven, multi-DMD, MOV output.

Public surface:

- :class:`SceneConfig` — the YAML-backed dataclass that fully describes a scene.
- :func:`render_scene` — headless rendering to a single ``.mov`` (alpha).
- :func:`launch_editor` — open the PySide6 scene editor.

CLI: ``python -m wisco_slap.viz.synapto_anim [config.yaml]``.
"""

from .config import (
    AcquisitionConfig as AcquisitionConfig,
    AnimationConfig as AnimationConfig,
    CanvasConfig as CanvasConfig,
    OutputConfig as OutputConfig,
    OverlayKwargs as OverlayKwargs,
    Placement as Placement,
    ReferenceImage as ReferenceImage,
    SceneConfig as SceneConfig,
    TargetStrip as TargetStrip,
    load as load_config,
    save as save_config,
)
from .render import LandingInfo as LandingInfo
from .render import landing_info_for_dmd as landing_info_for_dmd
from .render import render_scene as render_scene


def launch_editor(
    config_path: str | None = None,
    *,
    cmap: "str | object | None" = None,
) -> None:
    """Open the PySide6 scene editor. Optionally pre-load a YAML config.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML config to pre-load; if None, the New-scene dialog appears.
    cmap : str | matplotlib Colormap | None
        Override the colormap used for synapse colors. Strings are passed
        through as a colormap name; a ``Colormap`` object (e.g. from
        :func:`wisco_slap.viz.two_color_cmap`) is registered with matplotlib
        under its ``.name`` so it can be referenced by name everywhere
        (config field, YAML, dropdown).
    """
    from .editor import run_editor  # imported lazily — Qt is heavy
    run_editor(config_path, cmap=cmap)


__all__ = [
    "AcquisitionConfig",
    "AnimationConfig",
    "CanvasConfig",
    "LandingInfo",
    "OutputConfig",
    "OverlayKwargs",
    "Placement",
    "ReferenceImage",
    "SceneConfig",
    "TargetStrip",
    "landing_info_for_dmd",
    "launch_editor",
    "load_config",
    "render_scene",
    "save_config",
]
