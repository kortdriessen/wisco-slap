"""Configuration schema for the synapse float-off animation.

A ``SceneConfig`` is the single source of truth used by both the PySide6
editor (which reads/writes it visually) and the headless renderer (which
turns it into a ``.mov`` file). YAML on disk, dataclasses in memory.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# Canonical DMD keys used everywhere in the config.
DMD_KEYS: tuple[str, ...] = ("dmd_1", "dmd_2")


@dataclass
class CanvasConfig:
    width: int = 1920
    height: int = 1080
    # Background color: None = fully transparent, or [r, g, b, a] uint8.
    background_color: list[int] | None = None


@dataclass
class AcquisitionConfig:
    subject: str
    exp: str
    loc: str
    acq: str

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.subject, self.exp, self.loc, self.acq)


@dataclass
class Placement:
    """Where a DMD's cropped mean image sits on the canvas."""
    x: float = 100.0          # top-left, canvas pixels
    y: float = 100.0
    scale: float = 1.0        # uniform scale of the cropped mean image
    rotation_deg: float = 0.0  # reserved, v2 ignores


@dataclass
class TargetStrip:
    """Rectangle (canvas-coords) where synapses land. Long axis auto-detected."""
    x0: float
    y0: float
    x1: float
    y1: float
    end_mode: Literal["dots", "dendrite_bars"] = "dendrite_bars"
    bar_thickness_px: float = 4.0
    bar_gap_frac: float = 0.015
    dot_radius_px: float = 1.5

    @property
    def is_horizontal(self) -> bool:
        return abs(self.x1 - self.x0) >= abs(self.y1 - self.y0)


@dataclass
class AnimationConfig:
    duration_s: float = 4.0           # length of the *flight + morph* phase (the existing animation)
    fps: int = 30
    stagger: Literal["wave", "random", "none"] = "wave"
    stagger_frac: float = 0.30
    morph_frac: float = 0.15
    seed: int | None = 0
    # Optional intro phases prepended before the main flight:
    #   open pause (mean image only) → appear (synapses fade in one-by-one,
    #   proximal→distal within each dendrite, with a small gap between
    #   dendrites) → close pause (fully visible).
    # Total appear duration is computed from the per-synapse/per-dendrite
    # timings below: total = (sum_per_dend N_syn * per_syn_gap) +
    #     (n_dends - 1) * dend_gap + per_syn_fade.
    # Total video length = intro_open_pause_s + computed_appear + intro_close_pause_s + duration_s.
    intro_enabled: bool = True
    intro_open_pause_s: float = 0.4
    intro_close_pause_s: float = 0.4
    intro_per_syn_fade_s: float = 0.06      # how long each synapse takes to fade in
    intro_per_syn_gap_s: float = 0.015      # gap between consecutive synapses inside a dendrite
    intro_dend_gap_s: float = -0.10         # extra pause between dendrites; may be NEGATIVE
                                            # to overlap dendrites (e.g. -0.3 → next dendrite
                                            # starts 0.3s before the previous one's last syn)
    # Deprecated, kept for back-compat with older YAML configs (ignored at render time).
    intro_appear_s: float = 1.2
    intro_appear_stagger_frac: float = 0.6


@dataclass
class OverlayKwargs:
    """Forwarded to the sprite-building helpers (same names as
    ``plot_synapto_overlays``)."""
    soma: str | None = None
    channel: int | None = 0
    trace: Literal["denoised", "ls"] = "denoised"
    color_by_pos: bool = True
    cmap: str = "summer"
    fp_threshold: float = 0.02
    crop_buf_px: int = 5
    log_vmin: float = 5.0
    log_vmax_pct: float = 99.9
    mean_im_channel: int = 1
    require_dend: bool = True


@dataclass
class OutputConfig:
    path: str = "/tmp/synapto_anim.mov"
    codec: Literal["prores_ks", "qtrle", "png"] = "prores_ks"
    prores_profile: int = 4  # 4 = ProRes 4444
    also_write_pngs: bool = False


@dataclass
class ReferenceImage:
    """An image (typically a PNG of a correlation matrix, raster, or any plot)
    placed on the canvas as a visual alignment guide.

    By default references are **editor-only** — they appear in the editor as
    a draggable layout aid but are NOT rendered into the output ``.mov``.
    Set ``include_in_render=True`` to bake the reference into the rendered
    video (useful for quick-share previews; usually you'll keep this off and
    composite the real plot in After Effects on top).
    """
    path: str
    x: float = 100.0
    y: float = 100.0
    scale: float = 1.0
    opacity: float = 0.5
    visible: bool = True
    include_in_render: bool = False
    z_order: Literal["below", "above"] = "below"   # vs the synapse animation


@dataclass
class SceneConfig:
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    acquisition: AcquisitionConfig = field(default_factory=lambda: AcquisitionConfig(
        subject="", exp="", loc="", acq=""))
    placements: dict[str, Placement] = field(default_factory=dict)
    targets: dict[str, TargetStrip] = field(default_factory=dict)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    overlay_kwargs: OverlayKwargs = field(default_factory=OverlayKwargs)
    output: OutputConfig = field(default_factory=OutputConfig)
    references: list[ReferenceImage] = field(default_factory=list)

    # -- factories --------------------------------------------------------

    @classmethod
    def default(
        cls,
        subject: str,
        exp: str,
        loc: str,
        acq: str,
        *,
        canvas_size: tuple[int, int] = (1920, 1080),
        out_path: str = "/tmp/synapto_anim.mov",
        codec: Literal["prores_ks", "qtrle", "png"] = "prores_ks",
    ) -> "SceneConfig":
        """Sensible default scene for a fresh acquisition.

        - Canvas: 1920×1080 transparent
        - DMD 1 placed at (100, 100), DMD 2 at (100, 600), both scale 1.0
        - Target strips: thin vertical bars on the right side of the canvas
        - Animation: 4 s at 30 fps, wave stagger, dendrite_bars endpoints
        """
        W, H = canvas_size
        return cls(
            canvas=CanvasConfig(width=W, height=H, background_color=None),
            acquisition=AcquisitionConfig(subject=subject, exp=exp, loc=loc, acq=acq),
            placements={
                "dmd_1": Placement(x=100.0, y=100.0, scale=1.0),
                "dmd_2": Placement(x=100.0, y=H * 0.55, scale=1.0),
            },
            targets={
                "dmd_1": TargetStrip(
                    x0=W * 0.85, y0=80,
                    x1=W * 0.85, y1=H * 0.45,
                    end_mode="dendrite_bars",
                ),
                "dmd_2": TargetStrip(
                    x0=W * 0.92, y0=H * 0.55,
                    x1=W * 0.92, y1=H * 0.95,
                    end_mode="dendrite_bars",
                ),
            },
            animation=AnimationConfig(),
            overlay_kwargs=OverlayKwargs(),
            output=OutputConfig(path=out_path, codec=codec),
        )

    # -- (de)serialization ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # asdict converts dicts of dataclasses fine; nothing else to fix.
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SceneConfig":
        canvas = CanvasConfig(**d.get("canvas", {}))
        acq = AcquisitionConfig(**d["acquisition"])
        placements = {
            k: Placement(**v) for k, v in d.get("placements", {}).items()
        }
        targets = {
            k: TargetStrip(**v) for k, v in d.get("targets", {}).items()
        }
        animation = AnimationConfig(**d.get("animation", {}))
        overlay = OverlayKwargs(**d.get("overlay_kwargs", {}))
        output = OutputConfig(**d.get("output", {}))
        references = [ReferenceImage(**r) for r in d.get("references", [])]
        return cls(
            canvas=canvas, acquisition=acq, placements=placements,
            targets=targets, animation=animation, overlay_kwargs=overlay,
            output=output, references=references,
        )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------


def load(path: str | os.PathLike) -> SceneConfig:
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{p} did not contain a YAML mapping at the top level")
    return SceneConfig.from_dict(data)


def save(config: SceneConfig, path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    return p
