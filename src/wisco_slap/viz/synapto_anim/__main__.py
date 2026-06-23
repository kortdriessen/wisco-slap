"""CLI entry point: ``python -m wisco_slap.viz.synapto_anim [config.yaml]``.

Behavior:
- No args → launch editor with blank scene
- ``[config.yaml]`` → launch editor with this config pre-loaded
- ``--render config.yaml`` → headless render the config to its `output.path`
"""

from __future__ import annotations

import argparse
import sys

from . import launch_editor, render_scene


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wisco_slap.viz.synapto_anim")
    ap.add_argument("config", nargs="?", default=None,
                    help="Path to YAML config (optional)")
    ap.add_argument("--render", action="store_true",
                    help="Render headlessly instead of opening the editor")
    args = ap.parse_args(argv)

    if args.render:
        if args.config is None:
            ap.error("--render requires a config path")
        out = render_scene(args.config)
        print(out)
        return 0
    launch_editor(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
