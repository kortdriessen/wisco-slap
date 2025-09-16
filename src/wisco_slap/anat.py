# plot_injection_sites.py
# Usage: python plot_injection_sites.py /path/to/anatomy_locations.yaml

import sys
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csc.defs as DEFS

def anat_path(subject):
    """Return path to anatomy_locations.yaml for given subject."""
    path = f'{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_locations.yaml'
    return Path(path)

def load_points(yaml_path: Path):
    """Return list of (name, x, y) from a YAML file.

    Accepts either:
      name:
        x: <float>
        y: <float>
        z: <float or ignored>
    or:
      name: [x, y, ...]
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    points = []
    if isinstance(data, dict):
        for name, entry in data.items():
            if isinstance(entry, dict) and "x" in entry and "y" in entry:
                points.append((str(name), float(entry["x"]), float(entry["y"])))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                points.append((str(name), float(entry[0]), float(entry[1])))

    if not points:
        raise ValueError(
            "No (x, y) points found. Expected entries like "
            "`name: {x: <num>, y: <num>, ...}` or `name: [x, y, ...]`."
        )
    return points

def plot_points_with_boxes(points, rect_w=0.3, rect_h=0.2, out_png=None, show=False):
    """Plot centers and draw centered rectangles of given size in mm."""
    fig, ax = plt.subplots(figsize=(6, 6))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]

    # Scatter centers
    ax.scatter(xs, ys, s=20, zorder=3)

    # Centered rectangles
    for name, x, y in points:
        llx = x - rect_w / 2.0
        lly = y - rect_h / 2.0
        ax.add_patch(Rectangle((llx, lly), rect_w, rect_h, fill=False, linewidth=1.0, zorder=2))
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Injection Site Centers with {rect_w} mm × {rect_h} mm Boxes")
    ax.grid(True)
    plt.tight_layout()

    if out_png:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300)
    if show:
        plt.show()
    return fig, ax
