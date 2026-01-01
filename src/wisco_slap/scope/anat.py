from pathlib import Path

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Rectangle
import numpy as np
import wisco_slap.defs as DEFS
import wisco_slap as wis
import polars as pl


def anat_path(subject):
    """Return path to anatomy_locations.yaml for given subject."""
    path = f"{DEFS.data_root}/{subject}/{subject}_anatomy/anatomy_locations.yaml"
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
    with open(yaml_path) as f:
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
        ax.add_patch(
            Rectangle((llx, lly), rect_w, rect_h, fill=False, linewidth=1.0, zorder=2)
        )
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


def load_landmarks(subject):
    landmarks_path = f"{DEFS.data_root}/{subject}/{subject}_anatomy/landmarks.yaml"
    with open(landmarks_path) as f:
        landmarks = yaml.safe_load(f)
    return landmarks


def plot_3mm_window(subject, f, ax, diameter=3):
    # should take f and ax from plot_points_with_boxes, and plot a 3mm circle around the center point
    landmarks = load_landmarks(subject)
    center_position = landmarks["center_est"]
    center_x = center_position["x"]
    center_y = center_position["y"]
    ax.add_patch(
        Circle(
            (center_x, center_y),
            diameter / 2,
            fill=False,
            edgecolor="blue",
            linewidth=1.5,
            zorder=2,
        )
    )
    # Ensure the circle isn't distorted
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(center_x - diameter / 1.8, center_x + diameter / 1.8)
    ax.set_ylim(center_y - diameter / 1.8, center_y + diameter / 1.8)

    # plot an X at the center point
    ax.plot(center_x, center_y, "x", color="red", markersize=10, zorder=3)
    return f, ax


def get_somas_by_dmd(subject, exp, loc, acq):
    di = wis.util.info.load_dmd_info()
    dia = di[subject][exp][loc][acq]
    somas_by_dmd = {}
    for dmd in dia.keys():
        if len(dia[dmd]["somas"]) > 0:
            somas_by_dmd[dmd] = dia[dmd]["somas"]
    return somas_by_dmd
