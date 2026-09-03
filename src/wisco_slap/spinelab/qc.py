"""Per-spine QC figures for spine-size measurements.

One figure per measured spine: a montage of every display time bin cropped
around the fixed head ROI, the relative Wright-area trajectory for each
projection mode, and a context panel showing the head ROI with its automatic
shaft samples. Inspect these before believing any measurement.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from spine_labeller.core import compile_spines, load_annotations
from spine_labeller.data import (
    load_dataset_metadata,
    load_dmd_images,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "spine"


def write_spine_qc_figures(
    data_dir: Path | str,
    *,
    dmd: int,
    annotations_path: Path | str,
    rows: list[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Write one annotated montage and trajectory figure per measured spine."""

    directory = Path(data_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = load_dataset_metadata(directory)
    arrays = load_dmd_images(directory, dmd)
    document = load_annotations(annotations_path)
    spines = {spine.spine_id: spine for spine in compile_spines(document)}
    rows_by_spine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_spine[str(row["spine_id"])].append(row)

    stack = arrays["mean_stack"]
    reference = arrays["reference"]
    chunks = metadata["dmds"][str(dmd)]["chunks"]
    n_bins = int(stack.shape[0])
    bin_columns = 5
    bin_rows = max(1, math.ceil(n_bins / bin_columns))
    acquisition = metadata.get("acquisition", {})
    acq_label = " ".join(
        str(acquisition[key])
        for key in ("subject", "experiment", "location", "acquisition")
        if key in acquisition
    )

    written: list[Path] = []
    for spine_id, spine_rows in rows_by_spine.items():
        spine = spines[spine_id]
        head_rows, head_cols = np.where(spine.head_mask)
        if not len(head_rows):
            continue
        margin = 14
        row_slice = slice(
            max(0, int(head_rows.min()) - margin),
            min(reference.shape[0], int(head_rows.max()) + margin + 1),
        )
        col_slice = slice(
            max(0, int(head_cols.min()) - margin),
            min(reference.shape[1], int(head_cols.max()) + margin + 1),
        )
        crop_reference = reference[row_slice, col_slice]
        finite = crop_reference[np.isfinite(crop_reference)]
        if finite.size:
            vmin, vmax = np.percentile(finite, [1, 99.7])
        else:
            vmin, vmax = 0.0, 1.0

        figure = plt.figure(
            figsize=(16, 3.0 * bin_rows + 4.5), constrained_layout=True
        )
        grid = figure.add_gridspec(
            bin_rows + 1, bin_columns, height_ratios=[1.0] * bin_rows + [1.25]
        )
        for index in range(n_bins):
            axis = figure.add_subplot(
                grid[index // bin_columns, index % bin_columns]
            )
            axis.imshow(
                stack[index][row_slice, col_slice],
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.contour(
                spine.head_mask[row_slice, col_slice],
                levels=[0.5],
                colors=["#d14d8b"],
                linewidths=1.2,
            )
            chunk = chunks[index]
            suffix = " partial" if chunk["partial_bin"] else ""
            axis.set_title(
                f"bin {chunk['bin_index']} · "
                f"{chunk['start_seconds'] / 60:.1f} min{suffix}",
                fontsize=9,
            )
            axis.set_axis_off()

        trajectory_axis = figure.add_subplot(grid[bin_rows, :3])
        modes = sorted({str(row["projection_mode"]) for row in spine_rows})
        for mode in modes:
            selected = sorted(
                [row for row in spine_rows if row["projection_mode"] == mode],
                key=lambda row: int(row["bin_index"]),
            )
            x = np.asarray([float(row["start_seconds"]) / 60 for row in selected])
            y = np.asarray([float(row["relative_wright_area"]) for row in selected])
            trajectory_axis.plot(x, y, marker="o", label=mode)
        trajectory_axis.axhline(1.0, color="0.5", linewidth=1)
        trajectory_axis.axhline(0.75, color="#b45309", linestyle="--", linewidth=1)
        trajectory_axis.axhline(1.5, color="#0f766e", linestyle="--", linewidth=1)
        trajectory_axis.set_xlabel("recording time (min)")
        trajectory_axis.set_ylabel("Wright optical-area ratio")
        trajectory_axis.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.18),
            ncols=2,
            frameon=False,
        )
        trajectory_axis.grid(alpha=0.2)

        context_axis = figure.add_subplot(grid[bin_rows, 3:])
        context_axis.imshow(
            crop_reference,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        context_axis.contour(
            spine.head_mask[row_slice, col_slice],
            levels=[0.5],
            colors=["#d14d8b"],
            linewidths=1.5,
        )
        centers = spine.shaft_centers_rc.copy()
        if len(centers):
            centers[:, 0] -= row_slice.start
            centers[:, 1] -= col_slice.start
            inside = (
                (centers[:, 0] >= 0)
                & (centers[:, 0] < crop_reference.shape[0])
                & (centers[:, 1] >= 0)
                & (centers[:, 1] < crop_reference.shape[1])
            )
            context_axis.scatter(
                centers[inside, 1],
                centers[inside, 0],
                s=14,
                facecolors="none",
                edgecolors="#eab308",
                linewidths=0.8,
                label="shaft disks",
            )
        context_axis.set_title("fixed head ROI and local shaft samples")
        context_axis.set_axis_off()
        figure.suptitle(
            f"{spine_id} · {acq_label or Path(directory).name} DMD{dmd}\n"
            "Montage uses aligned all-frame display means; trajectory uses the "
            "selected measurement projection(s).",
            fontsize=13,
        )
        output = target / f"{_safe_name(spine_id)}.png"
        figure.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(figure)
        written.append(output)
    return written
