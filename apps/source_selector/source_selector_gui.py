# source_selector_gui.py
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


@dataclass
class _BBox:
    y0: int
    y1: int
    x0: int
    x1: int


class SourceSelector:
    def __init__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        output_path: str = "selected_source_ids.txt",
        show_centroids: bool = True,
        centroid_marker_size: int = 12,
    ) -> None:
        if image.ndim != 2:
            raise ValueError("image must be 2D array (H, W)")
        if masks.ndim != 3:
            raise ValueError("masks must be 3D array (N_sources, H, W)")
        if masks.shape[1:] != image.shape:
            raise ValueError("masks spatial shape must match image shape")
        self.image = image
        self.masks = masks.astype(bool, copy=False)
        self.N, self.H, self.W = self.masks.shape
        self.output_path = output_path

        # Precompute bounding boxes and centroids
        self._bboxes: list[_BBox] = []
        centroids = []
        for i in range(self.N):
            yy, xx = np.where(self.masks[i])
            if yy.size == 0:
                self._bboxes.append(_BBox(0, -1, 0, -1))
                centroids.append((np.nan, np.nan))
                continue
            self._bboxes.append(
                _BBox(int(yy.min()), int(yy.max()), int(xx.min()), int(xx.max()))
            )
            centroids.append((float(xx.mean()), float(yy.mean())))
        self._centroids = np.asarray(centroids, dtype=float)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # get the vmin and vmax from the image
        vmin = np.nanpercentile(self.image, 0)
        vmax = np.nanpercentile(self.image, 85)
        im = self.ax.imshow(
            self.image, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        im.set_cmap(cmap)

        if show_centroids and self._centroids.size:
            xy = self._centroids
            valid = ~np.isnan(xy).any(axis=1)
            self.ax.scatter(
                xy[valid, 0],
                xy[valid, 1],
                s=centroid_marker_size,
                edgecolors="white",
                linewidths=0.5,
            )

        self.ax.set_title("Lasso-select sources. Press 'c' to clear, 'q' to quit.")
        self.ax.set_xlim(0, self.W - 1)
        self.ax.set_ylim(self.H - 1, 0)

        self._poly_artist = None
        self.lasso = LassoSelector(self.ax, onselect=self._on_select)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "c":
            self._clear_selection()

    def _clear_selection(self):
        if self._poly_artist is not None:
            self._poly_artist.remove()
            self._poly_artist = None
            self.fig.canvas.draw_idle()
        self.ax.set_title("Lasso-select sources. Press 'c' to clear, 'q' to quit.")

    def _on_select(self, verts: list[tuple[float, float]]):
        if verts is None or len(verts) < 3:
            return
        poly = np.asarray(verts)
        if self._poly_artist is not None:
            self._poly_artist.remove()
        self._poly_artist = self.ax.fill(
            poly[:, 0], poly[:, 1], alpha=0.2, linewidth=1.5
        )[0]
        selected_ids = self._ids_in_polygon(poly)
        self._write_ids(selected_ids)
        self.ax.set_title(
            f"Selected {len(selected_ids)} sources: {selected_ids}  (written to {self.output_path})"
        )
        self.fig.canvas.draw_idle()

    def _ids_in_polygon(self, polygon_xy: np.ndarray) -> list[int]:
        x_min = max(int(np.floor(polygon_xy[:, 0].min())), 0)
        x_max = min(int(np.ceil(polygon_xy[:, 0].max())), self.W - 1)
        y_min = max(int(np.floor(polygon_xy[:, 1].min())), 0)
        y_max = min(int(np.ceil(polygon_xy[:, 1].max())), self.H - 1)
        if x_max < x_min or y_max < y_min:
            return []
        path = Path(polygon_xy)
        xs = np.arange(x_min, x_max + 1)
        ys = np.arange(y_min, y_max + 1)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.c_[XX.ravel(), YY.ravel()]
        inside = path.contains_points(pts)
        sel = np.zeros((self.H, self.W), dtype=bool)
        sel[y_min : y_max + 1, x_min : x_max + 1] = inside.reshape((len(ys), len(xs)))
        picked: list[int] = []
        for i, bb in enumerate(self._bboxes):
            if bb.y1 < y_min or bb.y0 > y_max or bb.x1 < x_min or bb.x0 > x_max:
                continue
            if bb.y1 < bb.y0 or bb.x1 < bb.x0:
                continue
            sub_sel = sel[bb.y0 : bb.y1 + 1, bb.x0 : bb.x1 + 1]
            sub_mask = self.masks[i, bb.y0 : bb.y1 + 1, bb.x0 : bb.x1 + 1]
            if np.any(sub_sel & sub_mask):
                picked.append(i)
        return picked

    def _write_ids(self, ids):
        ids = sorted(map(int, ids))
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, ids)))
        print(
            f"[source_selector_gui] wrote {len(ids)} IDs to {self.output_path}: {ids}"
        )

    def show(self):
        plt.show()


def run_selector(
    image,
    masks,
    output_path="selected_source_ids.txt",
    show_centroids=True,
    centroid_marker_size=12,
):
    sel = SourceSelector(
        image=image,
        masks=masks,
        output_path=output_path,
        show_centroids=show_centroids,
        centroid_marker_size=centroid_marker_size,
    )
    sel.show()
    return sel
