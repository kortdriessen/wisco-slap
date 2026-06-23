"""PySide6 scene editor for the synapse float-off animation.

A single window where the user places both DMDs' mean images on a canvas of
their chosen size, drags target strips to where the synapses should land,
tweaks animation/output settings, saves the layout to a YAML config, and
optionally renders the ``.mov`` from inside the GUI.

Imported lazily — the rest of the subpackage doesn't pay the Qt cost unless
you actually open the editor.
"""

from __future__ import annotations

import os
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import slap2_py as spy
from matplotlib.colors import LogNorm

from PySide6.QtCore import (
    QObject, QPointF, QRectF, QSize, QThread, Qt, Signal,
)
from PySide6.QtGui import (
    QAction, QBrush, QColor, QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPainterPath, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem,
    QGraphicsItemGroup, QGraphicsPathItem, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QStatusBar, QVBoxLayout, QWidget,
)

from wisco_slap.meta.get import esum_mirror_path
from wisco_slap.viz.synapto_overlays import _crop_bbox

from .config import (
    DMD_KEYS, AcquisitionConfig, CanvasConfig, OverlayKwargs, Placement,
    ReferenceImage, SceneConfig, TargetStrip,
    load as load_config, save as save_config,
)
from .render import (
    LandingInfo, _dend_blocks, _strip_param, landing_info_for_dmd, render_scene,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gray_pixmap(
    img: np.ndarray, log_vmin: float, log_vmax_pct: float,
) -> QPixmap:
    """Render a 2D float image as an RGBA QPixmap with LogNorm + gray cmap.
    NaN pixels → fully transparent."""
    vmax = float(np.nanpercentile(img, log_vmax_pct))
    nan_mask = np.isnan(img)
    cmap = plt.get_cmap("gray")
    norm = LogNorm(vmin=log_vmin, vmax=vmax)
    rgba = cmap(norm(np.where(nan_mask, log_vmin, img)))
    rgba_u8 = (rgba * 255).astype(np.uint8)
    rgba_u8[..., 3] = np.where(nan_mask, 0, 255)
    rgba_u8 = np.ascontiguousarray(rgba_u8)
    h, w = rgba_u8.shape[:2]
    qimg = QImage(rgba_u8.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


def _load_dmd_pixmaps(
    subject: str, exp: str, loc: str, acq: str, overlay: OverlayKwargs,
) -> dict[str, tuple[QPixmap, np.ndarray]]:
    """Load grayscale cropped mean-image pixmaps + the raw cropped mean image
    arrays for both DMDs."""
    esum_p = esum_mirror_path(subject, exp, loc, acq)
    mim = spy.xsum.get_meanIM(esum_p)
    out: dict[str, tuple[QPixmap, np.ndarray]] = {}
    for dmd_int, key in zip((1, 2), DMD_KEYS):
        img = mim[dmd_int][overlay.mean_im_channel]
        rmin, rmax, cmin, cmax = _crop_bbox(img, overlay.crop_buf_px)
        img_crop = img[rmin:rmax + 1, cmin:cmax + 1]
        pix = _gray_pixmap(img_crop, overlay.log_vmin, overlay.log_vmax_pct)
        out[key] = (pix, img_crop)
    return out


# ---------------------------------------------------------------------------
# Scene items
# ---------------------------------------------------------------------------


class CanvasFrameItem(QGraphicsRectItem):
    """Non-interactive frame showing the canvas bounds."""

    def __init__(self, w: int, h: int) -> None:
        super().__init__(0, 0, w, h)
        pen = QPen(QColor(120, 120, 120))
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(0)
        self.setPen(pen)
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setZValue(-10)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)


class _PixmapCallbackProxy(QObject):
    moved = Signal(str, float, float)  # dmd_key, x, y


class _RefCallbackProxy(QObject):
    moved = Signal(int, float, float)  # ref_index, x, y


class ReferenceImageItem(QGraphicsPixmapItem):
    """Draggable reference image (PNG of a plot, used as a layout guide)."""

    def __init__(self, index: int, pixmap: QPixmap, ref: ReferenceImage) -> None:
        super().__init__(pixmap)
        self._index = index
        self.proxy = _RefCallbackProxy()
        self._silent = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(ref.x, ref.y)
        self.setScale(ref.scale)
        self.setOpacity(ref.opacity if ref.visible else 0.0)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        # Z-order: below = -5 (under mean images), above = +5 (over everything)
        self.setZValue(-5 if ref.z_order == "below" else 5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._silent:
                p: QPointF = self.pos()
                self.proxy.moved.emit(self._index, float(p.x()), float(p.y()))
        return super().itemChange(change, value)


class MeanImageItem(QGraphicsPixmapItem):
    """Draggable pixmap for one DMD's cropped mean image."""

    def __init__(self, dmd_key: str, pixmap: QPixmap, x: float, y: float,
                 scale: float) -> None:
        super().__init__(pixmap)
        self._dmd_key = dmd_key
        self.proxy = _PixmapCallbackProxy()
        self._silent = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(x, y)
        self.setScale(scale)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.setZValue(1)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._silent:
                p: QPointF = self.pos()
                self.proxy.moved.emit(self._dmd_key, float(p.x()), float(p.y()))
        return super().itemChange(change, value)


# -- Target-strip item with corner resize handles ------------------------------


class _ResizeHandle(QGraphicsRectItem):
    """Small square draggable child item used as a corner resize handle."""

    SIZE = 10  # px on screen

    def __init__(self, parent: "TargetStripItem", corner: str) -> None:
        s = _ResizeHandle.SIZE
        super().__init__(-s / 2, -s / 2, s, s, parent)
        self._corner = corner  # 'tl' | 'tr' | 'bl' | 'br'
        self._parent_strip = parent
        self._silent = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setBrush(QBrush(QColor(255, 255, 255)))
        pen = QPen(QColor(0, 0, 0))
        pen.setWidth(1)
        self.setPen(pen)
        self.setZValue(2)
        self.setCursor(Qt.CursorShape.SizeAllCursor)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._silent:
                self._parent_strip._handle_moved(self._corner, self.pos())
        return super().itemChange(change, value)


class _RectCallbackProxy(QObject):
    changed = Signal(str, float, float, float, float)  # dmd_key, x0, y0, x1, y1


class TargetStripItem(QGraphicsRectItem):
    """Rectangle that can be dragged and resized via 4 corner handles.

    The rect's geometry (`x0, y0, x1, y1`) is reported via the `changed` signal.
    Internally maintained as a QRectF in scene coords.
    """

    def __init__(
        self, dmd_key: str, x0: float, y0: float, x1: float, y1: float,
        color: QColor,
    ) -> None:
        super().__init__()
        self._dmd_key = dmd_key
        self.proxy = _RectCallbackProxy()
        self._color = color
        pen = QPen(color)
        pen.setWidth(0)
        pen.setStyle(Qt.PenStyle.SolidLine)
        self.setPen(pen)
        c = QColor(color)
        c.setAlpha(40)
        self.setBrush(QBrush(c))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(3)
        # Children handles
        self._handles: dict[str, _ResizeHandle] = {
            "tl": _ResizeHandle(self, "tl"),
            "tr": _ResizeHandle(self, "tr"),
            "bl": _ResizeHandle(self, "bl"),
            "br": _ResizeHandle(self, "br"),
        }
        self._suppress_emit = False
        self.set_endpoints(x0, y0, x1, y1)

    def _bounds(self) -> tuple[float, float, float, float]:
        r = self.rect()
        p = self.pos()
        return (p.x() + r.left(), p.y() + r.top(),
                p.x() + r.right(), p.y() + r.bottom())

    def endpoints(self) -> tuple[float, float, float, float]:
        return self._bounds()

    def set_endpoints(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Set the rect spanning ``(x0, y0)`` to ``(x1, y1)`` in scene coords."""
        self._suppress_emit = True
        try:
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])
            self.setPos(xmin, ymin)
            self.setRect(0, 0, xmax - xmin, ymax - ymin)
            self._reposition_handles()
        finally:
            self._suppress_emit = False
        self._emit()

    def _reposition_handles(self) -> None:
        r = self.rect()
        positions = {
            "tl": (r.left(), r.top()),
            "tr": (r.right(), r.top()),
            "bl": (r.left(), r.bottom()),
            "br": (r.right(), r.bottom()),
        }
        for corner, (x, y) in positions.items():
            h = self._handles[corner]
            h._silent = True
            h.setPos(x, y)
            h._silent = False

    def _handle_moved(self, corner: str, new_local_pos: QPointF) -> None:
        """A handle was dragged; resize the parent rect accordingly."""
        r = self.rect()
        x_left, x_right = r.left(), r.right()
        y_top, y_bot = r.top(), r.bottom()
        nx, ny = new_local_pos.x(), new_local_pos.y()
        if corner == "tl":
            x_left, y_top = nx, ny
        elif corner == "tr":
            x_right, y_top = nx, ny
        elif corner == "bl":
            x_left, y_bot = nx, ny
        elif corner == "br":
            x_right, y_bot = nx, ny
        if x_right < x_left:
            x_left, x_right = x_right, x_left
        if y_bot < y_top:
            y_top, y_bot = y_bot, y_top
        self._suppress_emit = True
        try:
            self.setRect(x_left, y_top, x_right - x_left, y_bot - y_top)
            self._reposition_handles()
        finally:
            self._suppress_emit = False
        self._emit()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if not self._suppress_emit:
                self._emit()
        return super().itemChange(change, value)

    def _emit(self) -> None:
        x0, y0, x1, y1 = self._bounds()
        self.proxy.changed.emit(self._dmd_key, x0, y0, x1, y1)


# -- Landing-preview item ------------------------------------------------------


class LandingPreviewItem(QGraphicsItemGroup):
    """Static markers at the synapse landing positions for one DMD.
    Group of dot or bar children — wholly recomputed on each refresh."""

    def __init__(self, dmd_key: str, color: QColor) -> None:
        super().__init__()
        self._dmd_key = dmd_key
        self._color = color
        self.setZValue(2)

    def refresh(self, info: LandingInfo, target: TargetStrip, cmap_name: str) -> None:
        # Remove existing children
        for child in list(self.childItems()):
            child.setParentItem(None)
            scene = self.scene()
            if scene is not None:
                scene.removeItem(child)

        if not info.syn_ids:
            return

        cmap = plt.get_cmap(cmap_name)

        if target.end_mode == "dots":
            for i, _ in enumerate(info.syn_ids):
                npos = info.pos_norm[i] if np.isfinite(info.pos_norm[i]) else 0.5
                rgba = cmap(npos)
                color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
                r = max(1.5, target.dot_radius_px)
                d = QGraphicsRectItem(-r, -r, 2 * r, 2 * r, self)
                d.setBrush(QBrush(color))
                pen = QPen(QColor(0, 0, 0, 0))
                pen.setWidth(0)
                d.setPen(pen)
                ex, ey = info.end_xy[i]
                d.setPos(ex, ey)
        else:  # dendrite_bars
            dend_arr = np.asarray(info.dend_ids, dtype=object)
            blocks = _dend_blocks(dend_arr)
            n_blocks = len(blocks)
            ts = target
            total_gap = ts.bar_gap_frac * (n_blocks - 1) if n_blocks > 1 else 0.0
            usable = max(0.0, 1.0 - total_gap)
            block_lengths = [b[2] - b[1] + 1 for b in blocks]
            total_syn = sum(block_lengths)
            gap = ts.bar_gap_frac if n_blocks > 1 else 0.0
            cursor = 0.0
            thickness = ts.bar_thickness_px
            half_t = thickness / 2.0
            for (_, i_start, i_end), bl in zip(blocks, block_lengths):
                seg_len = (bl / total_syn) * usable
                p_start = _strip_param(ts, cursor)
                p_end = _strip_param(ts, cursor + seg_len)
                cursor += seg_len + gap

                pos_block = np.asarray(info.pos_norm[i_start:i_end + 1], dtype=float)
                finite = pos_block[np.isfinite(pos_block)]
                pmin, pmax = (0.0, 1.0) if (finite.size == 0 or finite.min() == finite.max()) \
                    else (float(finite.min()), float(finite.max()))

                # Build a rounded-end (stadium) bar via QPainterPath:
                # axis-align to strip's dominant axis, corner radius = thickness/2.
                if ts.is_horizontal:
                    x_lo, x_hi = sorted([p_start[0], p_end[0]])
                    y_mid = 0.5 * (p_start[1] + p_end[1])
                    bar_rect = QRectF(x_lo, y_mid - half_t, max(1.0, x_hi - x_lo), thickness)
                    grad = QLinearGradient(QPointF(x_lo, y_mid), QPointF(x_hi, y_mid))
                else:
                    y_lo, y_hi = sorted([p_start[1], p_end[1]])
                    x_mid = 0.5 * (p_start[0] + p_end[0])
                    bar_rect = QRectF(x_mid - half_t, y_lo, thickness, max(1.0, y_hi - y_lo))
                    grad = QLinearGradient(QPointF(x_mid, y_lo), QPointF(x_mid, y_hi))

                # Gradient stops from pmin → pmax through cmap
                n_stops = 8
                for s in range(n_stops + 1):
                    tt = s / n_stops
                    rgba = cmap(pmin + (pmax - pmin) * tt)
                    grad.setColorAt(tt, QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3]))

                path = QPainterPath()
                path.addRoundedRect(bar_rect, half_t, half_t)
                item = QGraphicsPathItem(path, self)
                item.setBrush(QBrush(grad))
                item.setPen(QPen(Qt.PenStyle.NoPen))


# ---------------------------------------------------------------------------
# QGraphicsView with pan + zoom (WISynaptic pattern)
# ---------------------------------------------------------------------------


class CanvasView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._zoom = 1.0
        self.setBackgroundBrush(QBrush(QColor(50, 50, 50)))

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        super().keyReleaseEvent(event)


# ---------------------------------------------------------------------------
# New-scene dialog (asks for acq + canvas size)
# ---------------------------------------------------------------------------


class NewSceneDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New scene")
        form = QFormLayout(self)
        self.subject = QLineEdit("alnair")
        self.exp = QLineEdit("exp_1")
        self.loc = QLineEdit("loc_D")
        self.acq = QLineEdit("acq_1")
        self.width = QSpinBox(); self.width.setRange(64, 16384); self.width.setValue(1920)
        self.height = QSpinBox(); self.height.setRange(64, 16384); self.height.setValue(1080)
        self.soma = QLineEdit("soma1")
        form.addRow("Subject", self.subject)
        form.addRow("Exp", self.exp)
        form.addRow("Loc", self.loc)
        form.addRow("Acq", self.acq)
        form.addRow("Canvas width", self.width)
        form.addRow("Canvas height", self.height)
        form.addRow("Soma filter (optional)", self.soma)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def make_config(self) -> SceneConfig:
        cfg = SceneConfig.default(
            self.subject.text().strip(), self.exp.text().strip(),
            self.loc.text().strip(), self.acq.text().strip(),
            canvas_size=(self.width.value(), self.height.value()),
        )
        soma = self.soma.text().strip()
        cfg.overlay_kwargs.soma = soma if soma else None
        return cfg


# ---------------------------------------------------------------------------
# Render worker
# ---------------------------------------------------------------------------


class _RenderWorker(QThread):
    progress = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, config: SceneConfig, parent=None) -> None:
        super().__init__(parent)
        self._config = config

    def run(self) -> None:
        try:
            self.progress.emit(f"Rendering → {self._config.output.path}")
            out = render_scene(self._config, progress=False)
            self.finished_ok.emit(str(out))
        except Exception as e:
            tb = traceback.format_exc()
            self.failed.emit(f"{e}\n{tb}")


# ---------------------------------------------------------------------------
# Main editor window
# ---------------------------------------------------------------------------


_DMD_COLORS: dict[str, QColor] = {
    "dmd_1": QColor(200, 100, 50),    # orange
    "dmd_2": QColor(60, 130, 200),    # blue
}


class EditorWindow(QMainWindow):
    def __init__(self, config: SceneConfig, config_path: str | None = None) -> None:
        super().__init__()
        self._config = config
        self._config_path: str | None = config_path
        self.setWindowTitle("synapto_anim — scene editor")
        self.resize(1600, 1000)

        # Scene + view
        self._scene = QGraphicsScene(0, 0, config.canvas.width, config.canvas.height)
        self._scene.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        self._view = CanvasView(self._scene)
        self.setCentralWidget(self._view)

        # Items
        self._frame_item = CanvasFrameItem(config.canvas.width, config.canvas.height)
        self._scene.addItem(self._frame_item)
        self._image_items: dict[str, MeanImageItem] = {}
        self._target_items: dict[str, TargetStripItem] = {}
        self._landing_items: dict[str, LandingPreviewItem] = {}
        self._cached_pixmaps: dict[str, tuple[QPixmap, np.ndarray]] = {}
        self._landing_info_cache: dict[str, LandingInfo] = {}
        self._ref_items: list[ReferenceImageItem] = []
        self._current_ref: int = -1   # currently selected reference index in the list

        # Side panels
        self._build_docks()
        self._build_menu()
        self._build_status()

        # Populate scene from the config
        self._populate_scene()
        self._refresh_panels()
        self._view.fitInView(self._frame_item, Qt.AspectRatioMode.KeepAspectRatio)

    # -- UI scaffolding -----------------------------------------------------

    def _build_docks(self) -> None:
        # Properties panel
        self._properties_dock = QDockWidget("Properties", self)
        pw = QWidget()
        pl = QVBoxLayout(pw)

        # Acquisition
        ag = QGroupBox("Acquisition")
        af = QFormLayout(ag)
        self._ed_subject = QLineEdit(); self._ed_subject.editingFinished.connect(self._sync_acq_from_ui)
        self._ed_exp = QLineEdit(); self._ed_exp.editingFinished.connect(self._sync_acq_from_ui)
        self._ed_loc = QLineEdit(); self._ed_loc.editingFinished.connect(self._sync_acq_from_ui)
        self._ed_acq = QLineEdit(); self._ed_acq.editingFinished.connect(self._sync_acq_from_ui)
        self._ed_soma = QLineEdit(); self._ed_soma.editingFinished.connect(self._sync_overlay_from_ui)
        self._chk_color_by_pos = QCheckBox(); self._chk_color_by_pos.stateChanged.connect(self._sync_overlay_from_ui)
        self._cb_cmap = QComboBox()
        for c in ("summer", "viridis", "plasma", "magma", "cividis"):
            self._cb_cmap.addItem(c)
        self._cb_cmap.currentTextChanged.connect(self._sync_overlay_from_ui)
        af.addRow("Subject", self._ed_subject)
        af.addRow("Exp", self._ed_exp)
        af.addRow("Loc", self._ed_loc)
        af.addRow("Acq", self._ed_acq)
        af.addRow("Soma filter", self._ed_soma)
        af.addRow("Color by pos", self._chk_color_by_pos)
        af.addRow("Colormap", self._cb_cmap)
        btn_reload = QPushButton("Reload images + previews")
        btn_reload.clicked.connect(self._reload_acquisition)
        af.addRow(btn_reload)
        pl.addWidget(ag)

        # Canvas
        cg = QGroupBox("Canvas")
        cf = QFormLayout(cg)
        self._sp_canvas_w = QSpinBox(); self._sp_canvas_w.setRange(64, 16384)
        self._sp_canvas_h = QSpinBox(); self._sp_canvas_h.setRange(64, 16384)
        self._sp_canvas_w.valueChanged.connect(self._sync_canvas_from_ui)
        self._sp_canvas_h.valueChanged.connect(self._sync_canvas_from_ui)
        cf.addRow("Width", self._sp_canvas_w)
        cf.addRow("Height", self._sp_canvas_h)
        pl.addWidget(cg)

        # Per-DMD placements + targets
        self._dmd_groups: dict[str, dict[str, QWidget]] = {}
        for key in DMD_KEYS:
            g = QGroupBox(key.upper())
            f = QFormLayout(g)
            sp_x = QDoubleSpinBox(); sp_x.setRange(-100000, 100000); sp_x.setDecimals(1)
            sp_y = QDoubleSpinBox(); sp_y.setRange(-100000, 100000); sp_y.setDecimals(1)
            sp_scale = QDoubleSpinBox(); sp_scale.setRange(0.05, 20.0); sp_scale.setDecimals(2); sp_scale.setSingleStep(0.1)
            sp_x.valueChanged.connect(lambda v, k=key: self._sync_placement_from_ui(k))
            sp_y.valueChanged.connect(lambda v, k=key: self._sync_placement_from_ui(k))
            sp_scale.valueChanged.connect(lambda v, k=key: self._sync_placement_from_ui(k))
            f.addRow("Image X", sp_x)
            f.addRow("Image Y", sp_y)
            f.addRow("Image scale", sp_scale)

            sp_x0 = QDoubleSpinBox(); sp_x0.setRange(-100000, 100000); sp_x0.setDecimals(1)
            sp_y0 = QDoubleSpinBox(); sp_y0.setRange(-100000, 100000); sp_y0.setDecimals(1)
            sp_x1 = QDoubleSpinBox(); sp_x1.setRange(-100000, 100000); sp_x1.setDecimals(1)
            sp_y1 = QDoubleSpinBox(); sp_y1.setRange(-100000, 100000); sp_y1.setDecimals(1)
            sp_x0.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_y0.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_x1.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_y1.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            f.addRow("Target x0", sp_x0); f.addRow("Target y0", sp_y0)
            f.addRow("Target x1", sp_x1); f.addRow("Target y1", sp_y1)

            cb_end = QComboBox(); cb_end.addItems(["dendrite_bars", "dots"])
            cb_end.currentTextChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_thick = QDoubleSpinBox(); sp_thick.setRange(1, 1000); sp_thick.setDecimals(1)
            sp_thick.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_dot = QDoubleSpinBox(); sp_dot.setRange(0.5, 50); sp_dot.setDecimals(1)
            sp_dot.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            sp_gap = QDoubleSpinBox(); sp_gap.setRange(0, 0.5); sp_gap.setDecimals(3); sp_gap.setSingleStep(0.01)
            sp_gap.valueChanged.connect(lambda v, k=key: self._sync_target_from_ui(k))
            f.addRow("End mode", cb_end)
            f.addRow("Bar thickness", sp_thick)
            f.addRow("Dot radius", sp_dot)
            f.addRow("Bar gap frac", sp_gap)
            pl.addWidget(g)

            self._dmd_groups[key] = dict(
                sp_x=sp_x, sp_y=sp_y, sp_scale=sp_scale,
                sp_x0=sp_x0, sp_y0=sp_y0, sp_x1=sp_x1, sp_y1=sp_y1,
                cb_end=cb_end, sp_thick=sp_thick, sp_dot=sp_dot, sp_gap=sp_gap,
            )

        # References section
        rg = QGroupBox("References (PNG layout guides)")
        rl = QVBoxLayout(rg)
        self._ref_list = QListWidget()
        self._ref_list.currentRowChanged.connect(self._on_ref_selection_changed)
        rl.addWidget(self._ref_list)
        row = QHBoxLayout()
        btn_add_ref = QPushButton("Add…"); btn_add_ref.clicked.connect(self._on_add_reference)
        btn_rm_ref = QPushButton("Remove"); btn_rm_ref.clicked.connect(self._on_remove_reference)
        row.addWidget(btn_add_ref); row.addWidget(btn_rm_ref)
        rl.addLayout(row)

        # Ref property controls
        rf = QFormLayout()
        self._ref_x = QDoubleSpinBox(); self._ref_x.setRange(-100000, 100000); self._ref_x.setDecimals(1)
        self._ref_y = QDoubleSpinBox(); self._ref_y.setRange(-100000, 100000); self._ref_y.setDecimals(1)
        self._ref_scale = QDoubleSpinBox(); self._ref_scale.setRange(0.01, 50); self._ref_scale.setDecimals(3); self._ref_scale.setSingleStep(0.05)
        self._ref_opacity = QDoubleSpinBox(); self._ref_opacity.setRange(0, 1); self._ref_opacity.setDecimals(2); self._ref_opacity.setSingleStep(0.05)
        self._ref_visible = QCheckBox()
        self._ref_include = QCheckBox()
        self._ref_z = QComboBox(); self._ref_z.addItems(["below", "above"])
        for w in (self._ref_x, self._ref_y, self._ref_scale, self._ref_opacity):
            w.valueChanged.connect(self._sync_reference_from_ui)
        self._ref_visible.stateChanged.connect(self._sync_reference_from_ui)
        self._ref_include.stateChanged.connect(self._sync_reference_from_ui)
        self._ref_z.currentTextChanged.connect(self._sync_reference_from_ui)
        rf.addRow("Ref X", self._ref_x)
        rf.addRow("Ref Y", self._ref_y)
        rf.addRow("Ref scale", self._ref_scale)
        rf.addRow("Ref opacity", self._ref_opacity)
        rf.addRow("Visible (editor)", self._ref_visible)
        rf.addRow("Include in render", self._ref_include)
        rf.addRow("Z-order", self._ref_z)
        rl.addLayout(rf)
        pl.addWidget(rg)

        pl.addStretch(1)
        prop_scroll = QScrollArea()
        prop_scroll.setWidget(pw)
        prop_scroll.setWidgetResizable(True)
        prop_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        prop_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._properties_dock.setWidget(prop_scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._properties_dock)

        # Animation/output panel
        self._anim_dock = QDockWidget("Animation + Output", self)
        aw = QWidget()
        al = QFormLayout(aw)
        self._sp_dur = QDoubleSpinBox(); self._sp_dur.setRange(0.1, 60); self._sp_dur.setDecimals(2)
        self._sp_fps = QSpinBox(); self._sp_fps.setRange(1, 120)
        self._cb_stagger = QComboBox(); self._cb_stagger.addItems(["wave", "random", "none"])
        self._sp_stagfrac = QDoubleSpinBox(); self._sp_stagfrac.setRange(0, 0.9); self._sp_stagfrac.setDecimals(2); self._sp_stagfrac.setSingleStep(0.05)
        self._sp_morphfrac = QDoubleSpinBox(); self._sp_morphfrac.setRange(0, 0.9); self._sp_morphfrac.setDecimals(2); self._sp_morphfrac.setSingleStep(0.05)
        self._sp_seed = QSpinBox(); self._sp_seed.setRange(0, 999999)
        self._cb_codec = QComboBox(); self._cb_codec.addItems(["prores_ks", "qtrle", "png"])
        self._ed_outpath = QLineEdit()
        btn_browse = QPushButton("Browse…"); btn_browse.clicked.connect(self._browse_output)
        outrow = QHBoxLayout(); outrow.addWidget(self._ed_outpath); outrow.addWidget(btn_browse)
        outw = QWidget(); outw.setLayout(outrow)

        for w in (self._sp_dur, self._sp_fps, self._sp_stagfrac, self._sp_morphfrac, self._sp_seed):
            w.valueChanged.connect(self._sync_anim_from_ui)
        self._cb_stagger.currentTextChanged.connect(self._sync_anim_from_ui)
        self._cb_codec.currentTextChanged.connect(self._sync_anim_from_ui)
        self._ed_outpath.editingFinished.connect(self._sync_anim_from_ui)

        al.addRow("Duration (s) — flight+morph", self._sp_dur)
        al.addRow("FPS", self._sp_fps)
        al.addRow("Stagger", self._cb_stagger)
        al.addRow("Stagger frac", self._sp_stagfrac)
        al.addRow("Morph frac", self._sp_morphfrac)
        al.addRow("Seed", self._sp_seed)
        al.addRow("Codec", self._cb_codec)
        al.addRow("Output path", outw)

        # Intro phases
        self._chk_intro = QCheckBox()
        self._sp_intro_open = QDoubleSpinBox(); self._sp_intro_open.setRange(0, 30); self._sp_intro_open.setDecimals(2); self._sp_intro_open.setSingleStep(0.1)
        self._sp_intro_close = QDoubleSpinBox(); self._sp_intro_close.setRange(0, 30); self._sp_intro_close.setDecimals(2); self._sp_intro_close.setSingleStep(0.1)
        self._sp_intro_syn_fade = QDoubleSpinBox(); self._sp_intro_syn_fade.setRange(0.01, 2.0); self._sp_intro_syn_fade.setDecimals(3); self._sp_intro_syn_fade.setSingleStep(0.01)
        self._sp_intro_syn_gap = QDoubleSpinBox(); self._sp_intro_syn_gap.setRange(0.00, 2.0); self._sp_intro_syn_gap.setDecimals(3); self._sp_intro_syn_gap.setSingleStep(0.01)
        self._sp_intro_dend_gap = QDoubleSpinBox(); self._sp_intro_dend_gap.setRange(-5.0, 5.0); self._sp_intro_dend_gap.setDecimals(2); self._sp_intro_dend_gap.setSingleStep(0.05)
        self._sp_intro_dend_gap.setToolTip("Pause between dendrites. Negative values overlap dendrites (next starts before previous finishes).")
        self._lbl_intro_total = QLabel("(auto)")
        for w in (self._sp_intro_open, self._sp_intro_close,
                  self._sp_intro_syn_fade, self._sp_intro_syn_gap, self._sp_intro_dend_gap):
            w.valueChanged.connect(self._sync_anim_from_ui)
        self._chk_intro.stateChanged.connect(self._sync_anim_from_ui)
        al.addRow("Intro: enabled", self._chk_intro)
        al.addRow("Intro: open pause (s)", self._sp_intro_open)
        al.addRow("  per-synapse fade (s)", self._sp_intro_syn_fade)
        al.addRow("  per-synapse gap (s)", self._sp_intro_syn_gap)
        al.addRow("  per-dendrite gap (s)", self._sp_intro_dend_gap)
        al.addRow("  (computed appear)", self._lbl_intro_total)
        al.addRow("Intro: close pause (s)", self._sp_intro_close)

        btn_render = QPushButton("Render now")
        btn_render.clicked.connect(self._on_render)
        al.addRow(btn_render)
        anim_scroll = QScrollArea()
        anim_scroll.setWidget(aw)
        anim_scroll.setWidgetResizable(True)
        anim_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        anim_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._anim_dock.setWidget(anim_scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._anim_dock)

        # Log panel
        self._log_dock = QDockWidget("Log", self)
        self._log = QPlainTextEdit(); self._log.setReadOnly(True)
        self._log_dock.setWidget(self._log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("File")
        act_new = QAction("&New scene…", self); act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new); m_file.addAction(act_new)
        act_open = QAction("&Open…", self); act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._on_open); m_file.addAction(act_open)
        act_save = QAction("&Save", self); act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._on_save); m_file.addAction(act_save)
        act_sa = QAction("Save &As…", self); act_sa.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_sa.triggered.connect(self._on_save_as); m_file.addAction(act_sa)
        m_file.addSeparator()
        act_render = QAction("&Render…", self); act_render.setShortcut("Ctrl+R")
        act_render.triggered.connect(self._on_render); m_file.addAction(act_render)
        m_file.addSeparator()
        act_quit = QAction("&Quit", self); act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close); m_file.addAction(act_quit)

    def _build_status(self) -> None:
        self.setStatusBar(QStatusBar(self))
        self._status_label = QLabel("")
        self.statusBar().addWidget(self._status_label)

    # -- Scene population ---------------------------------------------------

    def _populate_scene(self) -> None:
        # Resize scene to current canvas dims
        cw = self._config.canvas.width
        ch = self._config.canvas.height
        self._scene.setSceneRect(0, 0, cw, ch)
        self._frame_item.setRect(0, 0, cw, ch)

        # Load mean-image pixmaps from disk for the current acq
        try:
            self._cached_pixmaps = _load_dmd_pixmaps(
                self._config.acquisition.subject, self._config.acquisition.exp,
                self._config.acquisition.loc, self._config.acquisition.acq,
                self._config.overlay_kwargs,
            )
        except Exception as e:
            self._log_line(f"Failed to load mean images: {e}")
            self._cached_pixmaps = {}

        # Remove old items
        for key in DMD_KEYS:
            if key in self._image_items:
                self._scene.removeItem(self._image_items.pop(key))
            if key in self._target_items:
                self._scene.removeItem(self._target_items.pop(key))
            if key in self._landing_items:
                self._scene.removeItem(self._landing_items.pop(key))
        for item in self._ref_items:
            self._scene.removeItem(item)
        self._ref_items.clear()

        # Image items
        for key in DMD_KEYS:
            if key not in self._cached_pixmaps:
                continue
            pixmap, _ = self._cached_pixmaps[key]
            placement = self._config.placements.get(key)
            if placement is None:
                placement = Placement(x=100, y=100 + (350 if key == "dmd_2" else 0))
                self._config.placements[key] = placement
            item = MeanImageItem(key, pixmap, placement.x, placement.y, placement.scale)
            item.proxy.moved.connect(self._on_image_moved)
            self._scene.addItem(item)
            self._image_items[key] = item

        # Target strip items
        for key in DMD_KEYS:
            target = self._config.targets.get(key)
            if target is None:
                target = TargetStrip(
                    x0=cw * 0.85, y0=80,
                    x1=cw * 0.85, y1=ch * 0.45,
                )
                self._config.targets[key] = target
            color = _DMD_COLORS[key]
            t_item = TargetStripItem(key, target.x0, target.y0, target.x1, target.y1, color)
            t_item.proxy.changed.connect(self._on_target_changed)
            self._scene.addItem(t_item)
            self._target_items[key] = t_item

        # Landing previews
        for key in DMD_KEYS:
            lp = LandingPreviewItem(key, _DMD_COLORS[key])
            self._scene.addItem(lp)
            self._landing_items[key] = lp

        # Reference images (PNG layout guides)
        for i, ref in enumerate(self._config.references):
            item = self._create_ref_item(i, ref)
            if item is not None:
                self._ref_items.append(item)
                self._scene.addItem(item)

        # Compute landing info
        self._refresh_landing_info()

    def _refresh_landing_info(self) -> None:
        a = self._config.acquisition
        self._landing_info_cache = {}
        for key in DMD_KEYS:
            target = self._config.targets.get(key)
            if target is None:
                continue
            try:
                info = landing_info_for_dmd(
                    a.subject, a.exp, a.loc, a.acq,
                    dmd=int(key.split("_")[1]), target=target,
                    overlay_kwargs=self._config.overlay_kwargs,
                )
                self._landing_info_cache[key] = info
                self._landing_items[key].refresh(info, target, self._config.overlay_kwargs.cmap)
            except Exception as e:
                self._log_line(f"Failed landing preview for {key}: {e}")

    # -- Panel ↔ config sync (read panels → write config) ------------------

    def _sync_acq_from_ui(self) -> None:
        a = self._config.acquisition
        a.subject = self._ed_subject.text().strip()
        a.exp = self._ed_exp.text().strip()
        a.loc = self._ed_loc.text().strip()
        a.acq = self._ed_acq.text().strip()
        self._status_label.setText(f"{a.subject}/{a.exp}/{a.loc}/{a.acq}")

    def _sync_overlay_from_ui(self) -> None:
        ok = self._config.overlay_kwargs
        s = self._ed_soma.text().strip()
        ok.soma = s if s else None
        ok.color_by_pos = self._chk_color_by_pos.isChecked()
        ok.cmap = self._cb_cmap.currentText()

    def _sync_canvas_from_ui(self) -> None:
        self._config.canvas.width = int(self._sp_canvas_w.value())
        self._config.canvas.height = int(self._sp_canvas_h.value())
        self._scene.setSceneRect(0, 0, self._config.canvas.width, self._config.canvas.height)
        self._frame_item.setRect(0, 0, self._config.canvas.width, self._config.canvas.height)

    def _sync_placement_from_ui(self, key: str) -> None:
        g = self._dmd_groups[key]
        p = self._config.placements[key]
        p.x = float(g["sp_x"].value())
        p.y = float(g["sp_y"].value())
        p.scale = float(g["sp_scale"].value())
        item = self._image_items.get(key)
        if item is not None:
            item._silent = True
            item.setPos(p.x, p.y)
            item.setScale(p.scale)
            item._silent = False

    def _sync_target_from_ui(self, key: str) -> None:
        g = self._dmd_groups[key]
        t = self._config.targets[key]
        t.x0 = float(g["sp_x0"].value()); t.y0 = float(g["sp_y0"].value())
        t.x1 = float(g["sp_x1"].value()); t.y1 = float(g["sp_y1"].value())
        t.end_mode = g["cb_end"].currentText()
        t.bar_thickness_px = float(g["sp_thick"].value())
        t.dot_radius_px = float(g["sp_dot"].value())
        t.bar_gap_frac = float(g["sp_gap"].value())
        item = self._target_items.get(key)
        if item is not None:
            item.set_endpoints(t.x0, t.y0, t.x1, t.y1)
        info = self._landing_info_cache.get(key)
        if info is not None:
            self._landing_items[key].refresh(info, t, self._config.overlay_kwargs.cmap)

    def _sync_anim_from_ui(self) -> None:
        a = self._config.animation
        a.duration_s = float(self._sp_dur.value())
        a.fps = int(self._sp_fps.value())
        a.stagger = self._cb_stagger.currentText()
        a.stagger_frac = float(self._sp_stagfrac.value())
        a.morph_frac = float(self._sp_morphfrac.value())
        a.seed = int(self._sp_seed.value())
        a.intro_enabled = self._chk_intro.isChecked()
        a.intro_open_pause_s = float(self._sp_intro_open.value())
        a.intro_close_pause_s = float(self._sp_intro_close.value())
        a.intro_per_syn_fade_s = float(self._sp_intro_syn_fade.value())
        a.intro_per_syn_gap_s = float(self._sp_intro_syn_gap.value())
        a.intro_dend_gap_s = float(self._sp_intro_dend_gap.value())
        self._update_intro_total_label()
        self._config.output.codec = self._cb_codec.currentText()
        self._config.output.path = self._ed_outpath.text().strip()

    # -- Config → panels (write config → refresh panels) -------------------

    def _refresh_panels(self) -> None:
        a = self._config.acquisition
        for w, v in (
            (self._ed_subject, a.subject), (self._ed_exp, a.exp),
            (self._ed_loc, a.loc), (self._ed_acq, a.acq),
        ):
            w.blockSignals(True); w.setText(v); w.blockSignals(False)

        ok = self._config.overlay_kwargs
        for w, v in ((self._ed_soma, ok.soma or ""),):
            w.blockSignals(True); w.setText(v); w.blockSignals(False)
        self._chk_color_by_pos.blockSignals(True)
        self._chk_color_by_pos.setChecked(ok.color_by_pos)
        self._chk_color_by_pos.blockSignals(False)
        self._cb_cmap.blockSignals(True)
        i = self._cb_cmap.findText(ok.cmap)
        if i < 0:
            # Custom / user-registered cmap — add it to the dropdown.
            self._cb_cmap.addItem(ok.cmap)
            i = self._cb_cmap.findText(ok.cmap)
        if i >= 0:
            self._cb_cmap.setCurrentIndex(i)
        self._cb_cmap.blockSignals(False)

        self._sp_canvas_w.blockSignals(True); self._sp_canvas_w.setValue(self._config.canvas.width); self._sp_canvas_w.blockSignals(False)
        self._sp_canvas_h.blockSignals(True); self._sp_canvas_h.setValue(self._config.canvas.height); self._sp_canvas_h.blockSignals(False)

        for key in DMD_KEYS:
            g = self._dmd_groups[key]
            p = self._config.placements.get(key)
            t = self._config.targets.get(key)
            if p is not None:
                for w, v in (
                    (g["sp_x"], p.x), (g["sp_y"], p.y), (g["sp_scale"], p.scale),
                ):
                    w.blockSignals(True); w.setValue(v); w.blockSignals(False)
            if t is not None:
                for w, v in (
                    (g["sp_x0"], t.x0), (g["sp_y0"], t.y0),
                    (g["sp_x1"], t.x1), (g["sp_y1"], t.y1),
                    (g["sp_thick"], t.bar_thickness_px),
                    (g["sp_dot"], t.dot_radius_px), (g["sp_gap"], t.bar_gap_frac),
                ):
                    w.blockSignals(True); w.setValue(v); w.blockSignals(False)
                g["cb_end"].blockSignals(True)
                idx = g["cb_end"].findText(t.end_mode)
                if idx >= 0:
                    g["cb_end"].setCurrentIndex(idx)
                g["cb_end"].blockSignals(False)

        an = self._config.animation
        for w, v in (
            (self._sp_dur, an.duration_s), (self._sp_fps, an.fps),
            (self._sp_stagfrac, an.stagger_frac), (self._sp_morphfrac, an.morph_frac),
            (self._sp_seed, an.seed or 0),
            (self._sp_intro_open, an.intro_open_pause_s),
            (self._sp_intro_close, an.intro_close_pause_s),
            (self._sp_intro_syn_fade, an.intro_per_syn_fade_s),
            (self._sp_intro_syn_gap, an.intro_per_syn_gap_s),
            (self._sp_intro_dend_gap, an.intro_dend_gap_s),
        ):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self._chk_intro.blockSignals(True); self._chk_intro.setChecked(an.intro_enabled); self._chk_intro.blockSignals(False)
        self._update_intro_total_label()
        self._cb_stagger.blockSignals(True)
        idx = self._cb_stagger.findText(an.stagger)
        if idx >= 0:
            self._cb_stagger.setCurrentIndex(idx)
        self._cb_stagger.blockSignals(False)
        self._cb_codec.blockSignals(True)
        idx = self._cb_codec.findText(self._config.output.codec)
        if idx >= 0:
            self._cb_codec.setCurrentIndex(idx)
        self._cb_codec.blockSignals(False)
        self._ed_outpath.blockSignals(True); self._ed_outpath.setText(self._config.output.path); self._ed_outpath.blockSignals(False)

        self._refresh_ref_list()
        if self._config.references:
            self._ref_list.setCurrentRow(0)
            self._current_ref = 0
        else:
            self._current_ref = -1
        self._refresh_ref_controls()

        self._status_label.setText(f"{a.subject}/{a.exp}/{a.loc}/{a.acq}  | canvas {self._config.canvas.width}×{self._config.canvas.height}")

    # -- Scene → config (drag-driven) ---------------------------------------

    def _on_image_moved(self, key: str, x: float, y: float) -> None:
        p = self._config.placements[key]
        p.x = x; p.y = y
        g = self._dmd_groups[key]
        g["sp_x"].blockSignals(True); g["sp_x"].setValue(x); g["sp_x"].blockSignals(False)
        g["sp_y"].blockSignals(True); g["sp_y"].setValue(y); g["sp_y"].blockSignals(False)

    def _on_target_changed(self, key: str, x0: float, y0: float, x1: float, y1: float) -> None:
        t = self._config.targets[key]
        t.x0, t.y0, t.x1, t.y1 = x0, y0, x1, y1
        g = self._dmd_groups[key]
        for w, v in ((g["sp_x0"], x0), (g["sp_y0"], y0), (g["sp_x1"], x1), (g["sp_y1"], y1)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        info = self._landing_info_cache.get(key)
        if info is not None:
            self._landing_items[key].refresh(info, t, self._config.overlay_kwargs.cmap)

    # -- Menu / actions -----------------------------------------------------

    def _on_new(self) -> None:
        dlg = NewSceneDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            cfg = dlg.make_config()
            self._replace_config(cfg, path=None)

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            cfg = load_config(path)
        except Exception as e:
            QMessageBox.critical(self, "Open failed", str(e))
            return
        self._replace_config(cfg, path=path)

    def _on_save(self) -> None:
        if not self._config_path:
            self._on_save_as()
            return
        save_config(self._config, self._config_path)
        self._log_line(f"Saved → {self._config_path}")

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", "scene.yaml", "YAML (*.yaml)")
        if not path:
            return
        save_config(self._config, path)
        self._config_path = path
        self.setWindowTitle(f"synapto_anim — {os.path.basename(path)}")
        self._log_line(f"Saved → {path}")

    def _on_render(self) -> None:
        self._sync_anim_from_ui()
        if not self._config.output.path:
            QMessageBox.warning(self, "Render", "Set an output path first.")
            return
        self._worker = _RenderWorker(self._config, self)
        self._worker.progress.connect(self._log_line)
        self._worker.finished_ok.connect(lambda p: self._log_line(f"Render done → {p}"))
        self._worker.failed.connect(lambda msg: self._log_line(f"Render failed: {msg}"))
        self._worker.start()
        self._log_line("Render started…")

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output .mov", self._ed_outpath.text() or "out.mov", "QuickTime (*.mov)")
        if path:
            self._ed_outpath.setText(path)
            self._sync_anim_from_ui()

    def _reload_acquisition(self) -> None:
        self._sync_acq_from_ui()
        self._sync_overlay_from_ui()
        self._populate_scene()
        self._refresh_panels()

    # -- Helpers ------------------------------------------------------------

    def _replace_config(self, cfg: SceneConfig, path: str | None) -> None:
        self._config = cfg
        self._config_path = path
        title_suffix = f" — {os.path.basename(path)}" if path else ""
        self.setWindowTitle(f"synapto_anim{title_suffix}")
        self._populate_scene()
        self._refresh_panels()

    def _log_line(self, msg: str) -> None:
        self._log.appendPlainText(msg)

    def _update_intro_total_label(self) -> None:
        """Compute and display the total appear duration from current timings
        and the cached landing info. Mirrors the renderer's schedule including
        the shift-to-zero normalization that handles negative dend_gap."""
        if not hasattr(self, "_lbl_intro_total"):
            return
        an = self._config.animation
        per_syn_gap = an.intro_per_syn_gap_s
        dend_gap = an.intro_dend_gap_s
        per_syn_fade = an.intro_per_syn_fade_s

        # Build a flat list of (dmd, dend_id) in the same matrix-row order
        # that the renderer uses, then run the same cursor walk.
        flat_keys: list[tuple[int, str]] = []
        for key in DMD_KEYS:
            info = self._landing_info_cache.get(key)
            if info is None or not info.syn_ids:
                continue
            dmd_int = int(key.split("_")[1])
            for d in info.dend_ids:
                flat_keys.append((dmd_int, d))

        n_total = len(flat_keys)
        if n_total == 0:
            self._lbl_intro_total.setText("(auto: no synapses)")
            return

        starts = []
        cursor = 0.0
        prev_key = None
        n_dend_boundaries = 0
        for k in flat_keys:
            if prev_key is not None and k != prev_key:
                cursor += dend_gap
                n_dend_boundaries += 1
            starts.append(cursor)
            cursor += per_syn_gap
            prev_key = k
        mn = min(starts); mx = max(starts)
        total = (mx - mn) + per_syn_fade
        self._lbl_intro_total.setText(
            f"≈ {total:.2f} s ({n_total} syn, {n_dend_boundaries} dend gaps)"
        )

    # -- References --------------------------------------------------------

    def _create_ref_item(self, index: int, ref: ReferenceImage) -> ReferenceImageItem | None:
        if not os.path.exists(ref.path):
            self._log_line(f"Reference not found: {ref.path}")
            return None
        pix = QPixmap(ref.path)
        if pix.isNull():
            self._log_line(f"Failed to load reference: {ref.path}")
            return None
        item = ReferenceImageItem(index, pix, ref)
        item.proxy.moved.connect(self._on_ref_moved)
        return item

    def _on_add_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add reference image", "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        ref = ReferenceImage(path=path, x=50.0, y=50.0, scale=1.0, opacity=0.5)
        self._config.references.append(ref)
        idx = len(self._config.references) - 1
        item = self._create_ref_item(idx, ref)
        if item is not None:
            self._ref_items.append(item)
            self._scene.addItem(item)
        self._refresh_ref_list()
        self._ref_list.setCurrentRow(idx)

    def _on_remove_reference(self) -> None:
        row = self._ref_list.currentRow()
        if row < 0 or row >= len(self._config.references):
            return
        # Remove scene item and config entry
        item = self._ref_items.pop(row)
        self._scene.removeItem(item)
        del self._config.references[row]
        # Re-index remaining items
        for i, it in enumerate(self._ref_items):
            it._index = i
        self._refresh_ref_list()
        if self._config.references:
            self._ref_list.setCurrentRow(min(row, len(self._config.references) - 1))
        else:
            self._current_ref = -1
            self._refresh_ref_controls()

    def _refresh_ref_list(self) -> None:
        self._ref_list.blockSignals(True)
        self._ref_list.clear()
        for ref in self._config.references:
            self._ref_list.addItem(os.path.basename(ref.path))
        self._ref_list.blockSignals(False)

    def _on_ref_selection_changed(self, row: int) -> None:
        self._current_ref = row if 0 <= row < len(self._config.references) else -1
        self._refresh_ref_controls()

    def _refresh_ref_controls(self) -> None:
        i = self._current_ref
        enabled = 0 <= i < len(self._config.references)
        for w in (self._ref_x, self._ref_y, self._ref_scale, self._ref_opacity,
                  self._ref_visible, self._ref_include, self._ref_z):
            w.setEnabled(enabled)
        if not enabled:
            return
        ref = self._config.references[i]
        for w, v in (
            (self._ref_x, ref.x), (self._ref_y, ref.y),
            (self._ref_scale, ref.scale), (self._ref_opacity, ref.opacity),
        ):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self._ref_visible.blockSignals(True); self._ref_visible.setChecked(ref.visible); self._ref_visible.blockSignals(False)
        self._ref_include.blockSignals(True); self._ref_include.setChecked(ref.include_in_render); self._ref_include.blockSignals(False)
        self._ref_z.blockSignals(True)
        idx = self._ref_z.findText(ref.z_order)
        if idx >= 0:
            self._ref_z.setCurrentIndex(idx)
        self._ref_z.blockSignals(False)

    def _sync_reference_from_ui(self) -> None:
        i = self._current_ref
        if not (0 <= i < len(self._config.references)):
            return
        ref = self._config.references[i]
        ref.x = float(self._ref_x.value()); ref.y = float(self._ref_y.value())
        ref.scale = float(self._ref_scale.value())
        ref.opacity = float(self._ref_opacity.value())
        ref.visible = self._ref_visible.isChecked()
        ref.include_in_render = self._ref_include.isChecked()
        ref.z_order = self._ref_z.currentText()
        if i < len(self._ref_items):
            item = self._ref_items[i]
            item._silent = True
            item.setPos(ref.x, ref.y)
            item.setScale(ref.scale)
            item.setOpacity(ref.opacity if ref.visible else 0.0)
            item.setZValue(-5 if ref.z_order == "below" else 5)
            item._silent = False

    def _on_ref_moved(self, index: int, x: float, y: float) -> None:
        if not (0 <= index < len(self._config.references)):
            return
        ref = self._config.references[index]
        ref.x = x; ref.y = y
        if index == self._current_ref:
            self._ref_x.blockSignals(True); self._ref_x.setValue(x); self._ref_x.blockSignals(False)
            self._ref_y.blockSignals(True); self._ref_y.setValue(y); self._ref_y.blockSignals(False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _resolve_cmap_name(cmap) -> str | None:
    """Accept a matplotlib Colormap or a name string; register the Colormap
    with matplotlib (under its ``.name``) so subsequent ``plt.get_cmap(name)``
    calls resolve to it. Returns the resolved name, or None if ``cmap`` is None."""
    if cmap is None:
        return None
    if isinstance(cmap, str):
        return cmap
    from matplotlib import colormaps as _mpl_cmaps
    from matplotlib.colors import Colormap
    if isinstance(cmap, Colormap):
        name = cmap.name or "user_cmap"
        try:
            _mpl_cmaps.register(cmap, name=name, force=True)
        except TypeError:
            # Older matplotlib without `force`: unregister + register.
            try:
                _mpl_cmaps.unregister(name)
            except Exception:
                pass
            _mpl_cmaps.register(cmap, name=name)
        return name
    raise TypeError(f"cmap must be a name string or matplotlib Colormap; got {type(cmap)}")


def run_editor(config_path: str | None = None, *, cmap=None) -> None:
    """Launch the editor as a standalone application."""
    cmap_name = _resolve_cmap_name(cmap)

    app = QApplication.instance() or QApplication(sys.argv)
    if config_path:
        cfg = load_config(config_path)
    else:
        # Start with a blank/default scene. The user can use File → New to bind
        # a real acquisition, or File → Open an existing config.
        dlg = NewSceneDialog()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = dlg.make_config()
    if cmap_name is not None:
        cfg.overlay_kwargs.cmap = cmap_name
    w = EditorWindow(cfg, config_path=config_path)
    w.show()
    app.exec()
