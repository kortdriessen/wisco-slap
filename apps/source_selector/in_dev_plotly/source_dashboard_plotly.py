import os

import numpy as np
import plotly.graph_objects as go
import polars as pl
import slap2_py as spy
import streamlit as st
from PIL import Image
from streamlit_plotly_events import plotly_events

import wisco_slap as wis

st.title("Source Selector")

if "selector_url" not in st.session_state:
    st.session_state["selector_url"] = None
if "selector_proc" not in st.session_state:
    st.session_state["selector_proc"] = None

exp_info = wis.pipes.exp_info.load_exp_info_spreadsheet()
subjects = exp_info["subject"].unique()
subject = st.selectbox("Subject", subjects)
subject_exps = exp_info.filter(pl.col("subject") == subject)["experiment"].unique()
exp = st.selectbox("Experiment", subject_exps)
acq_ids = wis.util.info.get_unique_acquisitions_per_experiment(subject, exp)
acq_id = st.selectbox("Acquisition-ID", acq_ids)
loc, acq = acq_id.split("--")

dmd = st.radio("DMD", [1, 2])


esum_path = wis.util.info.sub_esum_path(subject, exp, loc, acq)
fp = spy.hf.load_any(esum_path, f"/exptSummary/E[{dmd - 1}][0]['footprints']")
fp = fp.swapaxes(1, 2)
masks = fp > 0
meanim = spy.hf.load_any(esum_path, f"/exptSummary/meanIM[{dmd - 1}][0]")
image = meanim[1, :, :].T

st.write(image.shape)
st.write(masks.shape)

# code for source selection should go here!
# --------------------------- Plotly lasso source selector ---------------------------


st.subheader("Lasso-select sources (centroid-based)")

# --- Sidebar controls specific to selection/output ---
st.sidebar.write("---")
out_path = st.sidebar.text_input(
    "Output text file for selected IDs",
    value="selected_source_ids.txt",
    help="File will be overwritten each time you click 'Use Current Selection'.",
)


def normalize_image_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Map (H, W) float with NaN/inf to grayscale RGB uint8.
    Non-finite -> white. Handles constant images.
    """
    if img.ndim != 2:
        raise ValueError("image must be (H, W)")
    H, W = img.shape
    rgb = np.empty((H, W, 3), dtype=np.uint8)

    finite = np.isfinite(img)
    if not finite.any():
        rgb[:] = 255
        return rgb

    vals = img[finite]
    vmin, vmax = float(vals.min()), float(vals.max())

    gray = np.zeros((H, W), dtype=np.uint8)
    if vmax > vmin:
        norm = (img[finite] - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        gray[finite] = (norm * 255.0).astype(np.uint8)
    else:
        gray[finite] = 0  # constant image; pick 0 or 127

    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    rgb[~finite] = 255
    return rgb


# ===================== Diagnostics =====================
H, W = image.shape
finite_mask = np.isfinite(image)
n_finite = int(finite_mask.sum())
n_nonfinite = int(image.size - n_finite)
n_nonempty_masks = int(sum(np.any(m) for m in masks))

st.caption(
    f"Image shape: (H={H}, W={W}) • finite pixels: {n_finite:,} • non-finite: {n_nonfinite:,} • "
    f"non-empty masks: {n_nonempty_masks:,}"
)

# Bail out early with helpful messages
if n_finite == 0:
    st.error("The loaded image has no finite pixels. All values are NaN/±inf.")
    st.stop()
if n_nonempty_masks == 0:
    st.warning("All masks are empty (no True pixels); nothing to plot.")
    # We’ll still show the image so you can verify loading.

# ===================== Prepare RGB and centroids =====================
image_rgb = normalize_image_to_rgb(image)  # (H, W, 3) uint8
pil_img = Image.fromarray(image_rgb, mode="RGB")  # PIL image for robust Plotly display


# ===================== Build figure via layout-image (more reliable) =====================
marker_size = st.sidebar.slider("Centroid marker size", 4, 20, 10, 1)
out_path = st.sidebar.text_input(
    "Output text file for selected IDs",
    value="selected_source_ids.txt",
    help="File is overwritten each time you click 'Use Current Selection'.",
)

fig = go.Figure()

# Add background image in data coords (x from 0..W, y from 0..H)
fig.update_layout(
    images=[
        dict(
            source=pil_img,
            xref="x",
            yref="y",
            x=0,
            y=0,  # We'll reverse y-axis below; keep y origin at 0 here
            sizex=W,
            sizey=H,
            sizing="stretch",
            layer="below",
        )
    ],
    xaxis=dict(range=[0, W], visible=False, showgrid=False, zeroline=False),
    yaxis=dict(
        range=[H, 0],
        visible=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        autorange=False,
    ),
    dragmode="lasso",
    clickmode="event+select",
    margin=dict(l=10, r=10, t=30, b=10),
    height=min(900, int(H * 0.9) + 100),
    title="Draw a lasso around centroids to select sources",
    legend=dict(orientation="h"),
)


# -------------------- Centroid overlay + events (robust) --------------------
# Validate/compute centroids
def compute_centroids(masks_np: np.ndarray) -> np.ndarray:
    N, H0, W0 = masks_np.shape
    cents = np.full((N, 2), np.nan, dtype=float)
    for i in range(N):
        yy, xx = np.where(masks_np[i])
        if yy.size:
            cents[i, 0] = float(xx.mean())  # x
            cents[i, 1] = float(yy.mean())  # y
    return cents


centroids = compute_centroids(masks)
N = masks.shape[0]
ids_array = np.arange(N, dtype=int)

# Filter NaNs
nan_mask = np.isnan(centroids).any(axis=1)
n_nan = int(nan_mask.sum())

# Clamp to the plotting extent to avoid out-of-range values
H, W = image.shape
clamped = centroids.copy()
clamped[:, 0] = np.clip(clamped[:, 0], 0, W - 1)
clamped[:, 1] = np.clip(clamped[:, 1], 0, H - 1)
# Points that moved due to clamping (were outside) — just for diagnostics
moved = np.any(np.abs(clamped - centroids) > 1e-9, axis=1)
n_oob = int((~nan_mask & moved).sum())

valid_mask = ~nan_mask
xv = clamped[valid_mask, 0]
yv = clamped[valid_mask, 1]
valid_ids = ids_array[valid_mask]

st.caption(
    f"Centroids: total {N} • valid {valid_ids.size} • NaN {n_nan} • out-of-bounds (clamped) {n_oob}"
)

# Build/augment the existing figure that already has the layout image
# (Assume 'fig' exists from your image section)
fig.add_trace(
    go.Scatter(
        x=xv,
        y=yv,
        mode="markers",
        marker=dict(
            size=marker_size, color="lime", line=dict(width=1.5, color="white")
        ),
        name="sources",
        text=[f"ID: {int(i)}" for i in valid_ids],
        customdata=valid_ids,
        hovertemplate="ID=%{customdata}<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>",
        selected=dict(marker=dict(size=marker_size + 4, color="red")),
        unselected=dict(marker=dict(opacity=0.7)),
    )
)
scatter_trace_index = len(fig.data) - 1  # index of the scatter we just added

st.caption("Tip: hold Shift to add to the selection; double-click to clear.")

events = plotly_events(
    fig,
    click_event=False,
    select_event=True,
    hover_event=False,
    override_height=fig.layout.height or 800,
    key="source_selector_plot_layoutimg",
)

# Map Plotly events → source IDs using the actual trace index
selected_ids = []
if events:
    for e in events:
        if e.get("curveNumber", -1) == scatter_trace_index and "pointNumber" in e:
            idx = int(e["pointNumber"])
            if 0 <= idx < valid_ids.size:
                selected_ids.append(int(valid_ids[idx]))

# Controls + persistence (keep your existing block if you already have it)
c1, c2, c3 = st.columns([1, 1, 2])
if "selected_ids" not in st.session_state:
    st.session_state["selected_ids"] = []

with c1:
    if st.button("Use Current Selection"):
        st.session_state["selected_ids"] = sorted(set(selected_ids))
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(map(str, st.session_state["selected_ids"])))
            st.success(
                f"Wrote {len(st.session_state['selected_ids'])} IDs → {out_path}"
            )
        except Exception as e:
            st.error(f"Failed to write file: {e}")

with c2:
    if st.button("Clear Selection"):
        st.session_state["selected_ids"] = []
        try:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")
            st.info(f"Cleared selection (empty file) → {out_path}")
        except Exception as e:
            st.error(f"Failed to write file: {e}")

with c3:
    st.write(f"**Output file:** `{out_path or '(set a path in the sidebar)'}`")

st.write("### Selected Source IDs")
if st.session_state["selected_ids"]:
    st.code("\n".join(map(str, st.session_state["selected_ids"])), language="text")
else:
    st.info(
        "No sources selected yet. Draw a lasso and click **Use Current Selection**."
    )
