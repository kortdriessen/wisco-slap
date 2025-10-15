import math
import os
from pathlib import Path

import numpy as np
import polars as pl

import wisco_slap.defs as DEFS


def record_camera_sync_mismatch(dr, n_frames, n_pulses_detected):
    filename = "pupil_sync_mismatch.txt"
    path = os.path.join(dr, filename)
    if os.path.exists(path):
        os.system(f"rm -rf {path}")
    with open(path, "a") as f:
        f.write(f"n_frames: {n_frames}, n_pulses_detected: {n_pulses_detected}\n")
        f.close()
    return


def clean_raw_pupil_df(df):
    # Reshape DLC CSV to tidy format: columns -> frame, bodypart, x, y, likelihood
    other_cols = [c for c in df.columns if c != "scorer"]

    # Try to read header rows that DLC writes into the CSV data rows
    bp_row_df = df.filter(
        pl.col("scorer").cast(pl.Utf8).str.replace_all('"', "") == "bodyparts"
    ).select(pl.all().exclude("scorer"))
    crd_row_df = df.filter(
        pl.col("scorer").cast(pl.Utf8).str.replace_all('"', "") == "coords"
    ).select(pl.all().exclude("scorer"))

    if (
        bp_row_df.height == 1
        and crd_row_df.height == 1
        and bp_row_df.width == len(other_cols) == crd_row_df.width
    ):
        bodyparts_vals = bp_row_df.row(0)
        coords_vals = crd_row_df.row(0)
    else:
        # Fallback if header rows aren't present in the data area
        n = len(other_cols)
        coords_vals = (["x", "y", "likelihood"] * ((n + 2) // 3))[:n]
        bodyparts_vals = [f"bp{i // 3}" for i in range(n)]

    # Build a rename map like: original_col -> "<bodypart>__<coord>"
    rename_map = {
        col: f"{bp}__{coord}"
        for col, bp, coord in zip(other_cols, bodyparts_vals, coords_vals, strict=False)
    }

    # Keep only numeric frame rows, cast to int, drop the DLC header rows
    _df_numeric = (
        df.filter(
            pl.col("scorer")
            .cast(pl.Utf8)
            .str.replace_all('"', "")
            .str.contains(r"^\d+$")
        )
        .with_columns(pl.col("scorer").cast(pl.Int64).alias("frame"))
        .drop("scorer")
        .rename(rename_map)
    )

    # Wide -> long, then pivot coords to columns
    _df_long = _df_numeric.melt(
        id_vars=["frame"], variable_name="bodypart_coord", value_name="value"
    ).with_columns(
        [
            pl.col("value").cast(pl.Float64),
            pl.col("bodypart_coord")
            .str.split_exact("__", 1)
            .struct.field("field_0")
            .alias("bodypart"),
            pl.col("bodypart_coord")
            .str.split_exact("__", 1)
            .struct.field("field_1")
            .alias("coord"),
        ]
    )

    df_tidy = (
        _df_long.select(["frame", "bodypart", "coord", "value"])
        .pivot(values="value", index=["frame", "bodypart"], columns="coord")
        .select(["frame", "bodypart", "x", "y", "likelihood"])  # enforce column order
    )
    return df_tidy


def load_cleaned_dlc_pupil_df(subject, exp, sb):
    pupil_dir = f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference"
    dlc_tag = "DLC_Resnet50_dlc_slap_pupilSep23shuffle0_snapshot_best-60"
    pupil_name = f"pupil-{sb}"
    csv_name = f"{pupil_name}{dlc_tag}.csv"
    csv_path = Path(f"{pupil_dir}/{csv_name}")
    if csv_path.exists():
        df_raw = pl.read_csv(csv_path)
        df_clean = clean_raw_pupil_df(df_raw)
        return df_clean
    else:
        print(f"File {csv_path} does not exist")
        return None


def _weighted_kasa_circle_fit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float, float]:
    """
    Solve x^2 + y^2 + A x + B y + C = 0 via weighted least squares.
    Returns (cx, cy, r).  x,y,w are 1D numpy arrays of same length.
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)

    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 1.0
    w = np.clip(w, 1e-6, None)

    Aw = A * w[:, None]
    lhs = A.T @ Aw
    rhs = A.T @ (w * b)
    try:
        p = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        p, *_ = np.linalg.lstsq(Aw, w * b, rcond=None)

    Acoef, Bcoef, Ccoef = p
    cx = -Acoef / 2.0
    cy = -Bcoef / 2.0
    r2 = cx * cx + cy * cy - Ccoef
    r = math.sqrt(max(r2, 0.0))
    return float(cx), float(cy), float(r)


def compute_pupil_metrics(
    df: pl.DataFrame,
    *,
    likelihood_threshold: float = 0.9,
    label_prefix: str = "pup_",
    center_label: str = "pup_center",
    min_edge_points: int = 4,
) -> pl.DataFrame:
    """
    Parameters
    ----------
    df : Polars DataFrame with columns ['frame','bodypart','x','y','likelihood'] (DLC long format)
    likelihood_threshold : keep only frames whose *mean* likelihood across pupil labels >= threshold
    label_prefix : prefix for pupil labels
    center_label : name of center label (excluded from rim fit)
    min_edge_points : minimum number of rim points required to fit circle

    Returns
    -------
    Polars DataFrame: ['frame','diameter','motion'].
    - diameter: 2*radius from a weighted circle fit on rim points
    - motion: Euclidean displacement of fitted center between consecutive kept frames (pixels/frame)
    """
    # 1) restrict to pupil labels; mark rim points (exclude center)
    dfp = df.filter(pl.col("bodypart").str.starts_with(label_prefix)).with_columns(
        is_edge=(pl.col("bodypart") != center_label)
    )
    if dfp.is_empty():
        return pl.DataFrame({"frame": [], "diameter": [], "motion": []})

    # 2) per-frame lists via .implode() (version-stable) + mean likelihood
    per_frame = (
        dfp.group_by("frame", maintain_order=True)
        .agg(
            xs=pl.col("x").filter(pl.col("is_edge")).implode(),
            ys=pl.col("y").filter(pl.col("is_edge")).implode(),
            ws=pl.col("likelihood").filter(pl.col("is_edge")).implode(),
            mean_lh=pl.col("likelihood").mean(),
        )
        .sort("frame")
        .with_columns(n_edge=pl.col("xs").list.len())
        .filter(pl.col("n_edge") >= min_edge_points)  # guarantee enough points
    )

    if per_frame.is_empty():
        return pl.DataFrame({"frame": [], "diameter": [], "motion": []})

    # 3) weighted circle fit per frame (explicit return_dtype for Polars 1.33.1)
    fit_dtype = pl.Struct(
        {
            "cx": pl.Float64,
            "cy": pl.Float64,
            "diameter": pl.Float64,
        }
    )

    def _fit_from_lists(s: dict) -> dict:
        xs = np.asarray(s["xs"], dtype=float)
        ys = np.asarray(s["ys"], dtype=float)
        ws = np.asarray(s["ws"], dtype=float)
        cx, cy, r = _weighted_kasa_circle_fit(xs, ys, ws)
        return {"cx": cx, "cy": cy, "diameter": 2.0 * r}

    fitted = (
        per_frame.with_columns(
            fit=pl.struct(["xs", "ys", "ws"]).map_elements(
                _fit_from_lists, return_dtype=fit_dtype
            )
        )
        .with_columns(
            cx=pl.col("fit").struct.field("cx"),
            cy=pl.col("fit").struct.field("cy"),
            diameter=pl.col("fit").struct.field("diameter"),
        )
        .drop(["fit", "xs", "ys", "ws", "n_edge"])
        .filter(pl.col("mean_lh") >= likelihood_threshold)
        .sort("frame")
    )

    if fitted.is_empty():
        return pl.DataFrame({"frame": [], "diameter": [], "motion": []})

    # 4) motion = displacement of fitted center between consecutive (kept) frames
    out = (
        fitted.with_columns(
            dx=pl.col("cx").diff().fill_null(0.0),
            dy=pl.col("cy").diff().fill_null(0.0),
        )
        .with_columns(
            motion=(pl.col("dx") * pl.col("dx") + pl.col("dy") * pl.col("dy")).sqrt()
        )
        .select(["frame", "diameter", "motion"])
    )
    return out


def compute_eyelid_openness(
    df: pl.DataFrame,
    *,
    likelihood_threshold: float = 0.9,
    upper_medial: str = "lid_upper_medial",
    upper_mid: str = "lid_upper_midpoint",
    upper_lateral: str = "lid_upper_lateral",
    lower_medial: str = "lid_lower_medial",
    lower_mid: str = "lid_lower_midpoint",
    lower_lateral: str = "lid_lower_lateral",
    medial_canthus: str = "tear_duct",
    lateral_canthus: str = "lid_lateral_edge",
) -> pl.DataFrame:
    """
    df: Polars DF with long/‘tidy’ DLC columns: frame, bodypart, x, y, likelihood
    Returns: Polars DF with columns ['frame', 'eyelid_open', 'eyelid_open_norm']
      - eyelid_open: pixels (likelihood-weighted fissure height)
      - eyelid_open_norm: unitless (height / canthus-to-canthus width)
    Keeps only frames whose mean likelihood across the used labels >= likelihood_threshold.
    """
    labels = [
        upper_medial,
        upper_mid,
        upper_lateral,
        lower_medial,
        lower_mid,
        lower_lateral,
        medial_canthus,
        lateral_canthus,
    ]

    # --- 1) Pivot to wide per-frame table using first() aggregations ---
    def _agg_one(lbl: str):
        return [
            pl.col("x")
            .filter(pl.col("bodypart") == lbl)
            .first()
            .cast(pl.Float64)
            .alias(f"{lbl}_x"),
            pl.col("y")
            .filter(pl.col("bodypart") == lbl)
            .first()
            .cast(pl.Float64)
            .alias(f"{lbl}_y"),
            pl.col("likelihood")
            .filter(pl.col("bodypart") == lbl)
            .first()
            .cast(pl.Float64)
            .alias(f"{lbl}_lh"),
        ]

    pf = (
        df.filter(pl.col("bodypart").is_in(labels))
        .group_by("frame", maintain_order=True)
        .agg(*[e for lbl in labels for e in _agg_one(lbl)])
        .sort("frame")
    )

    if pf.is_empty():
        return pl.DataFrame({"frame": [], "eyelid_open": [], "eyelid_open_norm": []})

    # --- 2) Eye axis and rotation terms built in sequential stages ---
    # axis vector (medial -> lateral)
    pf = pf.with_columns(
        (pl.col(f"{lateral_canthus}_x") - pl.col(f"{medial_canthus}_x")).alias("vx"),
        (pl.col(f"{lateral_canthus}_y") - pl.col(f"{medial_canthus}_y")).alias("vy"),
    )
    # eye width (norm of the axis)
    pf = pf.with_columns(
        ((pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt()).alias("eye_width")
    ).filter(pl.col("eye_width").is_not_null() & (pl.col("eye_width") > 0))

    # rotation cos/sin
    pf = pf.with_columns(
        (pl.col("vx") / pl.col("eye_width")).alias("cos_th"),
        (pl.col("vy") / pl.col("eye_width")).alias("sin_th"),
    )

    # helper: rotated vertical coordinate y' for a label, about medial canthus origin
    def yrot(lbl: str) -> pl.Expr:
        dx = pl.col(f"{lbl}_x") - pl.col(f"{medial_canthus}_x")
        dy = pl.col(f"{lbl}_y") - pl.col(f"{medial_canthus}_y")
        return ((-pl.col("sin_th") * dx) + (pl.col("cos_th") * dy)).alias(f"{lbl}_yrot")

    pf = pf.with_columns(
        yrot(upper_medial),
        yrot(lower_medial),
        yrot(upper_mid),
        yrot(lower_mid),
        yrot(upper_lateral),
        yrot(lower_lateral),
    )

    # --- 3) Distances at three columns + likelihood weights ---
    pf = pf.with_columns(
        (pl.col(f"{upper_medial}_yrot") - pl.col(f"{lower_medial}_yrot"))
        .abs()
        .alias("d_med"),
        (pl.col(f"{upper_mid}_yrot") - pl.col(f"{lower_mid}_yrot"))
        .abs()
        .alias("d_mid"),
        (pl.col(f"{upper_lateral}_yrot") - pl.col(f"{lower_lateral}_yrot"))
        .abs()
        .alias("d_lat"),
        ((pl.col(f"{upper_medial}_lh") + pl.col(f"{lower_medial}_lh")) / 2).alias(
            "w_med"
        ),
        ((pl.col(f"{upper_mid}_lh") + pl.col(f"{lower_mid}_lh")) / 2).alias("w_mid"),
        ((pl.col(f"{upper_lateral}_lh") + pl.col(f"{lower_lateral}_lh")) / 2).alias(
            "w_lat"
        ),
    )

    # effective weights: ignore a column if its distance is null
    pf = pf.with_columns(
        (
            pl.coalesce([pl.col("w_med"), pl.lit(0.0)])
            * pl.col("d_med").is_not_null().cast(pl.Float64)
        ).alias("w_med_eff"),
        (
            pl.coalesce([pl.col("w_mid"), pl.lit(0.0)])
            * pl.col("d_mid").is_not_null().cast(pl.Float64)
        ).alias("w_mid_eff"),
        (
            pl.coalesce([pl.col("w_lat"), pl.lit(0.0)])
            * pl.col("d_lat").is_not_null().cast(pl.Float64)
        ).alias("w_lat_eff"),
    )

    pf = pf.with_columns(
        (
            pl.coalesce([pl.col("d_med"), pl.lit(0.0)]) * pl.col("w_med_eff")
            + pl.coalesce([pl.col("d_mid"), pl.lit(0.0)]) * pl.col("w_mid_eff")
            + pl.coalesce([pl.col("d_lat"), pl.lit(0.0)]) * pl.col("w_lat_eff")
        ).alias("num"),
        (pl.col("w_med_eff") + pl.col("w_mid_eff") + pl.col("w_lat_eff")).alias("den"),
    ).with_columns(
        pl.when(pl.col("den") > 0)
        .then(pl.col("num") / pl.col("den"))
        .otherwise(None)
        .alias("eyelid_open"),
    )

    pf = pf.with_columns(
        pl.when(pl.col("eye_width") > 0)
        .then(pl.col("eyelid_open") / pl.col("eye_width"))
        .otherwise(None)
        .alias("eyelid_open_norm")
    )

    # --- 4) Likelihood gating across all used labels ---
    lh_cols = [f"{lbl}_lh" for lbl in labels]
    pf = (
        pf.with_columns(
            pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in lh_cols]).alias(
                "lh_sum"
            ),
            pl.sum_horizontal(
                [pl.col(c).is_not_null().cast(pl.Float64) for c in lh_cols]
            ).alias("lh_cnt"),
        )
        .with_columns(
            pl.when(pl.col("lh_cnt") > 0)
            .then(pl.col("lh_sum") / pl.col("lh_cnt"))
            .otherwise(None)
            .alias("lh_mean")
        )
        .filter(pl.col("lh_mean") >= likelihood_threshold)
    )

    return pf.select(["frame", "eyelid_open", "eyelid_open_norm"]).sort("frame")


def compute_eye_metric_df(subject, exp, sb):
    df = load_cleaned_dlc_pupil_df(subject, exp, sb)
    lid_df = df.filter(
        pl.col("bodypart").str.contains("lid") | pl.col("bodypart").str.contains("tear")
    )
    pup_df = df.filter(pl.col("bodypart").str.contains("pup"))
    pup = compute_pupil_metrics(pup_df, likelihood_threshold=0.0)
    lid = compute_eyelid_openness(lid_df, likelihood_threshold=0.0)
    eye = pup.with_columns(lid=lid["eyelid_open"]).with_columns(
        lid_norm=lid["eyelid_open_norm"]
    )
    eye = eye.with_columns(
        pup_likelihood=pup_df.group_by("frame")
        .agg(pl.col("likelihood").mean())
        .sort("frame")["likelihood"]
    )
    eye = eye.with_columns(
        lid_likelihood=lid_df.group_by("frame")
        .agg(pl.col("likelihood").mean())
        .sort("frame")["likelihood"]
    )
    return eye


def load_eye_metric_df(subject, exp, sb):
    eyedir = f"{DEFS.anmat_root}/{subject}/{exp}/pupil_inference/eye_metrics"
    edf_path = f"{eyedir}/eye_metrics-{sb}.parquet"
    return pl.read_parquet(edf_path)


def load_whisking_df(subject, exp, sb):
    path = f"{DEFS.anmat_root}/{subject}/{exp}/whisking/whisk_df-{sb}.parquet"
    return pl.read_parquet(path)
