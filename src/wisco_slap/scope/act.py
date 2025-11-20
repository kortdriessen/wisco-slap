# ====================================================================================
# MAIN FUNCTIONS FOR PROCESSING AND MANIPULATING ACTIVITY DATAFRAMES FROM io MODULE
# ====================================================================================

import polars as pl
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def get_glut_sums(
    syndf: pl.DataFrame,
    noise_multiple: float = 5,
    sum_on: list[str] = ["time", "soma-ID"],
) -> pl.DataFrame:
    """
    Get the glutamate sums from the syndf dataframe.

    Parameters
    ----------
    syndf : pl.DataFrame
        The syndf dataframe.
    noise_multiple : float, optional
        The multiple of the noise to use as the threshold, by default 5.
    sum_on : list[str], optional
        The columns to group by, by default ["time", "soma-ID"].

    Returns
    -------
    pl.DataFrame
        The glutamate sums dataframe.
    """
    glut_sums = syndf.group_by(sum_on).agg(pl.col("data").sum())
    glut_sums = glut_sums.sort(sum_on)
    syndf = syndf.with_columns((pl.col("noise") * noise_multiple).alias("threshold"))
    syndf = syndf.with_columns(
        pl.when(pl.col("data") > pl.col("threshold"))
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("active")
    )
    fractives = (
        syndf.group_by(sum_on)
        .agg(pl.col("active").sum() / pl.col("active").count())
        .sort(sum_on)
    )
    glut_sums = glut_sums.with_columns(fractives["active"].alias("fractive"))
    return glut_sums


def fsvd_to_dff_for_movie(
    fsvd: np.ndarray,
    sample_rate_hz: float,
    denoise_window_s: float = 0.5,  # median filter window (sec)
    baseline_window_s: float = 10.0,  # hull/windowed-min + mean window (sec)
    saturate_at: float = 1.0,  # dF/F that maps to 1.0 for display
    eps: float = 1e-6,
    return_f0: bool = False,
):
    """
    Convert a 1D Fsvd trace to dF/F and a [0,1] normalized version for comparable visualization.

    - Baseline F0 is estimated with a fast variant of computeF0 algo2:
      rolling median (denoise_window) -> rolling min (baseline_window) -> rolling mean (baseline_window).
    - Output 'dff_norm' uses a fixed saturation level (saturate_at) so different somas are directly comparable.

    Args:
        fsvd: 1D numpy array (time,)
        sample_rate_hz: sampling rate (Hz)
        denoise_window_s: seconds for median filter
        baseline_window_s: seconds for rolling min/mean
        saturate_at: dF/F value that maps to 1.0 (clip); use the same across somas for comparability
        eps: small clamp for F0 to avoid division blow-up
        return_f0: also return the baseline estimate

    Returns:
        dff:      (time,) deltaF/F
        dff_norm: (time,) in [0,1], with negative values clipped to 0 and saturated at 'saturate_at'
        (optional) f0: (time,) baseline
    """
    fsvd = np.asarray(fsvd).astype(float)
    if fsvd.ndim != 1:
        raise ValueError("fsvd must be a 1D array")

    # Helper: linear interpolate NaNs; leave leading/trailing NaNs as-is
    def _interp_nans_1d(x):
        t = np.arange(x.size)
        good = np.isfinite(x)
        if good.sum() == 0:
            return x.copy()
        x_filled = x.copy()
        # interior NaNs
        if good[0] and good[-1]:
            x_filled[~good] = np.interp(t[~good], t[good], x[good])
        else:
            # fill interior; keep leading/trailing NaNs
            first_good = np.argmax(good)
            last_good = len(good) - 1 - np.argmax(good[::-1])
            if last_good > first_good:
                sel = (t >= first_good) & (t <= last_good)
                bad_mid = sel & ~good
                x_filled[bad_mid] = np.interp(t[bad_mid], t[good], x[good])
        return x_filled

    # Windows in samples
    den_win = max(3, int(round(denoise_window_s * sample_rate_hz)))
    if den_win % 2 == 0:
        den_win += 1  # median filter requires odd
    base_win = max(3, int(round(baseline_window_s * sample_rate_hz)))

    # Fill NaNs for filtering; keep a mask to restore NaNs later
    nan_mask = ~np.isfinite(fsvd)
    x = _interp_nans_1d(fsvd)

    # Rolling median (denoise)
    x_med = medfilt(x, kernel_size=den_win)

    # Rolling minimum (convex-hull-like lower envelope) then smooth with rolling mean
    x_min = minimum_filter1d(x_med, size=base_win, mode="nearest")
    f0 = uniform_filter1d(x_min, size=base_win, mode="nearest")

    # Clamp F0 away from zero and negative values
    f0_clamp = np.maximum(f0, max(eps, np.nanmedian(f0) * eps))

    # dF/F
    dff = (fsvd - f0) / f0_clamp
    dff[nan_mask] = np.nan

    # Fixed-range normalization for cross-soma comparability
    dff_pos = np.clip(dff, 0, None)
    dff_norm = np.clip(dff_pos / float(saturate_at), 0.0, 1.0)

    if return_f0:
        return dff, dff_norm, f0
    return dff, dff_norm
