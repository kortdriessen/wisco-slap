"""Constants and default parameters for glutamate event detection."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# iGluSnFR4f sensor properties
# ---------------------------------------------------------------------------
IGLUSNFR_TAU_S: float = 0.026  # 26 ms decay constant (from iGluSnFR4f paper)
FS_HZ: float = 200.0  # standard SLAP2 sampling rate

# ---------------------------------------------------------------------------
# Event extraction defaults
# ---------------------------------------------------------------------------
DEFAULT_AMP_THRESH_SIGMA: float = (
    2.0  # threshold in noise-sigma units (nonneg trace is clean enough for 2σ)
)
DEFAULT_MIN_PEAK_DISTANCE_S: float = 0.010  # min distance between peaks (10 ms)

# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------
MAD_NORMAL_SCALE: float = 0.6744897501960817  # median(|N(0,1)|)

# ---------------------------------------------------------------------------
# Hierarchical empirical Bayes defaults
# ---------------------------------------------------------------------------
DEFAULT_EB_SECOND_PASS: bool = True
DEFAULT_EB_RATE_SHRINK_BETA: float = 0.6
DEFAULT_EB_ALPHA_SCALE_BOUNDS: tuple[float, float] = (0.5, 2.0)

# ---------------------------------------------------------------------------
# FISTA defaults
# ---------------------------------------------------------------------------
FISTA_LAM_MULT: float = 1.0
FISTA_MAX_ITER: int = 80
FISTA_TOL: float = 1e-4
FISTA_BASELINE_CUTOFF_HZ: float = 0.05

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR_NAME: str = "glut_events"
