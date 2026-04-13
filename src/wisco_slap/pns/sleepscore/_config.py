"""Configuration dataclasses for the sleep scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field


STATE_NAMES: list[str] = ["NREM", "IS", "REM", "Wake"]
STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(STATE_NAMES)}
N_STATES: int = len(STATE_NAMES)


@dataclass
class HSMMConfig:
    """Configuration for the Hidden Semi-Markov Model post-processing.

    Parameters
    ----------
    mean_dur_s : dict mapping state name to expected mean bout duration in seconds.
    max_dur_s : maximum bout duration (truncation point for duration distribution).
    lognorm_sigma : shape parameter for the truncated log-normal duration distribution.
    change_penalty : additive log-penalty for state transitions (more negative = stickier).
    stay_bonus : additive log-bonus for self-transitions (positive = stickier).
    """

    mean_dur_s: dict[str, float] = field(
        default_factory=lambda: {
            "NREM": 30.0,
            "IS": 15.0,
            "REM": 60.0,
            "Wake": 30.0,
        }
    )
    max_dur_s: float = 300.0
    lognorm_sigma: float = 0.75
    change_penalty: float = -0.8
    stay_bonus: float = 0.2


@dataclass
class ClassifierConfig:
    """Configuration for the LightGBM classifier.

    Parameters
    ----------
    lgbm_params : dict of parameters passed to LGBMClassifier.
    calibration_method : calibration method for CalibratedClassifierCV.
    calibration_cv : number of cross-validation folds for calibration.
    test_size : fraction of labeled data held out for calibration.
    random_state : random seed for reproducibility.
    """

    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "max_depth": -1,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
    )
    calibration_method: str = "isotonic"
    calibration_cv: int = 5
    test_size: float = 0.15
    random_state: int = 42


# EEG frequency band definitions used for feature extraction
EEG_BANDS_1S: list[tuple[str, float, float]] = [
    ("delta", 0.5, 4.0),
    ("delta1", 0.5, 2.0),
    ("delta2", 2.0, 4.0),
    ("theta", 6.0, 10.0),
    ("sigma", 10.0, 15.0),
    ("spindle", 11.0, 16.0),
    ("beta", 15.0, 30.0),
    ("lgamma", 30.0, 55.0),
    ("ugamma", 52.0, 70.0),
    ("hgamma", 55.0, 90.0),
]

# Bands for longer windows (2s, 4s) — only the ones that benefit from better resolution
EEG_BANDS_LONG: list[tuple[str, float, float]] = [
    ("delta", 0.5, 4.0),
    ("theta", 6.0, 10.0),
    ("sigma", 10.0, 15.0),
    ("spindle", 11.0, 16.0),
    ("lgamma", 30.0, 55.0),
    ("ugamma", 52.0, 70.0),
]

# Key features used for computing rolling context features
CONTEXT_KEY_FEATURES: list[str] = [
    "eeg_delta_1s",
    "eeg_theta_delta_1s",
    "eeg_theta_sigma_1s",
    "eeg_ugamma_1s",
    "pu_diam_mean",
    "pu_vel_absmean",
    "pu_eyelid_mean",
    "whisk_mean",
    "whisk_active_frac",
]

# Context window sizes (in epochs, which are 1 second each)
CONTEXT_WINDOWS: list[int] = [5, 15, 60]


@dataclass
class ScoreConfig:
    """Top-level configuration for the sleep scoring system.

    Parameters
    ----------
    epoch_len_s : length of each scoring epoch in seconds.
    states : list of sleep state names in canonical order.
    feature_windows_s : EEG analysis window durations for multi-resolution features.
    hsmm : HSMM post-processing configuration.
    classifier : LightGBM classifier configuration.
    min_overlap_frac : minimum fraction of epoch that must be covered by a label interval.
    """

    epoch_len_s: float = 1.0
    states: list[str] = field(default_factory=lambda: list(STATE_NAMES))
    feature_windows_s: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    hsmm: HSMMConfig = field(default_factory=HSMMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    min_overlap_frac: float = 0.0
