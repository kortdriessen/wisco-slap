"""Configuration dataclasses for the sleep scoring system."""

from __future__ import annotations

from dataclasses import dataclass, field


STATE_NAMES: list[str] = ["NREM", "REM", "Wake"]
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
    transition_log_adjustments : additive log-adjustments for named state transitions.
    allow_self_transitions : whether adjacent explicit-duration segments may share a state.
    """

    mean_dur_s: dict[str, float] = field(
        default_factory=lambda: {
            "NREM": 35.0,
            "REM": 55.0,
            "Wake": 30.0,
        }
    )
    max_dur_s: float = 450
    lognorm_sigma: float = 1.05
    change_penalty: float = 0.5
    stay_bonus: float = -0.25
    transition_log_adjustments: dict[tuple[str, str], float] = field(
        default_factory=lambda: {
            ("Wake", "REM"): -10.0,
            ("REM", "NREM"): -10.0,
        }
    )
    allow_self_transitions: bool = False


@dataclass
class ClassifierConfig:
    """Configuration for the LightGBM classifier.

    Parameters
    ----------
    lgbm_params : dict of parameters passed to LGBMClassifier.
    calibration_method : calibration method for CalibratedClassifierCV.
    calibration_cv : number of cross-validation folds for calibration.
    fit_n_jobs : number of threads used by each LightGBM fit.
    calibration_outer_n_jobs : number of outer workers used by the calibrator.
    random_state : random seed for reproducibility.
    """

    lgbm_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.10,
            "num_leaves": 15,
            "min_child_samples": 50,
            "reg_lambda": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
    )
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5
    fit_n_jobs: int = 4
    calibration_outer_n_jobs: int = 1
    random_state: int = 42


@dataclass
class VideoQualityConfig:
    """Configuration for masking unreliable video-derived measurements."""

    pupil_valid_likelihood_min: float = 0.75
    eyelid_valid_likelihood_min: float = 0.75
    camera_off_likelihood_max: float = 0.20
    camera_off_whisk_quantile: float = 0.05
    camera_off_min_frame_frac: float = 0.80


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

VIDEO_QUALITY_FEATURES: list[str] = [
    "pu_valid_frac",
    "eyelid_valid_frac",
    "whisk_valid_frac",
    "camera_off_frac",
    "camera_off_epoch",
]


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
    video_quality : masking configuration for unreliable video-derived measurements.
    min_overlap_frac : minimum fraction of epoch that must be covered by a label interval.
    """

    epoch_len_s: float = 1.0
    states: list[str] = field(default_factory=lambda: list(STATE_NAMES))
    feature_windows_s: list[float] = field(default_factory=lambda: [1.0, 2.0, 4.0])
    hsmm: HSMMConfig = field(default_factory=HSMMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    video_quality: VideoQualityConfig = field(default_factory=VideoQualityConfig)
    min_overlap_frac: float = 0.0
