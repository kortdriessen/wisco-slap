"""
Automatic sleep scoring system.

4-state classification (NREM, IS, REM, Wake) using LightGBM with
multi-resolution EEG features, pupil/whisking signals, and HSMM
post-processing for temporal smoothing.

Public API
----------
- ``ScoreConfig`` : top-level configuration dataclass
- ``train_model`` : train a new model from sessions and labels
- ``score_session`` : score a single session with a trained model
- ``evaluate_loso`` : leave-one-subject-out cross-validation
- ``save_model`` / ``load_model`` : model persistence
- ``print_feature_importance`` : display feature importance from a trained model
- ``create_session`` : build a session dict from raw data identifiers
- ``load_labels_csv`` : load and validate a label CSV file
- ``autoscore_session`` / ``autoscore_all_subjects`` : batch scoring
"""

from ._config import ScoreConfig, HSMMConfig, ClassifierConfig, STATE_NAMES, STATE_TO_IDX
from ._data import create_session, load_labels_csv, validate_and_standardize_labels
from ._model import train_model, score_session, save_model, load_model, print_feature_importance
from ._evaluate import evaluate_loso
from .score_all import autoscore_session, autoscore_all_subjects

__all__ = [
    "ScoreConfig",
    "HSMMConfig",
    "ClassifierConfig",
    "STATE_NAMES",
    "STATE_TO_IDX",
    "train_model",
    "score_session",
    "evaluate_loso",
    "save_model",
    "load_model",
    "print_feature_importance",
    "create_session",
    "load_labels_csv",
    "validate_and_standardize_labels",
    "autoscore_session",
    "autoscore_all_subjects",
]
