"""
Automatic sleep scoring system.

3-state classification (NREM, REM, Wake) using LightGBM with
multi-resolution EEG features, pupil/whisking signals, and HSMM
post-processing for temporal smoothing.

Public API
----------
- ``ScoreConfig`` : top-level configuration dataclass
- ``VideoQualityConfig`` : configuration for masking unreliable video measurements
- ``train_model`` : train a new model from sessions and labels
- ``profile_train_model`` : time the major stages of model training
- ``score_session`` : score a single session with a trained model
- ``evaluate_loso`` : leave-one-subject-out cross-validation
- ``evaluate_scored_session`` : compare one scored session against ground truth
- ``save_model`` / ``load_model`` : model persistence
- ``print_feature_importance`` : display feature importance from a trained model
- ``create_session`` : build a session dict from raw data identifiers
- ``load_labels_csv`` : load and validate a label CSV file
- ``score_sync_block`` : score one sync block with production-model versioning
- ``score_all_subjects`` : score all acquisition-master sync blocks
- ``autoscore_session`` / ``autoscore_all_subjects`` : batch scoring
"""

from ._config import (
    STATE_NAMES,
    STATE_TO_IDX,
    ClassifierConfig,
    HSMMConfig,
    ScoreConfig,
    VideoQualityConfig,
)
from ._data import create_session, load_labels_csv, validate_and_standardize_labels
from ._evaluate import evaluate_loso, evaluate_scored_session
from ._model import (
    load_model,
    print_feature_importance,
    profile_train_model,
    save_model,
    score_session,
    train_model,
)
from .score_all import (
    ScoreSyncBlockResult,
    autoscore_all_subjects,
    autoscore_session,
    score_all_subjects,
    score_sync_block,
)

__all__ = [
    "ScoreConfig",
    "HSMMConfig",
    "ClassifierConfig",
    "VideoQualityConfig",
    "STATE_NAMES",
    "STATE_TO_IDX",
    "train_model",
    "profile_train_model",
    "score_session",
    "evaluate_loso",
    "evaluate_scored_session",
    "save_model",
    "load_model",
    "print_feature_importance",
    "create_session",
    "load_labels_csv",
    "validate_and_standardize_labels",
    "ScoreSyncBlockResult",
    "score_sync_block",
    "score_all_subjects",
    "autoscore_session",
    "autoscore_all_subjects",
]
