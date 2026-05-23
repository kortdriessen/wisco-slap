# Sleep Scoring Module — Usage Instructions

## What this does

Classifies each 1-second epoch of a sync block recording into one of 3 states: **NREM**, **REM**, or **Wake**, using EEG bandpower features (at multiple resolutions), pupil/eyelid metrics, and whisker motion. A LightGBM classifier produces per-epoch probabilities, then an HSMM (Hidden Semi-Markov Model) smooths the output to enforce realistic state durations and transitions.

Low-likelihood video frames are masked before feature extraction:
- bad pupil tracking suppresses pupil diameter/motion features
- bad eyelid tracking suppresses eyelid-derived features
- true camera-off periods suppress both eye-derived and whisker-derived features
- those epochs are still scored, effectively using EEG-only evidence plus explicit quality flags

---

## Training data setup

For each sync block you want to include in training, save a label CSV somewhere under that sync block and pass its path to `load_labels_csv`. A common convention is:

```
/data/slap_analysis/analysis_materials/{subject}/{exp}/sync_block_data/sync_block-{sb}/hypnograms/training_labels/final_CC.csv
```

The CSV must have columns: `start_s`, `end_s`, `label`. Example:

```csv
start_s,end_s,label
100.0,250.0,NREM_SWS
250.0,340.0,REM
500.0,620.0,Wake
620.0,660.0,Artifact
```

- Gaps between labeled intervals are fine (unlabeled epochs are excluded from training).
- Labels are case-insensitive and accept several aliases: `NREM_SWS`, `NREM_light`, `REM_phasic`, `Wake_quiet`, `Artifact`, etc. They map to the 3 canonical states `NREM`, `REM`, and `Wake`.
- `Artifact` is treated as `Wake` by default. A held-out tuning sweep favored this default slightly, but `train_model(..., exclude_artifact=True)` can exclude Artifact intervals from classifier training.
- If you use `exclude_artifact=True`, reload labels with `load_labels_csv(...)` so the original label identity is available in the `raw_label` column.
- Obsolete `IS` / `intermediate_state` labels are rejected with a clear error.
- Every training run **must** have at least some examples of all 3 states across the combined training set.

---

## How to train

```python
import os
import polars as pl
import wisco_slap as wis
from wisco_slap.pns.sleepscore import (
    ScoreConfig,
    create_session,
    load_labels_csv,
    profile_train_model,
    train_model,
    save_model,
    print_feature_importance,
)

# 1. Define which sync blocks to use for training
training_blocks = [
    ("kaus", "exp_1", 1),
    ("alkaid", "exp_2", 1),
    ("alnair", "exp_1", 1),
    # ... add more as needed
]

# 2. Build sessions and load labels
sessions = []
labels_by_session = {}

for subject, exp, sb in training_blocks:
    session = create_session(subject, exp, sb)
    sessions.append(session)

    label_path = os.path.join(
        wis.defs.anmat_root, subject, exp,
        "sync_block_data", f"sync_block-{sb}",
        "hypnograms", "training_labels", "final_CC.csv",
    )
    labels_by_session[session["session_id"]] = load_labels_csv(label_path)

# 3. Train (uses tuned default config — 1s epochs, LightGBM, 3-state HSMM)
model = train_model(sessions, labels_by_session)

# 4. Inspect what the model learned
print_feature_importance(model, top_n=30)

# 5. Save
save_model(model, "/data/slap_analysis/analysis_materials/autoscore_model/MODEL.pkl")
```

Training should take seconds on this machine.

---

## How to score a single session

```python
from wisco_slap.pns.sleepscore import create_session, load_model, score_session

model = load_model("/data/slap_analysis/analysis_materials/autoscore_model/MODEL.pkl")
session = create_session("alkaid", "exp_1", 1)

bout_df, epoch_df = score_session(model, session)

# epoch_df has columns: session_id, epoch_idx, start_s, end_s,
#   pred_state_idx, pred_label, P_NREM, P_REM, P_Wake,
#   pu_valid_frac, eyelid_valid_frac, whisk_valid_frac,
#   camera_off_frac, camera_off_epoch
# bout_df has columns: session_id, start_s, end_s, label,
#   n_epochs, mean_P_NREM, mean_P_REM, mean_P_Wake,
#   mean_pu_valid_frac, mean_eyelid_valid_frac, mean_whisk_valid_frac,
#   mean_camera_off_frac, mean_camera_off_epoch
```

---

## How to score all subjects (batch)

For the production path, score a single sync block with:

```python
from wisco_slap.pns.sleepscore import score_sync_block

result = score_sync_block("avior", "exp_1", 1)
print(result.status, result.model_name)
```

By default, `score_sync_block(..., add_wake_epochs=0.8)` also patches the
saved bout predictions: any epoch with `P_Wake >= 0.8` is forced into the final
`bout_df`/`bout_hypno.csv` as Wake, with adjacent high-confidence Wake epochs
consolidated into one Wake bout. Disable that postprocessing with
`add_wake_epochs=False`, or pass another threshold such as
`add_wake_epochs=0.9`.

By default this uses the active model in:

```text
/data/slap_analysis/analysis_materials/autoscore_model/production_model/
```

and saves:

```text
analysis_materials/{subject}/{exp}/sync_block_data/sync_block-{sb}/hypnograms/model_scored/
    epoch_df.parquet
    bout_df.parquet
    epoch_hypno.csv
    bout_hypno.csv
    model_version.txt
```

The two `*_hypno.csv` files contain only `start_s`, `end_s`, and `label` for
direct Loupe loading. `bout_hypno.csv` comes from the HSMM bout predictions;
`epoch_hypno.csv` comes from raw per-epoch probability argmax predictions with
adjacent same-state epochs consolidated.

If `model_scored/` already exists with the same model version, it skips the
block. If the production model is newer or different, the existing
`model_scored/` folder is moved under `MODEL_ARCHIVED/` and fresh outputs are
written.

To score all currently scoreable sync blocks listed in
`acquisition_master.yaml`, use:

```python
from wisco_slap.pns.sleepscore import score_all_subjects

summary = score_all_subjects()
summary
```

This maps acquisition-master acquisitions to sync blocks via `sync_info.yaml`,
deduplicates repeated sync blocks, skips blocks missing raw ephys, eye metrics,
or whisking data, and returns a Polars summary table.

The older `autoscore_all_subjects(...)` helper is still available. It iterates
over every subject/exp/sync_block in `sync_info.yaml` and saves results as
parquet files at:

```python
from wisco_slap.pns.sleepscore import autoscore_all_subjects

autoscore_all_subjects(
    model_path="/data/slap_analysis/analysis_materials/autoscore_model/MODEL.pkl",
    overwrite=True,
)
```

```
analysis_materials/{subject}/{exp}/scoring_data/sync_block-{sb}/hypnograms/model_labelled/
    epochs.parquet
    bouts.parquet
```

---

## How to evaluate (leave-one-subject-out cross-validation)

```python
from wisco_slap.pns.sleepscore import evaluate_loso

# Use the same sessions and labels_by_session from training setup above
results = evaluate_loso(sessions, labels_by_session)

# Prints per-fold and overall precision/recall/F1/confusion matrix for each state.
# results["overall"] has the aggregate metrics.
# results["per_fold"]["kaus"] has per-subject metrics.
```

Requires at least 2 subjects with labeled data.

---

## Tuning

Everything is configurable via `ScoreConfig`:

```python
from wisco_slap.pns.sleepscore import (
    ClassifierConfig,
    ScoreConfig,
    HSMMConfig,
    VideoQualityConfig,
)

config = ScoreConfig(
    epoch_len_s=1.0,                        # epoch size (seconds)
    feature_windows_s=[1.0, 2.0, 4.0],     # EEG multi-resolution windows
    classifier=ClassifierConfig(
        lgbm_params={
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.10,
            "num_leaves": 15,
            "min_child_samples": 50,
            "reg_lambda": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
        },
        calibration_cv=5,
        fit_n_jobs=4,                      # per-LightGBM thread cap
        calibration_outer_n_jobs=1,       # avoid nested oversubscription
    ),
    hsmm=HSMMConfig(
        mean_dur_s={                        # expected bout durations (seconds)
            "NREM": 35.0,
            "REM": 55.0,
            "Wake": 30.0,
        },
        max_dur_s=450.0,                    # max bout duration
        lognorm_sigma=1.05,
        change_penalty=0.5,                 # larger = relatively more willing to switch
        stay_bonus=-0.25,                   # used only if self-transitions are allowed
        allow_self_transitions=False,       # explicit durations already encode persistence
        transition_log_adjustments={
            ("Wake", "REM"): -10.0,         # strongly discourage unlikely transitions
            ("REM", "NREM"): -10.0,
        },
    ),
    video_quality=VideoQualityConfig(
        pupil_valid_likelihood_min=0.75,
        eyelid_valid_likelihood_min=0.75,
        camera_off_likelihood_max=0.20,
        camera_off_whisk_quantile=0.05,
        camera_off_min_frame_frac=0.80,
    ),
)

model = train_model(sessions, labels_by_session, config=config)
```

The HSMM parameters are the main knobs for adjusting how smooth/sticky the output is. By default, explicit-duration self-transitions are disabled, so the decoder cannot create consecutive Wake -> Wake, NREM -> NREM, or REM -> REM segments; it must represent same-state persistence as one consolidated bout. Wake -> REM and REM -> NREM are strongly penalized but still possible if the classifier evidence is overwhelming.

---

## How to profile training time

```python
from wisco_slap.pns.sleepscore import (
    ClassifierConfig,
    ScoreConfig,
    profile_train_model,
)

debug_config = ScoreConfig(
    classifier=ClassifierConfig(calibration_cv=2),
)

profile = profile_train_model(
    sessions,
    labels_by_session,
    config=debug_config,
    feature_breakdown=True,
    verbose=True,
)

profile["timings_df"]
profile["feature_breakdown_df"]
profile["feature_session_df"]
```

This times the main `train_model(...)` stages:
- `build_features`
- `label_to_epoch_labels`
- `join_features_labels`
- `_prepare_matrix`
- zero-variance feature filtering
- calibration fit
- feature-importance aggregation from the calibrated estimators

If `fit_calibrator` dominates, check the configured `fit_n_jobs` and `calibration_outer_n_jobs`
before worrying about feature caching. If `build_features` dominates, the next likely step is
caching the generated feature table rather than recomputing those features every run.
