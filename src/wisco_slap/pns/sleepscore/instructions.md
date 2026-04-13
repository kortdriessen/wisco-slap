# Sleep Scoring Module — Usage Instructions

## What this does

Classifies each 1-second epoch of a sync block recording into one of 4 states: **NREM**, **IS** (intermediate state), **REM**, or **Wake**, using EEG bandpower features (at multiple resolutions), pupil/eyelid metrics, and whisker motion. A LightGBM classifier produces per-epoch probabilities, then an HSMM (Hidden Semi-Markov Model) smooths the output to enforce realistic state durations and transitions.

---

## Training data setup

For each sync block you want to include in training, place your labels at:

```
/data/slap_analysis/analysis_materials/{subject}/{exp}/sync_block_data/sync_block-{sb}/hypnograms/training_labels.csv
```

The CSV must have columns: `start_s`, `end_s`, `label`. Example:

```csv
start_s,end_s,label
100.0,250.0,NREM
250.0,268.0,IS
268.0,340.0,REM
500.0,620.0,Wake
```

- Gaps between labeled intervals are fine (unlabeled epochs are excluded from training).
- Labels are case-insensitive and accept several aliases: `NREM_SWS`, `NREM_light`, `intermediate_state`, `REM_phasic`, `Wake_quiet`, etc. — they all map to the 4 canonical states.
- Every training run **must** have at least some examples of all 4 states across the combined training set.

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
        "hypnograms", "training_labels.csv",
    )
    labels_by_session[session["session_id"]] = load_labels_csv(label_path)

# 3. Train (uses default config — 1s epochs, LightGBM, 4-state HSMM)
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
#   pred_state_idx, pred_label, P_NREM, P_IS, P_REM, P_Wake
# bout_df has columns: session_id, start_s, end_s, label,
#   n_epochs, mean_P_NREM, mean_P_IS, mean_P_REM, mean_P_Wake
```

---

## How to score all subjects (batch)

```python
from wisco_slap.pns.sleepscore import autoscore_all_subjects

autoscore_all_subjects(
    model_path="/data/slap_analysis/analysis_materials/autoscore_model/MODEL.pkl",
    overwrite=True,
)
```

This iterates over every subject/exp/sync_block in `sync_info.yaml` and saves results as parquet files at:
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
from wisco_slap.pns.sleepscore import ScoreConfig, HSMMConfig

config = ScoreConfig(
    epoch_len_s=1.0,                        # epoch size (seconds)
    feature_windows_s=[1.0, 2.0, 4.0],     # EEG multi-resolution windows
    hsmm=HSMMConfig(
        mean_dur_s={                        # expected bout durations (seconds)
            "NREM": 30.0,
            "IS": 15.0,
            "REM": 60.0,
            "Wake": 30.0,
        },
        max_dur_s=300.0,                    # max bout duration
        change_penalty=-0.8,                # more negative = stickier states
        stay_bonus=0.2,                     # more positive = stickier states
    ),
)

model = train_model(sessions, labels_by_session, config=config)
```

The HSMM parameters are the main knobs for adjusting how smooth/sticky the output is. If you're getting too many short spurious bouts, increase `change_penalty` magnitude or `stay_bonus`. If real transitions are being smoothed away, decrease them.
