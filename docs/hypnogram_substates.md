# Automatic hypnogram substates

All automatic model hypnogram loaders apply the approved three-pass detector:

```python
import wisco_slap as wis

# Hypnogram object, using the whole sync block for calibration.
h = wis.get.bout_hypno('alnilam', 'exp_1', 'loc_A', 'acq_1')
h = wis.get.sb_bout_hypno('alnilam', 'exp_1', 1)

# Same labels; preserves the existing pandas DataFrame return format.
table = wis.peri.load_auto_hypno('alnilam', 'exp_1', 1)
# Also available at wis.peri.anno.load_auto_hypno.

# Broad labels only; no EEG or camera loading.
broad = wis.get.bout_hypno(
    'alnilam', 'exp_1', 'loc_A', 'acq_1', apply_substates=False,
)
```

Final states are `Active wake`, `Quiet wake`, `NREM SWS`, `NREM sigma`, and
`REM`. Other original labels remain unchanged; unannotated EEG time remains
`Unscored`. Manual and validated annotations are always returned unchanged.

The contained, data-only entry point is:

```python
h = wis.peri.anno.label_hypnogram_substates(
    original_hypno, eeg, eye_metrics, whisk_df,
    step1_config={'slow_activity_max': .15, 'eeg_seed_fold': 2.},
    step2_config={'bridge_budget_s': 8., 'edge_budget_s': 2.},
    step3_config={'sigma_slow_min': 1., 'sigma_fraction_min': .28},
)
```

The input hypnogram can be an electro_py Hypnogram, polars DataFrame, or pandas
DataFrame. EEG is a one-dimensional xarray DataArray with `time` and `fs`.
Eye and whisker frames are optional. All inputs must use native TDT sync-block
seconds. Use a whole block for calibration and trim the result afterward.
Load cell is never used. Already subsetted inputs are rejected.

All three configuration classes are exported in `wis.peri.anno`:

| Configuration | Approved defaults |
|---|---|
| `ActiveWakeConfig` | `eeg_seed_fold=2`, `assisted_seed_fold=2`, `edge_relax_fraction=.1`, `slow_activity_max=.15`, `slow_fraction_max=.5`, `motion_z=3`, `seed_min_s=2`, `smooth_seconds=3`, `peak_fraction_max=.65` |
| `WakeMergeConfig` | `bridge_budget_s=8`, `edge_budget_s=2`, `quiet_slow_min=.5`, `still_motion_max=1.5`, `quiet_weight=4`, `quiet_min_s=2`, `smooth_seconds=3`, `peak_fraction_max=.65` |
| `NremSigmaConfig` | `sigma_slow_min=1`, `sigma_fraction_min=.28`, `support_fraction=.7`, `bridge_max_s=3`, `seed_min_s=2`, `sigma_beta_density_min=1.5`, `bridge_score_min=.5`, `slow_barrier_ratio=.5`, `smooth_seconds=3`, `peak_fraction_max=.65` |

Each argument accepts a configuration object or a partial dictionary; omitted
fields retain the approved defaults. Measurement options are `grid_origin_s`,
`window_s=4`, `nrem_window_s=4`, and `eye_likelihood_min=.75`. Decisions remain
on a one-second grid, also retaining off-grid original-state boundaries.
These parameters are forwarded by every automatic loader:

```python
h = wis.get.bout_hypno(
    'alnilam', 'exp_1', 'loc_A', 'acq_1',
    step3_config={'sigma_slow_min': 1.2, 'bridge_max_s': 2.},
)
```

1. Select clear Active wake from activated/gamma/muscle EEG or a separate
   low-slow route. Motion can assist but is not required. Delta never vetoes
   strong high-frequency evidence.
2. Freeze those anchors; absorb short weak-quiet gaps and edges using duration
   budgets weighted by slow/still evidence. Protect sustained slow-and-still
   cores. Remaining Wake becomes Quiet wake.
3. Seed sustained sigma-rich NREM using sigma/slow integrated power and sigma's
   share of total power, with a sigma/beta density guard. Grow supported edges,
   bridge only short supported gaps, and stop at clear slow dominance.
   Remaining NREM becomes SWS.

For continuous features, validity and decision audits:

```python
r = wis.peri.anno.load_auto_hypno('alnilam', 'exp_1', 1, return_details=True)
epochs = r['epochs']       # Original/intermediate/final labels and all features.
bouts = r['bouts']         # Consolidated start_s/end_s/label intervals.
metadata = r['metadata']   # Settings, calibration, source paths and code hashes.
```

Epochs retain `original_state`, `stage1_label`, `stage2_label`, `final_label`,
EEG/tracking validity, selection reasons, protected quiet cores, and sigma
seed/edge/bridge masks. Unusable EEG defaults remaining Wake to Quiet wake and
remaining NREM to SWS, with explicit validity/reason fields. Missing video is
not treated as stillness. SWS is the residual NREM label, not proof of strong
slow waves in every epoch. These are spectral substates, not individual spindle
annotations or calibrated confidence probabilities.

Current input is `sync_block_data/sync_block-N/hypnograms/model_scored/bout_hypno.csv`,
the HSMM bouts including production Wake insertion. Raw epoch argmax labels are
not substituted. The older `scoring_data/.../model_labelled/raw_epochs.csv`
layout remains a fallback. `filter_unclear` is still opt-in; current model
files provide raw probabilities, so use `filter_on='raw'` with that filter.
Unavailable smooth probabilities produce an explicit error.

No scoring files are modified and no persistent substate cache is created.
Automatic loading now reads EEG and available camera metrics and computes the
full block; expect several seconds per block. Missing EEG files raise an error;
`apply_substates=False` permits broad loading without EEG. Fleet occupancy
summaries reuse each computed block within a call.

Built-in broad occupancy summaries combine Active/Quiet wake and SWS/sigma NREM.
Other analyses using exact `Wake`/`NREM` filters should aggregate the new labels
or explicitly request broad labels. `wis.defs.substate_parent` supplies the
mapping. Notebook tuners import the packaged detector and explicitly load
original broad labels, so they cannot apply the detector twice.
