# Why spine size is measured on inactive-frame means, not on a pipeline "F0 image"

*Investigated 2026-08-31, prompted by a collaborator's suggestion to measure spine size on
"the F0 image". Pipeline code read: `ophys-slap2-analysis/matlab/preprocessing/`
(`processSLAP2.m → summarize_LoCo_ISOLATED → processAllTrials_Async → extractTrial`).*

## What the pipeline's F0 image actually is

- The LoCo solver (`extractTrial.m`) fits a constrained NMF, `Y ≈ H·X + B`. **`B` is the per-pixel,
  time-varying baseline** — the only true "F0 image" in the pipeline. Shape per trial per DMD:
  `[selected pixels × frames @ 200 Hz × 2 channels]`.
- The per-source `E{trial,dmd}.F0` in the ExperimentSummary (our `syn_F0` zarr) is
  `(Hᵢᵀ·B)/(HᵢᵀHᵢ)` — the footprint-weighted projection of B (`extractTrial.m:129`).
- **Data it is fit on:** raw `.dat` superpixels resampled to 200 Hz (not the 80 Hz registered TIFFs),
  motion-corrected, divided by `lambda = 25` (photon scale), and restricted to `selPix` = radius
  `ceil(2·dXY)` = 6 px (1.5 µm) disks around detected glutamate sources (~4 % of the FOV).
- **Estimator:** lower-envelope initialisation (`splitFreq`: 0.2 s block means → lowest ~15 % residuals
  → lowess over 4 s), then one per-pixel weighted `lsqlin` refit (`fitB`) regressing out `H·X` with sine
  drift + surround-neuropil predictors, floored at `lambda/10`. The calcium channel's B is the
  initialisation only.

## Is it saved?

**No.** `processAllTrials_Async.m:57`: `%E.baseline = single(B); %This is a huge variable and currently unused.`
(~17.6 GB per DMD per acquisition for bellatrix loc_Z acq_1.) Nothing else in the ExSum holds a per-pixel
baseline. The saved image-like fields are `perTrialMeanIMs(Aligned)` (all-frame trial means of the 80 Hz
TIFFs, full FOV — what spinelab's display planes use), `meanIM`, `actIM`, `selPix`.

## Could it be reconstructed?

| Route | Cost | Faithful? |
|---|---|---|
| Add a one-line save (e.g. `mean(B,2)` per trial, ~9 MB/DMD/acq) and reprocess | ~6.6 h wall per acquisition (measured); edit is in the collaborators' repo → needs their sign-off | yes |
| Per-pixel lower-envelope baseline on our existing aligned 80 Hz frame cache (`wis.pns.aligned_frames`) | minutes | no — different estimator, different data |
| Re-implement the `H·X + B` regression in Python | days | approximately |

## Conclusion: keep the Wright inactive-frame mean

1. **Coverage.** B exists only in 1.5 µm disks around sources. Wright's estimator needs shaft disks over
   ±20 µm and a background ROI, both outside `selPix` — the numerator and denominator would come from
   different estimators with different biases.
2. **Bias.** B is initialised as, and regularised toward, a lower envelope. Envelope/percentile baselines
   are biased low under shot noise, and the bias is count-dependent (dim heads more than bright shafts),
   so it distorts the head/shaft ratio and drifts with bleaching. An inactive-frame mean is unbiased.
3. **Activity removal** — the thing F0 is meant to provide — is exactly what the inactive-frame mask
   already does (`spinelab.activity.detect_active_frames`, Wright's recipe). `active_frame_fraction` is
   recorded per spine/bin in the measurement table, so its magnitude is checkable.
4. **Slow drift** largely cancels in Wright's ratio because head and shaft are read from the same projection.

**Worth borrowing:** the pipeline's activity model (`dF.events` / `dF.denoised` at 200 Hz) is a better
active-frame detector than the head-minus-background TIFF trace we build ourselves. Using it as the *mask
source* (downsampled to the 80 Hz cache, per trial) would tighten the projections without changing the
estimator. If a "pipeline-F0" sensitivity variant is ever wanted, the honest route is the one-line save +
reprocess; an in-house envelope baseline may be added as a clearly labelled secondary column but must not be
presented as the pipeline's F0.
