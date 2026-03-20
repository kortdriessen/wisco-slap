# Glutamate Event Pipeline Review

**Findings**
- High severity: output freshness ignores detector settings. [glut_ev_mon.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_mon.py#L68) only checks parquet existence plus the ExSum version file, while [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L688) writes `params.json` that is never consulted. Changing `amp_thresh_sigma`, `tau_s`, `lam_mult`, or EB settings can silently reuse stale results unless `overwrite=True`.
- High severity: trial structure is discarded before detection. [scopex_mon.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/scopex_mon.py#L157) loads per-trial data, [xarr_summ.py](/data/slap_analysis/slap2-py/src/slap2_py/core/xarr_summ.py#L14) concatenates trials into one continuous `time` axis, and [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L448) then fills NaNs and filters/deconvolves the whole acquisition as one trace. That makes the detector trial-blind and vulnerable to join/gap artifacts.
- Medium severity: saved EB metadata is wrong. The second pass rescales `alpha` at [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L476), but final event rows are stamped with the non-EB `alpha` and `eb_scale = 1.0` at [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L496). The calls may change, but the exported metadata does not report that honestly.

**Summary**
- Trace lineage: `scopex` reads ExSum trial data, saves `syn_dF-ls`, `syn_dF-denoised`, `syn_dF-events`, and `syn_F0`; this detector uses only `syn_dF-ls` and `syn_F0`, channel `0` by default. See [scopex_mon.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/scopex_mon.py#L162), [xarr_summ.py](/data/slap_analysis/slap2-py/src/slap2_py/core/xarr_summ.py#L8), and [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L56).
- What it actually does: estimate per-synapse sigma from first-difference MAD on LS, linearly fill NaNs, high-pass at `0.1 Hz`, deconvolve with a mono-exponential kernel (`tau=26 ms`) using nonnegative L1 FISTA, threshold `a_hat > amp_thresh_sigma * sigma`, merge within `10 ms`, then optionally do a second EB pass that nudges `alpha` from the population event-rate center. See [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L120), [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L165), and [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L383).
- Assessment: using the LS trace is the right foundation. I agree with the core choice not to deconvolve `denoised` or `events`. But the current detector is a solid prototype, not an optimal LS detector yet, because it is trial-blind, uses a generic baseline/noise model, and collapses time-varying `F0` to one number per synapse.
- Empirical check: in [loc_I--acq_1](/data/slap_analysis/analysis_materials/avior/exp_1/scopex/loc_I--acq_1), each DMD is `124` trials x `1936` frames concatenated to `240064` samples at `200 Hz`; the saved detector run in [params.json](/data/slap_analysis/analysis_materials/avior/exp_1/activity_data/loc_I/acq_1/glut_events/params.json) used `amp_thresh_sigma=8.0` and still marked about 97-99% of synapses active. `F0` also drifted substantially over time, so median-`F0` normalization is only coarse.
- EB looks helpful but not transformative. On a 20-synapse, 50k-sample real subset, EB changed calls from `146` to `175` events, mostly by adding events to already-active synapses rather than rescuing silent ones.

**Improvement Plan**
- Preserve trial structure in `scopex`. Save `trial_id`, trial start indices, and a valid/discard mask, or detect directly per trial instead of after flattening.
- Run baseline filtering and deconvolution per contiguous valid segment. Do not interpolate across bad trials, masked spans, or future discarded frames.
- Keep LS as the input, but replace whole-trace median `F0` with local `F0(t_event)` or a local baseline window for `amp_dff`.
- Recalibrate the final decision rule. Keep FISTA if you want, but threshold using deconvolved-noise or residual statistics, not only raw-LS sigma.
- Make kernel choice acquisition-aware. Start with the current mono-exponential model, but allow fitted `tau` or a rise+decay kernel when data show slower or broader transients.
- Keep EB only if it wins on validation. If retained, export the true `alpha`/`eb_scale` and treat EB as a tunable prior, not a hidden default.
- Add reproducibility guards: hash detector params into freshness checks, or compare `params.json` before skipping regeneration.
- Export QC alongside calls: residual MAD, reconstruction fit, pre/post-EB event counts, fraction of events near boundaries, and local-vs-median `F0` summaries.

**Test Plan**
- Synthetic injections into real LS backgrounds: precision/recall, timing error, overlap handling, and tau mismatch.
- Boundary tests at trial joins, bad-trial gaps, and masked spans: confirm no bridge artifacts.
- Reproducibility tests: changing `tau_s` or threshold must force regeneration.
- Normalization tests: compare event amplitudes from median `F0` vs local `F0(t)` on bleaching-heavy acquisitions.
- EB validation: compare no-EB vs EB on held-out annotated or synthetic data; keep EB only if quality improves, not just call count.
- I did not find dedicated tests for this pipeline in `wisco-slap`, so I would treat adding these as part of the improvement work.

**Predicted Failure Modes**
- False events at trial joins or gap boundaries once bad trials/masked frames appear, because the current pipeline treats the acquisition as one continuous signal.
- Biased amplitude comparisons across time or behavioral state, because `amp_dff` uses a whole-acquisition median `F0` rather than the local baseline.
- Synapse-to-synapse rate bias, because the EB pass pushes rates toward the population center and, in my slice test, mainly boosted already-active synapses.

**Assumptions**
- I assumed channel `0` is the iGluSnFR channel for this detector, consistent with [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L647).
- I assumed the stored LS trace is the temporally unmodeled spatial-unmixing output and therefore the correct raw substrate for a custom temporal detector, consistent with [glut_ev_gen.py](/data/slap_analysis/wisco-slap/src/wisco_slap/pns/glut_ev_gen.py#L1) and [scope/pro.py](/data/slap_analysis/wisco-slap/src/wisco_slap/scope/pro.py#L157).
