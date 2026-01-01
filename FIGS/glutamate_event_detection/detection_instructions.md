# Role
You are an expert Bio-Signal Processing Engineer specializing in synaptic physiology and optical imaging. Your task is to write a complete Python pipeline to detect glutamate release events from 2-photon iGluSnFR4f traces.

# Context
Sensor: iGluSnFR4f (Fast variant).
Decay Constant ($\tau$): ~26 ms.
Sampling Rate ($fs$): 200 Hz.
Data Structure: data will be given as a numpy array of samples, starting at t=0. You can find some example data (sampled at 200Hz) at example_data.npy
Artifacts: The data contains NaN values representing time periods intentionally excluded due to motion correction.

# Pipeline Architecture
You will implement a pipeline using Constrained Deconvolution (OASIS/FOOPSI).

## Step 1: Preprocessing & Cleaning
- NaN Handling (CRITICAL): Do not interpolate over NaN gaps, as they represent invalid data/motion. Instead, split the trace into valid contiguous segments. Process each segment independently, then stitch the results back together, leaving NaN in the gaps.
- Baseline Correction ($F_0$): Implementation of a rolling percentile filter.
  - Window: 1.0 second (200 frames).
  - Percentile: 20th (Estimates the noise floor while ignoring events).
- Normalization: Calculate $\Delta F/F_0 = (F - F_0) / F_0$.
- Trend Removal: If necessary, detrend segments linearly to remove photobleaching drift, but the rolling baseline usually handles this.

## Step 2: Deconvolution (OASIS Algorithm)
- Use the OASIS algorithm (Online Active Set method for Spikes).
- Model: Autoregressive AR(1).
- Parameter $g$ (Gamma): Calculate explicitly using the sensor kinetics:
$$g = e^{-1 / (\tau \cdot fs)}$$
For $\tau=26\text{ms}$ and $fs=200\text{Hz}$, $g \approx 0.82$.
- Sparsity / Noise Constraint:
    - Estimate noise ($\sigma$) robustly using the median absolute deviation (MAD) of the first derivative of the trace.
    - Set the min_spike_magnitude ($s_{min}$) constraint to $3 \sigma$.

## Step 3: Event Extraction
- The deconvolution outputs a "spike" trace ($S$) which is sparse (mostly zeros).
- Identify events where $S > 0$ (or a small threshold like $0.1 \sigma$).
- Outputs per synapse: List of (timestamp, magnitude) tuples.
## Step 4: Quality Control (QC) & Debugging
- Generate a "QC Dashboard" plot for the user to inspect:
- Top Panel: Overlay of Raw Trace (Black) and Reconstructed Trace (Red, convolution of extracted spikes).
- Middle Panel: The Deconvolved Spikes ($S$) (Stem plot).
- Bottom Panel: Residuals ($Data - Model$).
- Metrics: Calculate Signal-to-Noise Ratio (SNR) for the detected events.
# Implementation (Python)
- Write the solution in a single Python file.
- Libraries: pandas, numpy, scipy, matplotlib.
- OASIS: If oasis-deconv is not available in the environment, implement a simplified "Thresholded Non-Negative Deconvolution" using scipy.optimize or a standard FOOPSI loop. However, prefer using the library logic if possible.
