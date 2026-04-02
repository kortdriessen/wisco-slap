Raw least-squares (LS) fluorescence traces for each synapse were convolved with a template kernel matched to the expected shape of a single glutamate transient. The kernel was a causal exponential decay with a time constant of 30 ms (approximating the iGluSnFR4f impulse response), truncated at 5 time constants and normalized to unit energy. The convolution was computed via FFT for all synapses simultaneously.

To handle missing data (NaN-valued samples from censored or motion-artifact timepoints), NaNs were replaced with zeros before filtering, and a parallel convolution of a binary validity mask with a uniform kernel of the same length was used to count how many valid samples contributed to each output timepoint. Output samples where fewer than 50% of the kernel window contained valid data were set to NaN, preventing edge artifacts and unreliable estimates from corrupting the filtered trace.

Time-Varying Noise Estimation
Noise was estimated locally in time for each synapse using a rolling median absolute deviation (MAD) approach applied to the matched-filtered trace. Specifically:

A rolling median of the filtered trace was computed over a centered 8-second window.
The absolute deviations from this rolling median were computed sample-by-sample.
A second rolling median (same 8-second window) was applied to these absolute deviations, yielding a rolling MAD.
The MAD was converted to a Gaussian-equivalent standard deviation by multiplying by 1.4826 (the theoretical ratio σ/MAD for a normal distribution).
A minimum of 25% of the window samples had to be non-NaN for the estimate to be valid. This procedure was parallelized across synapses using a thread pool.

The key advantage of this approach over a global (single-value) noise estimate is that it tracks changes in noise level throughout the recording — for example, due to photobleaching, slow motion artifacts, or brain-state-dependent changes in baseline fluorescence. By using the median rather than the mean, and by operating on deviations from a local median baseline, the estimator is robust to the sparse, transient glutamate events themselves: genuine events are outliers relative to the local noise floor and do not substantially bias the MAD.

Event Detection via SNR Thresholding
An SNR trace was computed for each synapse by dividing the matched-filter output by the local noise estimate (scaled by a threshold factor of 4). Timepoints where SNR exceeded 1 (equivalently, where the filtered signal exceeded 4 times the local noise standard deviation) were marked as "active." Contiguous runs of active samples were grouped into discrete events.

For each event, the following quantities were recorded: onset and offset times, duration, peak and mean SNR within the event, the peak matched-filter amplitude, and the time-integral of the filtered signal over the event duration.