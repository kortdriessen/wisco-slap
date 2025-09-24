"""
Video generation utilities for synaptic activity visualization
"""

import os

import cv2
import numpy as np
from tqdm import tqdm


def generate_synaptic_video(
    base_image_data: np.ndarray,
    source_traces: list[np.ndarray],
    source_masks: list[np.ndarray],
    fs: float = 20.0,
    output_path: str = "synaptic_activity.mp4",
    # Red overlay controls
    red_alpha_max: float = 0.8,
    red_alpha_min: float = 0.0,
    red_intensity_max: float = 1.0,
    red_intensity_min: float = 0.0,
    trace_percentile_clip: float = 99.0,
    # Base image controls
    base_vmin_percentile: float = 1.0,
    base_vmax_percentile: float = 99.0,
    # Video quality controls
    video_codec: str = "mp4v",
    video_quality: int = 80,
    # Performance controls
    use_parallel: bool = True,
    chunk_size: int = 100,
) -> str:
    """
    Generate a video of synaptic activity overlaid on a base neuron image.

    This function creates a real-time video where synaptic sources flash red
    according to their fluorescence traces, overlaid on a grayscale base image.

    Parameters
    ----------
    base_image_data : np.ndarray
        Base image of the neuron structure (2D array). NaN values will appear black.
    source_traces : List[np.ndarray]
        List of N_sources fluorescence traces over time. Each trace should be 1D.
    source_masks : List[np.ndarray]
        List of N_sources boolean masks, same shape as base_image_data.
        True pixels belong to each respective source.
    fs : float, default=20.0
        Sampling frequency of the traces (Hz). Determines video frame rate.
    output_path : str, default="synaptic_activity.mp4"
        Path where the output video will be saved.

    Red Overlay Controls
    -------------------
    red_alpha_max : float, default=0.8
        Maximum opacity for red overlay at peak activity.
    red_alpha_min : float, default=0.0
        Minimum opacity for red overlay at baseline activity.
    red_intensity_max : float, default=1.0
        Maximum red color intensity (0-1).
    red_intensity_min : float, default=0.0
        Minimum red color intensity (0-1).
    trace_percentile_clip : float, default=99.0
        Percentile for clipping extreme trace values (helps with dynamic range).

    Base Image Controls
    -------------------
    base_vmin_percentile : float, default=1.0
        Percentile for minimum grayscale value in base image.
    base_vmax_percentile : float, default=99.0
        Percentile for maximum grayscale value in base image.

    Video Quality Controls
    ----------------------
    video_codec : str, default="mp4v"
        Video codec to use. Options: "mp4v", "XVID", "H264" (if available).
    video_quality : int, default=80
        Video quality (0-100). Higher values = better quality, larger files.

    Performance Controls
    --------------------
    use_parallel : bool, default=True
        Whether to use optimized frame generation (recommended).
    chunk_size : int, default=100
        Number of frames to process in memory at once (affects memory usage).

    Returns
    -------
    str
        Path to the generated video file.

    Raises
    ------
    ValueError
        If input arrays have mismatched dimensions or invalid parameters.
    FileNotFoundError
        If output directory doesn't exist.

    Examples
    --------
    >>> # Generate video with default settings
    >>> video_path = generate_synaptic_video(
    ...     base_image_data=base_img,
    ...     source_traces=[trace1, trace2],
    ...     source_masks=[mask1, mask2],
    ...     fs=30.0,
    ...     output_path="my_video.mp4"
    ... )

    >>> # High-quality video with custom red overlay
    >>> video_path = generate_synaptic_video(
    ...     base_image_data=base_img,
    ...     source_traces=traces,
    ...     source_masks=masks,
    ...     fs=20.0,
    ...     red_alpha_max=0.9,
    ...     red_intensity_max=0.8,
    ...     video_quality=95
    ... )
    """

    # Input validation
    _validate_inputs(base_image_data, source_traces, source_masks, fs)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get video dimensions and parameters
    height, width = base_image_data.shape
    n_sources = len(source_traces)
    n_frames = len(source_traces[0])

    print(f"Generating video: {width}x{height}, {n_frames} frames, {n_sources} sources")
    print(f"Duration: {n_frames / fs:.2f} seconds at {fs} fps")

    # Prepare base image for video
    base_rgb = _prepare_base_image(
        base_image_data, base_vmin_percentile, base_vmax_percentile
    )

    # Prepare normalized traces
    normalized_traces = _prepare_traces(source_traces, trace_percentile_clip)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fs, (width, height), True)

    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        # Generate video frames
        if use_parallel and chunk_size > 1:
            _generate_frames_chunked(
                video_writer,
                base_rgb,
                source_masks,
                normalized_traces,
                red_alpha_min,
                red_alpha_max,
                red_intensity_min,
                red_intensity_max,
                chunk_size,
                n_frames,
            )
        else:
            _generate_frames_sequential(
                video_writer,
                base_rgb,
                source_masks,
                normalized_traces,
                red_alpha_min,
                red_alpha_max,
                red_intensity_min,
                red_intensity_max,
                n_frames,
            )

    finally:
        video_writer.release()

    # Verify output file
    if not os.path.exists(output_path):
        raise RuntimeError(f"Failed to create video file: {output_path}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Video saved successfully: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")

    return output_path


def _validate_inputs(
    base_image_data: np.ndarray,
    source_traces: list[np.ndarray],
    source_masks: list[np.ndarray],
    fs: float,
) -> None:
    """Validate all input parameters."""

    # Check base image
    if not isinstance(base_image_data, np.ndarray):
        raise ValueError("base_image_data must be a numpy array")
    if base_image_data.ndim != 2:
        raise ValueError("base_image_data must be a 2D array")

    # Check lists
    if not source_traces or not source_masks:
        raise ValueError("source_traces and source_masks cannot be empty")
    if len(source_traces) != len(source_masks):
        raise ValueError("source_traces and source_masks must have same length")

    # Check traces
    trace_lengths = [len(trace) for trace in source_traces]
    if len(set(trace_lengths)) > 1:
        raise ValueError("All source_traces must have the same length")

    # Check masks
    base_shape = base_image_data.shape
    for i, mask in enumerate(source_masks):
        if not isinstance(mask, np.ndarray):
            raise ValueError(f"source_masks[{i}] must be a numpy array")
        if mask.shape != base_shape:
            raise ValueError(
                f"source_masks[{i}] shape {mask.shape} doesn't match base_image_data shape {base_shape}"
            )

    # Check sampling frequency
    if fs <= 0:
        raise ValueError("fs must be positive")


def _prepare_base_image(
    base_image_data: np.ndarray, vmin_percentile: float, vmax_percentile: float
) -> np.ndarray:
    """Prepare base image for video overlay."""

    # Handle NaN values - set to black
    base_clean = base_image_data.copy()
    nan_mask = np.isnan(base_clean)

    # Compute percentiles on non-NaN values
    valid_data = base_clean[~nan_mask]
    if len(valid_data) == 0:
        raise ValueError("Base image contains only NaN values")

    vmin = np.percentile(valid_data, vmin_percentile)
    vmax = np.percentile(valid_data, vmax_percentile)

    # Normalize to 0-1 range
    base_normalized = np.clip((base_clean - vmin) / (vmax - vmin), 0, 1)

    # Set NaN values to black (0)
    base_normalized[nan_mask] = 0

    # Convert to RGB (grayscale)
    base_rgb = np.stack([base_normalized] * 3, axis=-1)

    return base_rgb


def _prepare_traces(
    source_traces: list[np.ndarray], percentile_clip: float
) -> list[np.ndarray]:
    """Normalize traces to 0-1 range."""

    normalized_traces = []

    for trace in source_traces:
        trace_clean = np.array(trace, dtype=np.float64)

        # Remove any NaN/inf values
        valid_mask = np.isfinite(trace_clean)
        if not np.any(valid_mask):
            # All values are invalid, use zeros
            normalized_traces.append(np.zeros_like(trace_clean))
            continue

        # Clip extreme values
        valid_values = trace_clean[valid_mask]
        clip_min = np.percentile(valid_values, 100 - percentile_clip)
        clip_max = np.percentile(valid_values, percentile_clip)

        trace_clipped = np.clip(trace_clean, clip_min, clip_max)

        # Normalize to 0-1
        trace_range = clip_max - clip_min
        if trace_range > 0:
            trace_normalized = (trace_clipped - clip_min) / trace_range
        else:
            trace_normalized = np.zeros_like(trace_clipped)

        # Handle invalid values
        trace_normalized[~valid_mask] = 0

        normalized_traces.append(trace_normalized)

    return normalized_traces


def _generate_frames_sequential(
    video_writer: cv2.VideoWriter,
    base_rgb: np.ndarray,
    source_masks: list[np.ndarray],
    normalized_traces: list[np.ndarray],
    alpha_min: float,
    alpha_max: float,
    intensity_min: float,
    intensity_max: float,
    n_frames: int,
) -> None:
    """Generate video frames one at a time."""

    for frame_idx in tqdm(range(n_frames), desc="Generating frames"):
        frame = _create_single_frame(
            base_rgb,
            source_masks,
            normalized_traces,
            frame_idx,
            alpha_min,
            alpha_max,
            intensity_min,
            intensity_max,
        )
        video_writer.write(frame)


def _generate_frames_chunked(
    video_writer: cv2.VideoWriter,
    base_rgb: np.ndarray,
    source_masks: list[np.ndarray],
    normalized_traces: list[np.ndarray],
    alpha_min: float,
    alpha_max: float,
    intensity_min: float,
    intensity_max: float,
    chunk_size: int,
    n_frames: int,
) -> None:
    """Generate video frames in chunks for better performance."""

    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="Processing chunks"):
        chunk_end = min(chunk_start + chunk_size, n_frames)

        # Generate chunk of frames
        for frame_idx in range(chunk_start, chunk_end):
            frame = _create_single_frame(
                base_rgb,
                source_masks,
                normalized_traces,
                frame_idx,
                alpha_min,
                alpha_max,
                intensity_min,
                intensity_max,
            )
            video_writer.write(frame)


def _create_single_frame(
    base_rgb: np.ndarray,
    source_masks: list[np.ndarray],
    normalized_traces: list[np.ndarray],
    frame_idx: int,
    alpha_min: float,
    alpha_max: float,
    intensity_min: float,
    intensity_max: float,
) -> np.ndarray:
    """Create a single video frame with red activity overlay."""

    # Start with base image
    frame = base_rgb.copy()

    # Add red overlay for each active source
    for _source_idx, (mask, trace) in enumerate(
        zip(source_masks, normalized_traces, strict=False)
    ):
        activity = trace[frame_idx]

        if activity <= 0:
            continue  # Skip inactive sources

        # Calculate red overlay properties
        alpha = alpha_min + activity * (alpha_max - alpha_min)
        intensity = intensity_min + activity * (intensity_max - intensity_min)

        # Create red overlay
        red_overlay = np.zeros_like(frame)
        red_overlay[mask, 0] = intensity  # Red channel only

        # Blend with frame using alpha
        mask_3d = np.stack([mask] * 3, axis=-1)
        frame = np.where(mask_3d, (1 - alpha) * frame + alpha * red_overlay, frame)

    # Convert to uint8 BGR for OpenCV
    frame_uint8 = (frame * 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

    return frame_bgr


# Convenience function for quick video generation
def quick_synaptic_video(
    base_image_data: np.ndarray,
    source_traces: list[np.ndarray],
    source_masks: list[np.ndarray],
    output_path: str = "quick_video.mp4",
    fs: float = 20.0,
) -> str:
    """
    Quick video generation with sensible defaults.

    Parameters
    ----------
    base_image_data : np.ndarray
        Base neuron image
    source_traces : List[np.ndarray]
        Fluorescence traces for each source
    source_masks : List[np.ndarray]
        Boolean masks for each source
    output_path : str
        Output video path
    fs : float
        Sampling frequency (frame rate)

    Returns
    -------
    str
        Path to generated video
    """
    return generate_synaptic_video(
        base_image_data=base_image_data,
        source_traces=source_traces,
        source_masks=source_masks,
        fs=fs,
        output_path=output_path,
        # Optimized defaults for quick generation
        red_alpha_max=0.7,
        video_quality=75,
        chunk_size=50,
    )
