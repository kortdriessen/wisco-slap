#!/usr/bin/env python3
"""
Example usage of the synaptic video generation function.

This script demonstrates how to use the generate_synaptic_video function
with different parameter configurations to create videos of synaptic activity.
"""

import numpy as np

import wisco_slap.util.video as video


# Example 1: Basic usage with real-like data
def example_basic_usage():
    """Basic example with synthetic but realistic data."""

    print("Example 1: Basic Usage")
    print("-" * 30)

    # Create base image data (simulating a 2-photon neuron image)
    height, width = 256, 256
    base_image_data = np.random.normal(0.3, 0.1, (height, width))

    # Add dendrite-like structures
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Cell body
    cell_body = ((x - center_x) ** 2 + (y - center_y) ** 2) < 40**2
    base_image_data[cell_body] = 0.8

    # Some dendrites
    for angle in [0, np.pi / 3, 2 * np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3]:
        for r in range(10, 80, 3):
            x_pos = int(center_x + r * np.cos(angle))
            y_pos = int(center_y + r * np.sin(angle))
            if 0 <= x_pos < width and 0 <= y_pos < height:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x_pos + dx, y_pos + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            base_image_data[ny, nx] = max(
                                base_image_data[ny, nx],
                                0.6 - r * 0.005 + np.random.normal(0, 0.05),
                            )

    # Set background to NaN for black appearance
    background_mask = base_image_data < 0.2
    base_image_data[background_mask] = np.nan

    # Create source masks (synaptic locations)
    n_sources = 5
    source_masks = []

    # Random synapse locations
    np.random.seed(42)  # For reproducible results
    for i in range(n_sources):
        mask = np.zeros((height, width), dtype=bool)
        # Random location not too close to center or edges
        sx = np.random.randint(50, width - 50)
        sy = np.random.randint(50, height - 50)
        # Small circular region
        synapse_region = ((x - sx) ** 2 + (y - sy) ** 2) < np.random.randint(3, 8) ** 2
        mask[synapse_region] = True
        source_masks.append(mask)

    # Create fluorescence traces (5 seconds at 30 fps = 150 frames)
    fs = 30.0
    duration = 5.0
    n_frames = int(fs * duration)
    time = np.linspace(0, duration, n_frames)

    source_traces = []
    for i in range(n_sources):
        # Different activity patterns for each synapse
        if i == 0:
            # Slow oscillation
            trace = 0.3 + 0.4 * np.sin(2 * np.pi * 0.5 * time)
        elif i == 1:
            # Fast oscillation with bursts
            trace = 0.2 + 0.3 * np.sin(2 * np.pi * 3 * time)
            # Add random bursts
            for _ in range(3):
                burst_start = np.random.randint(0, n_frames - 15)
                trace[burst_start : burst_start + 15] += 0.5 * np.exp(
                    -np.arange(15) / 5
                )
        elif i == 2:
            # Random spikes
            trace = np.random.exponential(0.15, n_frames)
        elif i == 3:
            # Combination pattern
            trace = (
                0.25
                + 0.2 * np.sin(2 * np.pi * 0.8 * time)
                + 0.1 * np.random.normal(0, 1, n_frames)
            )
        else:
            # Decaying activity
            trace = 0.6 * np.exp(-time / 2) + 0.1 * np.random.normal(0, 1, n_frames)

        # Ensure non-negative and clip extreme values
        trace = np.maximum(trace, 0)
        trace = np.minimum(trace, 1.0)
        source_traces.append(trace)

    # Generate video with basic settings
    output_path = video.quick_synaptic_video(
        base_image_data=base_image_data,
        source_traces=source_traces,
        source_masks=source_masks,
        output_path="example_basic_synaptic_activity.mp4",
        fs=fs,
    )

    print("✓ Basic video generated:", output_path)
    return base_image_data, source_traces, source_masks


def example_high_quality():
    """Example with high-quality settings for publication."""

    print("\nExample 2: High-Quality Video")
    print("-" * 30)

    # Use the same data from basic example
    base_image_data, source_traces, source_masks = example_basic_usage()

    # Generate high-quality video
    output_path = video.generate_synaptic_video(
        base_image_data=base_image_data,
        source_traces=source_traces,
        source_masks=source_masks,
        fs=30.0,
        output_path="example_high_quality_synaptic_activity.mp4",
        # High-quality red overlay settings
        red_alpha_max=0.9,
        red_alpha_min=0.05,
        red_intensity_max=0.85,
        red_intensity_min=0.1,
        trace_percentile_clip=98.0,
        # High-quality base image settings
        base_vmin_percentile=0.5,
        base_vmax_percentile=99.5,
        # High video quality
        video_codec="mp4v",
        video_quality=95,
        # Performance settings
        chunk_size=30,
    )

    print("✓ High-quality video generated:", output_path)


def example_fast_preview():
    """Example optimized for fast generation and small file size."""

    print("\nExample 3: Fast Preview Video")
    print("-" * 30)

    # Create smaller data for faster processing
    height, width = 128, 128
    base_image_data = np.random.normal(0.4, 0.1, (height, width))

    # Simple structures
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    central_region = ((x - center_x) ** 2 + (y - center_y) ** 2) < 20**2
    base_image_data[central_region] = 0.8

    # Set background to NaN
    base_image_data[base_image_data < 0.25] = np.nan

    # Two simple sources
    source_masks = []
    for sx, sy in [(40, 40), (90, 90)]:
        mask = np.zeros((height, width), dtype=bool)
        synapse_region = ((x - sx) ** 2 + (y - sy) ** 2) < 5**2
        mask[synapse_region] = True
        source_masks.append(mask)

    # Short traces for fast generation (2 seconds at 15 fps)
    fs = 15.0
    n_frames = 30
    time = np.linspace(0, 2, n_frames)

    source_traces = [
        0.5 + 0.3 * np.sin(2 * np.pi * 1 * time),  # Source 1
        0.4 + 0.4 * np.sin(2 * np.pi * 0.5 * time),  # Source 2
    ]

    # Generate fast preview video
    output_path = video.generate_synaptic_video(
        base_image_data=base_image_data,
        source_traces=source_traces,
        source_masks=source_masks,
        fs=fs,
        output_path="example_fast_preview.mp4",
        # Fast settings
        red_alpha_max=0.7,
        video_quality=60,
        chunk_size=100,
        use_parallel=True,
    )

    print("✓ Fast preview video generated:", output_path)


def example_custom_appearance():
    """Example with custom red overlay appearance."""

    print("\nExample 4: Custom Red Overlay Appearance")
    print("-" * 30)

    # Use the same data structure but with custom overlay settings
    base_image_data, source_traces, source_masks = example_basic_usage()

    # Generate video with custom red appearance
    output_path = video.generate_synaptic_video(
        base_image_data=base_image_data,
        source_traces=source_traces,
        source_masks=source_masks,
        fs=20.0,
        output_path="example_custom_red_overlay.mp4",
        # Custom red overlay - more subtle
        red_alpha_max=0.6,
        red_alpha_min=0.2,  # Never fully transparent
        red_intensity_max=0.7,  # Not fully saturated red
        red_intensity_min=0.3,  # Always some red visible
        trace_percentile_clip=95.0,  # More sensitive to activity changes
        # Custom base image contrast
        base_vmin_percentile=5.0,
        base_vmax_percentile=95.0,
        video_quality=85,
    )

    print("✓ Custom appearance video generated:", output_path)


if __name__ == "__main__":
    print("Synaptic Video Generation Examples")
    print("=" * 50)

    # Run all examples
    try:
        example_basic_usage()
        example_high_quality()
        example_fast_preview()
        example_custom_appearance()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nGenerated videos:")
        print("1. example_basic_synaptic_activity.mp4 - Basic usage")
        print("2. example_high_quality_synaptic_activity.mp4 - High quality")
        print("3. example_fast_preview.mp4 - Fast generation")
        print("4. example_custom_red_overlay.mp4 - Custom appearance")
        print("\nThese examples demonstrate the full range of capabilities!")

    except Exception as e:
        print("Error running examples:", e)
        import traceback

        traceback.print_exc()
