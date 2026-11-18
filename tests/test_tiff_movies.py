"""Trial clocks, bounded smoothing and pixel-exact movie encoding."""

from pathlib import Path

import av
import numpy as np
import pytest
import tifffile
from scipy.ndimage import uniform_filter1d

from wisco_slap.viz import tiff_movies as m


def fake_plan(
    tmp_path: Path, raw: np.ndarray, split: int = 9, gap: bool = False
) -> m.FramePlan:
    n, h, w = raw.shape
    paths = []
    for index, piece in enumerate((raw[:split], raw[split:])):
        path = tmp_path / f"trial{index}.tif"
        pages = np.stack((piece, piece * 2), axis=1).reshape(-1, h, w)
        tifffile.imwrite(path, pages, photometric="minisblack")
        paths.append(str(path))
    times = np.arange(n) / 80 + 100
    if gap:
        times[split:] += 10
    trials = [
        m.Trial(paths[0], 1, 1, 0, split, split),
        m.Trial(paths[1], 2, 2 if gap else 1, split, n, n - split),
    ]
    return m.FramePlan(
        1,
        trials,
        times,
        [split, n - split],
        (h, w),
        2,
        80,
        np.array([0, split, n] if gap else [0, n]),
    )


def test_chunks_match_whole_trace_and_preserve_trial_halos(tmp_path):
    rng = np.random.default_rng(12)
    raw = rng.uniform(0, 50, (23, 5, 7)).astype("f4")
    raw[:, 0, 0] = np.nan
    plan = fake_plan(tmp_path, raw)
    expected = uniform_filter1d(raw, size=6, axis=0, mode="nearest")
    for start, stop in [(0, 4), (4, 11), (11, 23)]:
        actual, coords = m._packed_smooth(plan, 0, start, stop, 6)
        np.testing.assert_allclose(
            actual, expected[start:stop].reshape(stop - start, -1)[:, coords], atol=1e-5
        )
    np.testing.assert_array_equal(m._read(plan, 1, 7, 12), raw[7:12] * 2)


def test_missing_pixels_do_not_poison_later_frames(tmp_path):
    raw = np.full((23, 3, 5), 12.0, dtype="f4")
    raw[7, 1, 2] = np.nan
    plan = fake_plan(tmp_path, raw)
    values, coords = m._packed_smooth(plan, 0, 0, 23, 6)
    actual = values[:, np.flatnonzero(coords == 7)[0]]
    assert np.isnan(actual[7])
    np.testing.assert_allclose(np.delete(actual, 7), 12.0)


def test_no_smoothing_across_epoch_gap(tmp_path):
    raw = np.zeros((23, 3, 5), dtype="f4")
    raw[9:] = 100
    plan = fake_plan(tmp_path, raw, gap=True)
    a, _ = m._packed_smooth(plan, 0, 6, 9, 6)
    b, _ = m._packed_smooth(plan, 0, 9, 12, 6)
    assert np.all(a == 0) and np.all(b == 100)
    with pytest.raises(ValueError, match="epoch"):
        m._packed_smooth(plan, 0, 8, 11, 6)


@pytest.mark.parametrize("name", ["hot", "viridis", "tab10", None])
def test_palette_matches_matplotlib(name):
    from matplotlib import colormaps

    values = np.array([np.nan, -1, 0, 0.01, 0.1, 0.5, 0.999, 1, 2], dtype="f4")
    rgb = m._render_packed(
        values,
        np.arange(len(values)),
        (1, len(values)),
        0,
        1,
        m._palette(name),
        0,
        name is None,
    )[0]
    scaled = np.clip(np.nan_to_num(values), 0, 1)
    if name is None:
        expected = np.repeat((scaled * 255 + 0.5).astype("u1")[:, None], 3, axis=1)
    else:
        expected = colormaps[name](scaled, bytes=True)[:, :3]
    expected[~np.isfinite(values)] = 0
    np.testing.assert_array_equal(rgb, expected)


def test_lossless_encoding_and_concat_keep_every_pixel_and_frame(tmp_path):
    rng = np.random.default_rng(9)
    raw = rng.uniform(0, 1, (23, 5, 7)).astype("f4")
    raw[:, 0, 0] = np.nan
    plan = fake_plan(tmp_path, raw)
    palette = m._palette("viridis")
    paths = []
    for i, (start, stop) in enumerate([(0, 10), (10, 23)]):
        path = tmp_path / f"segment{i}.mp4"
        paths.append(str(path))
        m._encode_segment(
            str(path),
            plan,
            0,
            start,
            stop,
            1,
            (0, 1),
            palette,
            1,
            80,
            "libx264rgb",
            0,
            "veryfast",
            1,
            False,
        )
    output = tmp_path / "joined.mp4"
    m._concat(paths, output)
    with av.open(str(output)) as container:
        actual = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    assert len(actual) == 23
    for i, frame in enumerate(actual):
        v, c = m._packed_smooth(plan, 0, i, i + 1, 1)
        expected = m._render_packed(v[0], c, plan.shape, 0, 1, palette, 1)
        np.testing.assert_array_equal(frame, expected)


def test_make_preserves_gap_clock_and_selected_frame_identity(tmp_path, monkeypatch):
    raw = np.random.default_rng(9).uniform(0, 1, (23, 5, 7)).astype("f4")
    plan = fake_plan(tmp_path, raw, gap=True)
    provenance = dict(
        schema_version=1,
        timebase="ephys",
        acquisition=dict(subject="test", exp="exp_1", loc="loc_A", acq="acq_1"),
        sync_block=1,
        clock_ratio=1.0,
        ephys_offset_s=100.0,
        channel_schema=dict(signalByRawChannel=["calcium", "glutamate"]),
    )
    monkeypatch.setattr(m, "plan_acquisition", lambda *args: ([plan], provenance))
    output = tmp_path / "movies"
    monkeypatch.setattr(m, "default_movie_dir", lambda *args: output)
    manifest = m.make(
        "test",
        "exp_1",
        "loc_A",
        "acq_1",
        t1=plan.times[2],
        t2=plan.times[-2],
        smoothing_s=0.05,
        cmaps=("hot", "viridis"),
        workers=2,
        chunk_frames=5,
        encoder_threads=1,
        progress=False,
    )
    assert manifest["timebase"] == "ephys"
    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.is_file()
    assert manifest_path.parent.parent == output / "clips"
    assert manifest["lossless_display_encoding"]
    assert manifest["path_convention"] == "relative_to_manifest"
    # Moving the entire export leaves all movie/timestamp references usable.
    relocated = tmp_path / "relocated_export"
    manifest_path.parent.rename(relocated)
    for video in manifest["videos"]:
        assert not Path(video["video_path"]).is_absolute()
        assert not Path(video["frame_times_path"]).is_absolute()
        np.testing.assert_array_equal(
            np.load(relocated / video["frame_times_path"]), plan.times[2:-2]
        )
        assert video["n_frames"] == 19
        assert video["gaps_after_frame"] == [6]
        with av.open(str(relocated / video["video_path"])) as container:
            assert len(list(container.decode(video=0))) == 19
    with pytest.raises(FileExistsError):
        m.make("test", "exp_1", "loc_A", "acq_1", out_dir=relocated, progress=False)
