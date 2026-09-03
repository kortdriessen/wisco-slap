from __future__ import annotations

import numpy as np

from slap2_py.utils.sync_recovery import CHUNK, CHUNK_BYTES, Template, carve

FS = 5000.0


def _write_fake_bad(
    path, chunks_s, chunks_e, order, gaps_after, first_offset=6096
) -> None:
    """Write a fake corrupt DataRec file: junk header, chunks in ``order`` with
    zero-filled metadata blocks of ``gaps_after[pos]`` bytes after chunk ``pos``."""
    with open(path, "wb") as f:
        f.write(np.random.default_rng(1).bytes(first_offset))
        i_s = i_e = 0
        for pos, tag in enumerate(order):
            if tag == "S":
                f.write(chunks_s[i_s].astype("<f4").tobytes())
                i_s += 1
            else:
                f.write(chunks_e[i_e].astype("<f4").tobytes())
                i_e += 1
            f.write(b"\x00" * gaps_after.get(pos, 0))


def test_carve_reassembles_channels_across_metadata_gaps_and_order_changes(
    tmp_path,
) -> None:
    rng = np.random.default_rng(0)
    n = 12  # chunks per channel
    # trigger: 0 V baseline with ~1 mV noise, then 5 V with 1-ms dips every 15 ms
    trig = np.full(n * CHUNK, 0.001) + rng.normal(0, 0.0005, n * CHUNK)
    hi = np.full(6 * CHUNK, 5.05) + rng.normal(0, 0.002, 6 * CHUNK)
    hi.reshape(-1, 75)[:, :5] = 0.6
    trig[5 * CHUNK :] = (
        hi[: 7 * CHUNK] if False else np.concatenate([hi, hi])[: 7 * CHUNK]
    )
    # ephys copy: 4 mV baseline, step to 0.15 V + slow physiological wander
    eph = np.full(n * CHUNK, 0.004) + rng.normal(0, 0.0005, n * CHUNK)
    eph[2 * CHUNK + 1234 :] = (
        0.15
        + 0.02 * np.sin(np.arange(eph.size - 2 * CHUNK - 1234) / 4e4)
        + rng.normal(0, 0.001, eph.size - 2 * CHUNK - 1234)
    )
    chunks_s = [trig[i * CHUNK : (i + 1) * CHUNK] for i in range(n)]
    chunks_e = [eph[i * CHUNK : (i + 1) * CHUNK] for i in range(n)]
    # alternate for a while, then the order changes (S S E E ...) like a real file
    order = list("SESESESESESE") + list("SSEESSEESSEE")
    gaps = {3: 4192, 4: 4192, 9: 2096, 10: 2096, 17: 2096, 18: 2096}
    bad = tmp_path / "bad.h5"
    _write_fake_bad(bad, chunks_s, chunks_e, order, gaps)
    tpl = Template(
        first_offset=6096,
        order=list("SE" * n),
        extra_after={3: 4192, 4: 4192, 9: 2096, 10: 2096},
        fs=FS,
        attrs={},
    )

    rec = carve(str(bad), tpl)
    assert len(rec.chunks["S"]) == n and len(rec.chunks["E"]) == n
    assert rec.corrections == []
    np.testing.assert_array_equal(np.concatenate(rec.chunks["S"]), trig.astype("<f4"))
    np.testing.assert_array_equal(np.concatenate(rec.chunks["E"]), eph.astype("<f4"))
    assert "end of file" in rec.stop_reason or "no plausible chunk" in rec.stop_reason


def test_carve_stops_at_unflushed_tail(tmp_path) -> None:
    rng = np.random.default_rng(2)
    n = 4
    trig = np.full(n * CHUNK, 0.001) + rng.normal(0, 0.0005, n * CHUNK)
    eph = np.full(n * CHUNK, 0.004) + rng.normal(0, 0.0005, n * CHUNK)
    chunks_s = [trig[i * CHUNK : (i + 1) * CHUNK] for i in range(n)]
    chunks_e = [eph[i * CHUNK : (i + 1) * CHUNK] for i in range(n)]
    bad = tmp_path / "bad.h5"
    _write_fake_bad(bad, chunks_s, chunks_e, list("SE" * n), {})
    with open(bad, "ab") as f:  # trailing garbage shorter than a chunk
        f.write(b"\xff" * 1000)
    tpl = Template(
        first_offset=6096, order=list("SE" * n), extra_after={}, fs=FS, attrs={}
    )
    rec = carve(str(bad), tpl)
    assert len(rec.chunks["S"]) == n and len(rec.chunks["E"]) == n
    assert (n * 2) * CHUNK_BYTES + 6096 + 1000 == bad.stat().st_size
