#!/usr/bin/env python3
"""
Recover two float32 1-D datasets ('electrophysiology' & 'slap2_acquiring_trigger')
from a corrupted HDF5 file using a healthy GOOD file as a template.
- No compression assumed (learned from GOOD).
- BAD opens cleanly -> copy datasets verbatim.
- BAD doesn't open -> carve chunk payloads directly from disk:
    * infer dominant chunk-byte alignment (modulo CHUNK_BYTES)
    * classify each aligned chunk as ephys or trigger using stats learned from GOOD
    * build /raw streams (maximally recovered)
    * build /aligned streams by pairing nearest-in-file-order chunks (one ephys + one trigger)
      so sample-N corresponds across the two outputs (subset only, discarding unpaired chunks).
"""

import argparse
import mmap
from pathlib import Path

import h5py
import numpy as np


def learn_from_good(good_path):
    with h5py.File(good_path, "r") as fg:
        names = list(fg.keys())
        e_name = next(n for n in names if "trigger" not in n.lower())
        t_name = next(n for n in names if "trigger" in n.lower())
        e = fg[e_name]
        t = fg[t_name]
        if e.dtype.str not in ("<f4", "|f4"):  # little-endian float32
            raise RuntimeError(f"Expected float32 in GOOD; got {e.dtype}")
        if not e.chunks or len(e.chunks) != 1:
            raise RuntimeError(
                f"Expected 1-D chunked dataset in GOOD; got chunks={e.chunks}"
            )
        chunk = int(e.chunks[0])
        chunk_bytes = 4 * chunk
        fs = float(fg.attrs.get("samplerate", 5000.0))
        # trigger bands from percentiles
        ts = t[: min(len(t), chunk * 50)]
        low = float(np.percentile(ts, 5))
        high = float(np.percentile(ts, 95))
        spread = max(1e-6, high - low)
        tol = max(0.02 * spread, 0.05)  # ~0.1 for a 0..5V trigger
        # ephys variability threshold
        es = e[: min(len(e), chunk * 50)]
        if es.size >= chunk:
            K = es.size // chunk
            e_std_med = float(
                np.median([np.std(es[i * chunk : (i + 1) * chunk]) for i in range(K)])
            )
        else:
            e_std_med = float(np.std(es)) if es.size else 1e-4
        e_std_thr = max(1e-7, 0.02 * e_std_med)  # 2% of median
    return dict(
        e_name=e_name,
        t_name=t_name,
        fs=fs,
        chunk=chunk,
        chunk_bytes=chunk_bytes,
        trig_low=low,
        trig_high=high,
        trig_tol=tol,
        e_std_thr=e_std_thr,
    )


def try_clean_copy(bad_path, out_path, meta, report):
    """If BAD opens, copy datasets verbatim."""
    try:
        with h5py.File(bad_path, "r") as fb, h5py.File(out_path, "w") as fo:
            fo.attrs["samplerate"] = meta["fs"]
            fo.attrs["chunk_size"] = meta["chunk"]
            fo.attrs["recovery_method"] = "clean_copy"
            for k, v in fb.attrs.items():
                try:
                    fo.attrs[k] = v
                except:
                    pass
            for ds in (meta["e_name"], meta["t_name"]):
                if ds in fb:
                    d = fb[ds]
                    fo.create_dataset(ds, data=d[...], chunks=d.chunks, dtype=d.dtype)
                    for ak, av in d.attrs.items():
                        try:
                            fo[ds].attrs[ak] = av
                        except:
                            pass
        report.append("BAD opened with h5py; copied datasets intact.")
        return True
    except Exception as e:
        report.append(f"Cannot open BAD with h5py (will carve): {e}")
        return False


def carve_bad(
    bad_path,
    meta,
    report,
    coarse_step=4096,
    mod_grid_step=512,
    scan_limit_bytes=64_000_000,
):
    """Return dict with raw arrays and paired aligned arrays; plus stats."""
    # mmap + memoryview so we can scan quickly
    with open(bad_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mv = memoryview(mm)
    N = len(mm)

    CHUNK = meta["chunk"]
    CHUNK_BYTES = meta["chunk_bytes"]
    DT = np.dtype("<f4")
    low, high, tol = meta["trig_low"], meta["trig_high"], meta["trig_tol"]
    e_std_thr = meta["e_std_thr"]

    def arr_at(off, copy=False):
        a = np.frombuffer(mv, dtype=DT, count=CHUNK, offset=off)
        return a.copy() if copy else a

    def looks_like_trigger(a):
        if a.size != CHUNK or not np.isfinite(a).all():
            return False
        near_low = np.mean(np.abs(a - low) <= tol)
        near_high = np.mean(np.abs(a - high) <= tol)
        # accept if almost all samples lie near either band
        return (near_low + near_high) > 0.95 or near_low > 0.95 or near_high > 0.95

    def looks_like_ephys(a):
        if a.size != CHUNK or not np.isfinite(a).all():
            return False
        return float(a.std()) > e_std_thr and np.mean(np.abs(a - high) <= tol) < 0.02

    # Pass 1: vote best modulo alignment
    scan_bytes = min(max(0, N - CHUNK_BYTES), scan_limit_bytes)
    mods = []
    for mod in range(0, CHUNK_BYTES, mod_grid_step):
        votes = 0
        for off in range(mod, scan_bytes, coarse_step):
            a = arr_at(off, copy=False)
            if not np.isfinite(a).all():
                continue
            if looks_like_trigger(a) or looks_like_ephys(a):
                votes += 1
        mods.append((votes, mod))
    mods.sort(reverse=True)
    base_mod = mods[0][1] if mods else 0
    report.append(
        f"Inferred base alignment: {base_mod} bytes (votes {mods[0][0] if mods else 0})"
    )

    # Pass 2: classify all aligned slots
    offsets = list(range(base_mod, max(0, N - CHUNK_BYTES + 1), CHUNK_BYTES))
    classified = []  # list of (off, label), label in {"ephys","trigger","bad"}
    for off in offsets:
        a = arr_at(off, copy=False)
        if not np.isfinite(a).all():
            classified.append((off, "bad"))
        elif looks_like_trigger(a):
            classified.append((off, "trigger"))
        elif looks_like_ephys(a):
            classified.append((off, "ephys"))
        else:
            classified.append((off, "bad"))

    # Build raw streams (max keep) and also build aligned pairs by nearest-in-file-order pairing
    raw_e = []
    raw_t = []
    pairs_e = []
    pairs_t = []
    # queues for pairing
    Qe = []
    Qt = []

    for off, lab in classified:
        if lab == "ephys":
            # always keep raw (copy so mv can be released later)
            raw_e.append(arr_at(off, copy=True))
            # pairing
            if Qt:
                t_chunk = Qt.pop(0)
                e_chunk = arr_at(off, copy=True)
                pairs_e.append(e_chunk)
                pairs_t.append(t_chunk)
            else:
                Qe.append(arr_at(off, copy=True))
        elif lab == "trigger":
            raw_t.append(arr_at(off, copy=True))
            if Qe:
                e_chunk = Qe.pop(0)
                t_chunk = arr_at(off, copy=True)
                pairs_e.append(e_chunk)
                pairs_t.append(t_chunk)
            else:
                Qt.append(arr_at(off, copy=True))
        # 'bad' -> skip

    # Release the exported pointer BEFORE closing the mmap to avoid BufferError
    try:
        mv.release()
    except Exception:
        pass
    del mv
    mm.close()

    raw_e = np.concatenate(raw_e) if raw_e else np.array([], dtype=DT)
    raw_t = np.concatenate(raw_t) if raw_t else np.array([], dtype=DT)
    ali_e = np.concatenate(pairs_e) if pairs_e else np.array([], dtype=DT)
    ali_t = np.concatenate(pairs_t) if pairs_t else np.array([], dtype=DT)

    stats = dict(
        base_mod=base_mod,
        total_slots=len(offsets),
        chunks_e=sum(1 for _, l in classified if l == "ephys"),
        chunks_t=sum(1 for _, l in classified if l == "trigger"),
        chunks_bad=sum(1 for _, l in classified if l == "bad"),
        raw_e_samples=int(raw_e.size),
        raw_t_samples=int(raw_t.size),
        aligned_chunks=int(len(pairs_e)),
        aligned_samples=int(ali_e.size),
    )
    return raw_e, raw_t, ali_e, ali_t, stats


def write_output(out_path, meta, raw_e, raw_t, ali_e, ali_t, stats, method, report):
    with h5py.File(out_path, "w") as fo:
        fo.attrs["samplerate"] = meta["fs"]
        fo.attrs["chunk_size"] = meta["chunk"]
        fo.attrs["recovery_method"] = method
        for k, v in stats.items():
            fo.attrs[f"recovery_{k}"] = v
        # raw
        grp_raw = fo.create_group("raw")
        if raw_e.size:
            grp_raw.create_dataset(
                meta["e_name"], data=raw_e, chunks=(meta["chunk"],), dtype="<f4"
            )
        if raw_t.size:
            grp_raw.create_dataset(
                meta["t_name"], data=raw_t, chunks=(meta["chunk"],), dtype="<f4"
            )
        # aligned (paired)
        grp_al = fo.create_group("aligned")
        if ali_e.size and ali_t.size:
            grp_al.create_dataset(
                meta["e_name"], data=ali_e, chunks=(meta["chunk"],), dtype="<f4"
            )
            grp_al.create_dataset(
                meta["t_name"], data=ali_t, chunks=(meta["chunk"],), dtype="<f4"
            )

    # human-readable report
    rep_path = Path(str(out_path) + "_report.txt")
    lines = []
    lines.append(f"Output: {out_path}")
    lines.append(f"Recovery method: {method}")
    lines.append(
        f"Samplerate: {meta['fs']} Hz  Chunk: {meta['chunk']} samples ({meta['chunk_bytes']} bytes)"
    )
    lines.append(
        f"Trigger bands learned from GOOD: low≈{meta['trig_low']:.4f}, high≈{meta['trig_high']:.4f}, tol≈{meta['trig_tol']:.4f}"
    )
    lines.append(f"Ephys std threshold: {meta['e_std_thr']:.6g}")
    for r in report:
        lines.append(r)
    if method == "carve":
        lines += [
            f"total_slots={stats['total_slots']}, chunks_e={stats['chunks_e']}, chunks_t={stats['chunks_t']}, chunks_bad={stats['chunks_bad']}",
            f"raw_e_samples={stats['raw_e_samples']}, raw_t_samples={stats['raw_t_samples']}",
            f"aligned_chunks={stats['aligned_chunks']}, aligned_samples={stats['aligned_samples']}",
            "NOTE: '/aligned' datasets are guaranteed aligned (paired by nearest-in-file-order chunk);",
            "      '/raw' datasets are maximal recoveries, not guaranteed to match lengths.",
        ]
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    return rep_path


def main():
    ap = argparse.ArgumentParser(
        description="Recover aligned electrophysiology & trigger from a corrupted HDF5."
    )
    ap.add_argument(
        "--good", required=True, help="Path to healthy GOOD file (same acquisition)."
    )
    ap.add_argument("--bad", required=True, help="Path to BAD/corrupted file.")
    ap.add_argument("--out", required=True, help="Output HDF5 path.")
    ap.add_argument(
        "--coarse-step",
        type=int,
        default=4096,
        help="Coarse step (bytes) for modulo voting.",
    )
    ap.add_argument(
        "--mod-grid-step",
        type=int,
        default=512,
        help="Modulo grid step (bytes) in [0, CHUNK_BYTES).",
    )
    ap.add_argument(
        "--scan-limit-bytes",
        type=int,
        default=64_000_000,
        help="Max bytes scanned during modulo voting.",
    )
    args = ap.parse_args()

    good = Path(args.good)
    bad = Path(args.bad)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = []
    meta = learn_from_good(good)
    report.append(
        f"GOOD learned: e='{meta['e_name']}', t='{meta['t_name']}', fs={meta['fs']} Hz, chunk={meta['chunk']}"
    )
    # clean-copy if possible
    if try_clean_copy(bad, out, meta, report):
        stats = dict()
        rep = write_output(
            out,
            meta,
            raw_e=np.array([], dtype="<f4"),
            raw_t=np.array([], dtype="<f4"),
            ali_e=np.array([], dtype="<f4"),
            ali_t=np.array([], dtype="<f4"),
            stats=stats,
            method="clean_copy",
            report=report,
        )
        print(f"Recovered OK (clean copy).\nOut: {out}\nReport: {rep}")
        return

    # carve path
    raw_e, raw_t, ali_e, ali_t, stats = carve_bad(
        bad,
        meta,
        report,
        coarse_step=args.coarse_step,
        mod_grid_step=args.mod_grid_step,
        scan_limit_bytes=args.scan_limit_bytes,
    )
    method = "carve"
    rep = write_output(out, meta, raw_e, raw_t, ali_e, ali_t, stats, method, report)
    print(f"Recovered OK (carved).\nOut: {out}\nReport: {rep}")
    if ali_e.size == 0 or ali_t.size == 0:
        print("WARNING: No aligned pairs could be formed. See report for details.")


if __name__ == "__main__":
    main()
