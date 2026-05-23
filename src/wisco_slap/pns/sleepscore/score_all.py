"""Batch scoring entry point: score individual sessions or all subjects."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS

from ._config import VIDEO_QUALITY_FEATURES
from ._data import create_session, load_scoring_times
from ._model import load_model, score_session

DEFAULT_PRODUCTION_MODEL_DIR = (
    Path(DEFS.anmat_root) / "autoscore_model" / "production_model"
)
MODEL_SCORED_DIRNAME = "model_scored"
MODEL_ARCHIVE_DIRNAME = "MODEL_ARCHIVED"
MODEL_VERSION_FILENAME = "model_version.txt"
EPOCH_DF_FILENAME = "epoch_df.parquet"
BOUT_DF_FILENAME = "bout_df.parquet"
EPOCH_HYPNO_FILENAME = "epoch_hypno.csv"
BOUT_HYPNO_FILENAME = "bout_hypno.csv"
REQUIRED_SYNC_BLOCK_DATA_REL_PATHS = (
    ("pupil", "eye_metrics", "eye_metrics.parquet"),
    ("whisking", "whisk_df.parquet"),
)


@dataclass(frozen=True)
class ScoreSyncBlockResult:
    """Summary of a sync-block scoring attempt."""

    subject: str
    exp: str
    sync_block: int
    status: str
    model_name: str
    model_path: Path
    hypnogram_dir: Path
    model_scored_dir: Path
    epoch_path: Path
    bout_path: Path
    epoch_hypno_path: Path
    bout_hypno_path: Path
    archived_dir: Path | None = None
    reason: str | None = None


def _sync_block_hypnogram_dir(subject: str, exp: str, sync_block: int) -> Path:
    return (
        Path(DEFS.anmat_root)
        / subject
        / exp
        / "sync_block_data"
        / f"sync_block-{sync_block}"
        / "hypnograms"
    )


def _resolve_production_model_path(production_model_dir: str | Path) -> Path:
    """Return the active production model path.

    If a directory contains multiple ``*.pkl`` files, the newest file by
    modification time is treated as the active production model.
    """
    path = Path(production_model_dir)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Production model directory does not exist: {path}")

    candidates = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix == ".pkl"],
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No .pkl model files found in {path}")
    return candidates[-1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_signature(model_path: Path) -> dict[str, str]:
    stat = model_path.stat()
    return {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "model_sha256": _sha256_file(model_path),
        "model_size_bytes": str(stat.st_size),
        "model_mtime_ns": str(stat.st_mtime_ns),
    }


def _read_model_version(model_scored_dir: Path) -> dict[str, str] | None:
    """Read model-version metadata from a scored hypnogram directory.

    Supports the current key-value text format and a legacy one-line format
    containing only the model filename.
    """
    for filename in (MODEL_VERSION_FILENAME, "model_version"):
        path = model_scored_dir / filename
        if not path.is_file():
            continue
        text = path.read_text().strip()
        if not text:
            return None

        metadata: dict[str, str] = {}
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if "=" not in line:
                if "model_name" not in metadata:
                    metadata["model_name"] = line
                continue
            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()
        return metadata or None
    return None


def _same_model(
    existing: dict[str, str] | None,
    current: dict[str, str],
    scoring_signature: dict[str, str] | None = None,
) -> bool:
    if existing is None:
        return False
    if existing.get("model_name") != current["model_name"]:
        return False

    # Old model_version files may only have the filename. Treat filename match
    # as enough for those, but use the hash whenever it is available.
    existing_hash = existing.get("model_sha256")
    if existing_hash is not None and existing_hash != current["model_sha256"]:
        return False

    if scoring_signature is not None:
        for key, value in scoring_signature.items():
            if existing.get(key) != value:
                return False

    return True


def _resolve_wake_insert_threshold(add_wake_epochs: bool | float) -> float | None:
    """Return the Wake-insertion threshold, or None if disabled."""
    if add_wake_epochs is False:
        return None
    if add_wake_epochs is True:
        return 0.8

    threshold = float(add_wake_epochs)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "add_wake_epochs must be False or a probability threshold between 0 and 1"
        )
    return threshold


def _scoring_signature(add_wake_epochs: bool | float) -> dict[str, str]:
    threshold = _resolve_wake_insert_threshold(add_wake_epochs)
    return {
        "add_wake_epochs": "false" if threshold is None else f"{threshold:.12g}",
    }


def _write_model_version(
    model_scored_dir: Path,
    model_signature: dict[str, str],
    scoring_signature: dict[str, str] | None = None,
) -> None:
    scored_at = datetime.now(UTC).isoformat()
    lines = [
        f"model_name={model_signature['model_name']}",
        f"model_sha256={model_signature['model_sha256']}",
        f"model_path={model_signature['model_path']}",
        f"model_size_bytes={model_signature['model_size_bytes']}",
        f"model_mtime_ns={model_signature['model_mtime_ns']}",
        f"scored_at_utc={scored_at}",
    ]
    if scoring_signature is not None:
        lines.extend(f"{key}={value}" for key, value in scoring_signature.items())
    (model_scored_dir / MODEL_VERSION_FILENAME).write_text("\n".join(lines) + "\n")


def _sanitize_path_part(value: str | None) -> str:
    if not value:
        return "unknown_model"
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    clean = clean.strip("._")
    return clean or "unknown_model"


def _archive_model_scored_dir(
    hypnogram_dir: Path,
    model_scored_dir: Path,
    existing_signature: dict[str, str] | None,
) -> Path:
    archive_root = hypnogram_dir / MODEL_ARCHIVE_DIRNAME
    archive_root.mkdir(parents=True, exist_ok=True)

    old_model = _sanitize_path_part(
        existing_signature.get("model_name") if existing_signature else None
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dest = archive_root / f"{MODEL_SCORED_DIRNAME}__{old_model}__{timestamp}"
    dest = base_dest
    suffix = 2
    while dest.exists():
        dest = Path(f"{base_dest}__{suffix}")
        suffix += 1

    shutil.move(str(model_scored_dir), str(dest))
    return dest


def _merge_consecutive_label_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Merge adjacent interval rows that have the same label."""
    if df.is_empty():
        return pl.DataFrame(
            schema={"start_s": pl.Float64, "end_s": pl.Float64, "label": pl.String}
        )

    return (
        df.select(["start_s", "end_s", "label"])
        .sort("start_s")
        .with_columns(
            pl.col("label")
            .ne(pl.col("label").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_group")
        )
        .group_by("_group", maintain_order=True)
        .agg(
            pl.col("start_s").first().alias("start_s"),
            pl.col("end_s").last().alias("end_s"),
            pl.col("label").first().alias("label"),
        )
        .select(["start_s", "end_s", "label"])
    )


def _bout_df_to_hypno(bout_df: pl.DataFrame) -> pl.DataFrame:
    """Return a Loupe-ready hypnogram from HSMM bout predictions."""
    required = {"start_s", "end_s", "label"}
    missing = sorted(required - set(bout_df.columns))
    if missing:
        raise ValueError(f"bout_df is missing hypnogram columns: {missing}")
    return _merge_consecutive_label_rows(bout_df.select(["start_s", "end_s", "label"]))


def _epoch_df_to_argmax_hypno(epoch_df: pl.DataFrame) -> pl.DataFrame:
    """Return a Loupe-ready hypnogram from raw per-epoch classifier argmax."""
    required = {"start_s", "end_s"}
    missing = sorted(required - set(epoch_df.columns))
    if missing:
        raise ValueError(f"epoch_df is missing hypnogram columns: {missing}")

    prob_cols = [col for col in epoch_df.columns if col.startswith("P_")]
    if not prob_cols:
        raise ValueError("epoch_df does not contain any P_<state> probability columns")
    state_names = [col.removeprefix("P_") for col in prob_cols]

    probs = epoch_df.select(prob_cols).to_numpy()
    argmax_idx = probs.argmax(axis=1)
    labels = [state_names[int(idx)] for idx in argmax_idx]

    return _merge_consecutive_label_rows(
        pl.DataFrame(
            {
                "start_s": epoch_df["start_s"].to_list(),
                "end_s": epoch_df["end_s"].to_list(),
                "label": labels,
            }
        )
    )


def _rebuild_bout_df_from_epoch_labels(
    epoch_df: pl.DataFrame,
    label_col: str,
) -> pl.DataFrame:
    """Build a bout-level table from an epoch table and one label column."""
    if epoch_df.is_empty():
        return pl.DataFrame()

    required = {"session_id", "start_s", "end_s", label_col}
    missing = sorted(required - set(epoch_df.columns))
    if missing:
        raise ValueError(
            f"epoch_df is missing columns needed for bout rebuild: {missing}"
        )

    prob_aggs = [
        pl.col(col).mean().alias(f"mean_{col}")
        for col in epoch_df.columns
        if col.startswith("P_")
    ]
    quality_aggs = [
        pl.col(col).mean().alias(f"mean_{col}")
        for col in VIDEO_QUALITY_FEATURES
        if col in epoch_df.columns
    ]

    return (
        epoch_df.sort("start_s")
        .with_columns(
            pl.col(label_col)
            .ne(pl.col(label_col).shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_group")
        )
        .group_by("_group", maintain_order=True)
        .agg(
            pl.col("session_id").first().alias("session_id"),
            pl.col("start_s").first().alias("start_s"),
            pl.col("end_s").last().alias("end_s"),
            pl.col(label_col).first().alias("label"),
            pl.len().alias("n_epochs"),
            *prob_aggs,
            *quality_aggs,
        )
        .select(
            [
                "session_id",
                "start_s",
                "end_s",
                "label",
                "n_epochs",
                *[expr.meta.output_name() for expr in prob_aggs],
                *[expr.meta.output_name() for expr in quality_aggs],
            ]
        )
    )


def _insert_high_probability_wake_epochs(
    bout_df: pl.DataFrame,
    epoch_df: pl.DataFrame,
    add_wake_epochs: bool | float,
) -> pl.DataFrame:
    """Force high-confidence Wake epochs into the saved bout predictions."""
    threshold = _resolve_wake_insert_threshold(add_wake_epochs)
    if threshold is None:
        return bout_df

    required = {"pred_label", "P_Wake"}
    missing = sorted(required - set(epoch_df.columns))
    if missing:
        raise ValueError(f"epoch_df is missing Wake-insertion columns: {missing}")

    patched_epoch_df = epoch_df.with_columns(
        pl.when(pl.col("P_Wake") >= threshold)
        .then(pl.lit("Wake"))
        .otherwise(pl.col("pred_label"))
        .alias("_wake_insert_label")
    )
    return _rebuild_bout_df_from_epoch_labels(patched_epoch_df, "_wake_insert_label")


def _write_hypnogram_outputs(
    model_scored_dir: Path,
    bout_df: pl.DataFrame,
    epoch_df: pl.DataFrame,
) -> tuple[Path, Path]:
    bout_hypno_path = model_scored_dir / BOUT_HYPNO_FILENAME
    epoch_hypno_path = model_scored_dir / EPOCH_HYPNO_FILENAME

    _bout_df_to_hypno(bout_df).write_csv(bout_hypno_path)
    _epoch_df_to_argmax_hypno(epoch_df).write_csv(epoch_hypno_path)

    return bout_hypno_path, epoch_hypno_path


def _score_session_for_sync_block(
    subject: str,
    exp: str,
    sync_block: int,
    model: dict[str, Any],
    store_chans: dict[str, list[int]] | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    times = load_scoring_times(subject, exp, sync_block)
    if len(times[0]) > 1:
        raise NotImplementedError(
            "Multiple scoring times per sync block not yet supported"
        )
    t1 = times[0][0]
    t2 = times[1][0]

    session = create_session(subject, exp, sync_block, t1, t2, store_chans)
    return score_session(model, session)


def _normalize_filter(values: Iterable[str] | None) -> set[str] | None:
    if isinstance(values, str):
        return {values}
    return set(values) if values is not None else None


def _split_acq_id(acq_id: str) -> tuple[str, str]:
    parts = acq_id.split("--")
    if len(parts) != 2:
        raise ValueError(f"Invalid acquisition id in acquisition_master.yaml: {acq_id}")
    return parts[0], parts[1]


def _sync_info_get_block_entry(
    sync_blocks: dict[Any, Any],
    sync_block: int,
) -> dict[str, Any]:
    return sync_blocks.get(sync_block, sync_blocks.get(str(sync_block), {})) or {}


def _iter_acquisition_master_sync_blocks(
    *,
    subjects: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Return unique sync blocks implied by acquisition_master.yaml."""
    subject_filter = _normalize_filter(subjects)
    experiment_filter = _normalize_filter(experiments)
    acq_master = wis.meta.get.acq_master()
    sync_info = wis.meta.get.sync_info()

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for subject, exps in acq_master.items():
        if subject_filter is not None and subject not in subject_filter:
            continue
        for exp, acq_ids in exps.items():
            if experiment_filter is not None and exp not in experiment_filter:
                continue
            for acq_id in acq_ids:
                try:
                    _split_acq_id(acq_id)
                    sync_block = int(
                        sync_info[subject][exp]["acquisitions"][acq_id]["sync_block"]
                    )
                    sync_block_entry = _sync_info_get_block_entry(
                        sync_info[subject][exp].get("sync_blocks", {}),
                        sync_block,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "subject": subject,
                            "exp": exp,
                            "sync_block": None,
                            "source_acq_ids": acq_id,
                            "sync_block_corrupt": None,
                            "status": "skipped_missing_sync_info",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue

                key = (subject, exp, sync_block)
                if key in seen:
                    continue
                seen.add(key)
                block_acq_ids = []
                for candidate in acq_ids:
                    candidate_info = (
                        sync_info.get(subject, {})
                        .get(exp, {})
                        .get("acquisitions", {})
                        .get(candidate)
                    )
                    if candidate_info is None:
                        continue
                    if int(candidate_info["sync_block"]) == sync_block:
                        block_acq_ids.append(candidate)
                rows.append(
                    {
                        "subject": subject,
                        "exp": exp,
                        "sync_block": sync_block,
                        "source_acq_ids": ",".join(block_acq_ids),
                        "sync_block_corrupt": bool(
                            sync_block_entry.get("corrupt", False)
                        ),
                        "status": "candidate",
                        "reason": None,
                    }
                )
    return rows


def _required_sync_block_data_paths(
    subject: str,
    exp: str,
    sync_block: int,
) -> dict[str, Path]:
    sync_block_dir = (
        Path(DEFS.anmat_root)
        / subject
        / exp
        / "sync_block_data"
        / f"sync_block-{sync_block}"
    )
    required = {
        "raw_ephys": (
            Path(DEFS.data_root)
            / subject
            / exp
            / "ephys"
            / f"ephys-{sync_block}"
        )
    }
    for parts in REQUIRED_SYNC_BLOCK_DATA_REL_PATHS:
        required["/".join(parts)] = sync_block_dir.joinpath(*parts)
    return required


def _missing_sync_block_data(
    subject: str,
    exp: str,
    sync_block: int,
) -> list[str]:
    missing = []
    for name, path in _required_sync_block_data_paths(
        subject, exp, sync_block
    ).items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    return missing


def score_sync_block(
    subject: str,
    exp: str,
    sync_block: int,
    *,
    production_model_dir: str | Path = DEFAULT_PRODUCTION_MODEL_DIR,
    store_chans: dict[str, list[int]] | None = None,
    add_wake_epochs: bool | float = 0.8,
    force: bool = False,
    verbose: bool = True,
) -> ScoreSyncBlockResult:
    """Score one sync block with the current production model.

    Results are written under the sync block's hypnogram directory:

    ``sync_block_data/sync_block-{n}/hypnograms/model_scored/``

    If that directory already contains outputs from the same production model,
    scoring is skipped. If it contains outputs from an older model, the whole
    directory is moved into ``MODEL_ARCHIVED`` before the new outputs are saved.
    """
    model_path = _resolve_production_model_path(production_model_dir)
    current_signature = _model_signature(model_path)
    scoring_signature = _scoring_signature(add_wake_epochs)

    hypnogram_dir = _sync_block_hypnogram_dir(subject, exp, sync_block)
    model_scored_dir = hypnogram_dir / MODEL_SCORED_DIRNAME
    epoch_path = model_scored_dir / EPOCH_DF_FILENAME
    bout_path = model_scored_dir / BOUT_DF_FILENAME
    epoch_hypno_path = model_scored_dir / EPOCH_HYPNO_FILENAME
    bout_hypno_path = model_scored_dir / BOUT_HYPNO_FILENAME

    existing_signature = (
        _read_model_version(model_scored_dir) if model_scored_dir.exists() else None
    )
    outputs_exist = all(
        path.is_file()
        for path in (epoch_path, bout_path, epoch_hypno_path, bout_hypno_path)
    )
    if (
        model_scored_dir.is_dir()
        and not force
        and outputs_exist
        and _same_model(existing_signature, current_signature, scoring_signature)
    ):
        if verbose:
            print(
                f"Skipping {subject} {exp} sync_block-{sync_block}: "
                f"already scored with {current_signature['model_name']}"
            )
        return ScoreSyncBlockResult(
            subject=subject,
            exp=exp,
            sync_block=sync_block,
            status="skipped",
            model_name=current_signature["model_name"],
            model_path=model_path,
            hypnogram_dir=hypnogram_dir,
            model_scored_dir=model_scored_dir,
            epoch_path=epoch_path,
            bout_path=bout_path,
            epoch_hypno_path=epoch_hypno_path,
            bout_hypno_path=bout_hypno_path,
            reason="same_model",
        )

    model = load_model(model_path)
    bout_df, epoch_df = _score_session_for_sync_block(
        subject, exp, sync_block, model, store_chans
    )
    bout_df = _insert_high_probability_wake_epochs(
        bout_df, epoch_df, add_wake_epochs
    )

    archived_dir = None
    if model_scored_dir.exists():
        archived_dir = _archive_model_scored_dir(
            hypnogram_dir, model_scored_dir, existing_signature
        )

    model_scored_dir.mkdir(parents=True, exist_ok=True)
    epoch_df.write_parquet(epoch_path)
    bout_df.write_parquet(bout_path)
    bout_hypno_path, epoch_hypno_path = _write_hypnogram_outputs(
        model_scored_dir, bout_df, epoch_df
    )
    _write_model_version(model_scored_dir, current_signature, scoring_signature)

    if verbose:
        archive_note = (
            f" archived previous outputs to {archived_dir}" if archived_dir else ""
        )
        print(
            f"Scored {subject} {exp} sync_block-{sync_block} "
            f"with {current_signature['model_name']}.{archive_note}"
        )

    return ScoreSyncBlockResult(
        subject=subject,
        exp=exp,
        sync_block=sync_block,
        status="scored",
        model_name=current_signature["model_name"],
        model_path=model_path,
        hypnogram_dir=hypnogram_dir,
        model_scored_dir=model_scored_dir,
        epoch_path=epoch_path,
        bout_path=bout_path,
        epoch_hypno_path=epoch_hypno_path,
        bout_hypno_path=bout_hypno_path,
        archived_dir=archived_dir,
    )


def score_all_subjects(
    *,
    subjects: Iterable[str] | None = None,
    experiments: Iterable[str] | None = None,
    production_model_dir: str | Path = DEFAULT_PRODUCTION_MODEL_DIR,
    store_chans: dict[str, list[int]] | None = None,
    add_wake_epochs: bool | float = 0.8,
    force: bool = False,
    dry_run: bool = False,
    skip_corrupt_sync_blocks: bool = True,
    continue_on_error: bool = True,
    verbose: bool = True,
) -> pl.DataFrame:
    """Score every scoreable sync block implied by acquisition_master.yaml.

    The acquisition master is used as the source of truth for which
    subject/experiment/acquisition entries matter. Acquisitions are mapped to
    sync blocks via ``sync_info.yaml`` and each unique sync block is attempted
    once if its required ephys, eye-metric, and whisking data are present.
    """
    rows: list[dict[str, Any]] = []
    candidates = _iter_acquisition_master_sync_blocks(
        subjects=subjects,
        experiments=experiments,
    )

    for candidate in candidates:
        subject = candidate["subject"]
        exp = candidate["exp"]
        sync_block = candidate["sync_block"]
        base_row = {
            "subject": subject,
            "exp": exp,
            "sync_block": sync_block,
            "source_acq_ids": candidate.get("source_acq_ids"),
            "sync_block_corrupt": candidate.get("sync_block_corrupt"),
            "model_name": None,
            "model_scored_dir": None,
            "archived_dir": None,
            "missing_data": None,
            "reason": candidate.get("reason"),
        }

        if candidate["status"] != "candidate":
            rows.append({**base_row, "status": candidate["status"]})
            if verbose:
                print(f"Skipping {subject} {exp}: {candidate['reason']}")
            continue

        if sync_block is None:
            rows.append(
                {
                    **base_row,
                    "status": "skipped_missing_sync_info",
                    "reason": "sync_block was not resolved",
                }
            )
            continue

        if skip_corrupt_sync_blocks and candidate.get("sync_block_corrupt"):
            rows.append(
                {
                    **base_row,
                    "status": "skipped_corrupt_sync_block",
                    "reason": "sync_info marks this sync block as corrupt",
                }
            )
            if verbose:
                print(f"Skipping {subject} {exp} sync_block-{sync_block}: corrupt")
            continue

        missing_data = _missing_sync_block_data(subject, exp, sync_block)
        if missing_data:
            rows.append(
                {
                    **base_row,
                    "status": "skipped_missing_data",
                    "missing_data": "\n".join(missing_data),
                    "reason": "required scoring inputs are missing",
                }
            )
            if verbose:
                print(
                    f"Skipping {subject} {exp} sync_block-{sync_block}: "
                    f"{len(missing_data)} missing input(s)"
                )
            continue

        if dry_run:
            rows.append({**base_row, "status": "ready", "reason": "dry_run"})
            if verbose:
                print(f"Ready {subject} {exp} sync_block-{sync_block}")
            continue

        try:
            result = score_sync_block(
                subject,
                exp,
                sync_block,
                production_model_dir=production_model_dir,
                store_chans=store_chans,
                add_wake_epochs=add_wake_epochs,
                force=force,
                verbose=verbose,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            if verbose:
                print(
                    f"Error scoring {subject} {exp} sync_block-{sync_block}: {exc}"
                )
            continue

        rows.append(
            {
                **base_row,
                "status": result.status,
                "model_name": result.model_name,
                "model_scored_dir": str(result.model_scored_dir),
                "archived_dir": (
                    str(result.archived_dir)
                    if result.archived_dir is not None
                    else None
                ),
                "reason": result.reason,
            }
        )

    return pl.DataFrame(rows) if rows else pl.DataFrame()


def autoscore_session(
    subject: str,
    exp: str,
    sync_block: int,
    model: dict[str, Any],
    store_chans: dict[str, list[int]] | None = None,
    overwrite: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score a single session and save results.

    Parameters
    ----------
    subject, exp, sync_block : identifiers.
    model : trained model dict.
    store_chans : EEG channel mapping.
    overwrite : whether to overwrite existing results.

    Returns
    -------
    bout_df, epoch_df : scoring results as polars DataFrames.
    """
    times = load_scoring_times(subject, exp, sync_block)
    if len(times[0]) > 1:
        raise NotImplementedError(
            "Multiple scoring times per sync block not yet supported"
        )
    t1 = times[0][0]
    t2 = times[1][0]

    session = create_session(subject, exp, sync_block, t1, t2, store_chans)
    bout_df, epoch_df = score_session(model, session)

    # Save results
    hypno_dir = os.path.join(
        DEFS.anmat_root, subject, exp,
        "scoring_data", f"sync_block-{sync_block}",
        "hypnograms", "model_labelled",
    )
    wis.util.check_dir(hypno_dir)

    epoch_path = os.path.join(hypno_dir, "epochs.parquet")
    bout_path = os.path.join(hypno_dir, "bouts.parquet")

    if not os.path.exists(epoch_path) or overwrite:
        epoch_df.write_parquet(epoch_path)

    if not os.path.exists(bout_path) or overwrite:
        bout_df.write_parquet(bout_path)

    return bout_df, epoch_df


def autoscore_all_subjects(
    model_path: str | None = None,
    model: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    """Score all subjects/experiments/sync_blocks.

    Parameters
    ----------
    model_path : path to saved model file (used if ``model`` is None).
    model : pre-loaded model dict; takes precedence over ``model_path``.
    overwrite : whether to overwrite existing results.
    """
    if model is None:
        if model_path is None:
            raise ValueError("Must provide either model or model_path")
        model = load_model(model_path)

    si = wis.meta.get.sync_info()
    for subject in si:
        for exp in si[subject]:
            for sb in si[subject][exp]["sync_blocks"]:
                try:
                    autoscore_session(subject, exp, sb, model, overwrite=overwrite)
                    print(f"  Scored {subject} {exp} sync_block-{sb}")
                except Exception as e:
                    print(f"  Error scoring {subject} {exp} sync_block-{sb}: {e}")
