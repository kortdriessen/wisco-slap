"""Acquisition-level classifications ("acquisition types").

Classifications are created and labeled in WISynaptic via the "Classify"
toolbar button. Each one is a JSON file in
``annotation_materials/_CLASSIFICATIONS/<name>.json`` holding the declared
value scheme (with hotkeys) and a flat per-acquisition label record list.
``"NO_VALUE"`` is the placeholder recorded for acquisitions that have been
seen but not yet labeled.
"""

import json
from pathlib import Path

import polars as pl

from wisco_slap.defs import anmat_root

NO_VALUE = "NO_VALUE"
_ACQ_SCHEMA = {
    "subject": pl.String,
    "exp": pl.String,
    "loc": pl.String,
    "acq": pl.String,
    "value": pl.String,
}


def _classifications_dir() -> Path:
    return Path(anmat_root) / "annotation_materials" / "_CLASSIFICATIONS"


def classification_names() -> list[str]:
    """Names of all saved acquisition classifications, sorted."""
    cdir = _classifications_dir()
    if not cdir.is_dir():
        return []
    return sorted(p.stem for p in cdir.glob("*.json"))


def _load_classification_json(name: str) -> dict:
    """Read one classification's raw JSON, with a helpful error if absent."""
    path = _classifications_dir() / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No classification named '{name}' at {path}. "
            f"Available: {classification_names()}"
        )
    with open(path) as f:
        return json.load(f)


def classification_values(name: str) -> list[str]:
    """Declared value scheme for one classification, in declaration order.

    Parameters
    ----------
    name : str
        Classification name, e.g. ``"sleep_condition"`` (see
        :func:`classification_names`).

    Returns
    -------
    list[str]
        The values an acquisition may be labeled with, e.g.
        ``["sd_early", "sleep_early", "sd_late", "sleep_late"]``. Does not
        include ``"NO_VALUE"``.
    """
    return [entry["value"] for entry in _load_classification_json(name)["values"]]


def classification(name: str) -> pl.DataFrame:
    """Load one acquisition classification as a tidy dataframe.

    Parameters
    ----------
    name : str
        Classification name, e.g. ``"sleep_condition"`` (see
        :func:`classification_names`).

    Returns
    -------
    pl.DataFrame
        One row per acquisition with columns ``subject``, ``exp``, ``loc``,
        ``acq``, ``value``. A value of ``"NO_VALUE"`` means the acquisition
        has not been labeled yet.
    """
    return pl.DataFrame(_load_classification_json(name)["labels"], schema=_ACQ_SCHEMA)


def load_acquisition_class_subset(classification_name: str, value: str) -> pl.DataFrame:
    """Acquisitions labeled ``value`` in one classification.

    Parameters
    ----------
    classification_name : str
        Which classification to read, e.g. ``"sleep_condition"`` (see
        :func:`classification_names`).
    value : str
        The label to select, e.g. ``"sd_early"``.

    Returns
    -------
    pl.DataFrame
        Columns ``subject``, ``exp``, ``loc``, ``acq``, sorted; one row per
        acquisition carrying that label.

    Raises
    ------
    ValueError
        If ``value`` is not part of the classification's declared scheme.
    """
    declared = classification_values(classification_name)
    if value not in declared:
        raise ValueError(
            f"{value!r} is not a declared value of {classification_name!r}. "
            f"Declared: {declared}"
        )
    return (
        classification(classification_name)
        .filter(pl.col("value") == value)
        .drop("value")
        .sort("subject", "exp", "loc", "acq")
    )


def add_acq_class(df: pl.DataFrame, classification_name: str) -> pl.DataFrame:
    """Add a column holding each acquisition's label in one classification.

    Parameters
    ----------
    df : pl.DataFrame
        Any frame with (at least) ``subject``, ``exp``, ``loc``, ``acq``
        columns; other columns and row order are preserved.
    classification_name : str
        Which classification to read, e.g. ``"sleep_condition"``. Its name
        becomes the new column's name.

    Returns
    -------
    pl.DataFrame
        ``df`` with a ``<classification_name>`` string column appended.
        Acquisitions absent from the classification file get ``null``;
        acquisitions seen but not yet labeled keep ``"NO_VALUE"``.

    Raises
    ------
    ValueError
        If ``df`` lacks one of the four ID columns.
    """
    keys = ["subject", "exp", "loc", "acq"]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"df is missing required column(s) {missing}.")
    labels = classification(classification_name).rename({"value": classification_name})
    return df.join(labels, on=keys, how="left", maintain_order="left")
