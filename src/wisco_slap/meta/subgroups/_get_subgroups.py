"""Loading of experimental subgroup sets.

Membership is computed live: each set's generator script exposes a
``members()`` function, and the loader calls it. Nothing is cached on disk, so
a subgroup always reflects the current state of the data.

See ``subgroup_info.md`` in this directory for the full description of what a
subgroup set is, the folder layout every set must follow, and the
generator-script contract.
"""

import contextlib
import importlib
import io
from pathlib import Path
from types import ModuleType
from typing import Literal, overload

import polars as pl

from ..acq_types import classification as _load_classification
from ..acq_types import classification_values as _load_classification_values

SUBGROUPS_ROOT = Path(__file__).resolve().parent

ACQ_ID_FIELDS = ("subject", "exp", "loc", "acq")

#: Column name for the set-specific unit suffix of a member ID (the part after
#: ``subject--exp--loc--acq``) when ``return_df=True``. Sets not listed here
#: fall back to ``"unit"``.
UNIT_COLUMN_NAMES: dict[str, str] = {"single_soma": "soma_id"}


def list_subgroup_sets() -> list[str]:
    """Names of all subgroup sets defined under the subgroups package.

    A set counts as present when its generator script
    ``<set_name>/<set_name>.py`` exists (the same-name rule of the folder
    contract in ``subgroup_info.md``).
    """
    return sorted(
        path.parent.name
        for path in SUBGROUPS_ROOT.glob("*/*.py")
        if path.stem == path.parent.name
    )


def _set_module(set_name: str) -> ModuleType:
    """Import the generator module of subgroup set ``set_name``."""
    script = SUBGROUPS_ROOT / set_name / f"{set_name}.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"No subgroup set '{set_name}': expected a generator script at "
            f"{script}. Available subgroup sets: {list_subgroup_sets()}"
        )
    return importlib.import_module(f"{__package__}.{set_name}.{set_name}")


def _set_members(set_name: str, verbose: bool) -> list[str]:
    """Call the ``members()`` entry point of subgroup set ``set_name``."""
    module = _set_module(set_name)
    members_fn = getattr(module, "members", None)
    if not callable(members_fn):
        raise AttributeError(
            f"The generator script for subgroup '{set_name}' "
            f"({SUBGROUPS_ROOT / set_name / f'{set_name}.py'}) does not define "
            "a members() function. Every subgroup set must expose "
            "members(verbose: bool = True) -> list[str]; see subgroup_info.md."
        )
    if verbose:
        return list(members_fn(verbose=True))
    # Generators pull in helpers (valid_acquisitions, ...) that print
    # unconditionally, so verbose=False is enforced here rather than plumbed
    # through every dependency.
    with contextlib.redirect_stdout(io.StringIO()):
        return list(members_fn(verbose=False))


@overload
def load_subgroup_set(
    set_name: str,
    classification: dict[str, str | list[str]] | None = ...,
    verbose: bool = ...,
    return_df: Literal[False] = ...,
) -> list[str]: ...


@overload
def load_subgroup_set(
    set_name: str,
    classification: dict[str, str | list[str]] | None = ...,
    verbose: bool = ...,
    *,
    return_df: Literal[True],
) -> pl.DataFrame: ...


def load_subgroup_set(
    set_name: str,
    classification: dict[str, str | list[str]] | None = None,
    verbose: bool = True,
    return_df: bool = False,
) -> list[str] | pl.DataFrame:
    """Load the member identifiers of an experimental subgroup.

    Parameters
    ----------
    set_name : str
        Name of the subgroup directory, such as ``"single_soma"`` (see
        :func:`list_subgroup_sets`).
    classification : dict, optional
        Further filter the returned set on acquisition-level classifications
        (the JSON files in ``annotation_materials/_CLASSIFICATIONS/``, see
        ``wis.meta.acq_types.classification``). Keys are classification names,
        values are one allowed value or a list of allowed values, e.g.
        ``{"sleep_condition": "sd_early"}`` or
        ``{"sleep_condition": ["sd_early", "sd_late"]}``. Multiple keys are
        combined with AND. Requested values are checked against the
        classification's declared value scheme, so typos raise immediately.
        Members whose acquisition is unlabeled (absent or ``"NO_VALUE"``)
        are dropped.
    verbose : bool, optional
        Print the generator's selection log and a one-line summary of how many
        members survived each filtering stage. Defaults to ``True``.
    return_df : bool, optional
        If ``True``, return the members as a polars DataFrame with the member
        IDs split on ``"--"`` into ``subject``, ``exp``, ``loc``, ``acq`` and,
        when the set appends a unit suffix, one more column named per
        :data:`UNIT_COLUMN_NAMES` (``soma_id`` for ``single_soma``, ``unit``
        for sets without an entry). Defaults to ``False``.

    Returns
    -------
    list[str] or pl.DataFrame
        By default, member identifiers in the order the set's generator
        produced them, formatted as ``"subject--exp--loc--acq"`` plus any
        set-specific suffix (e.g. ``"--soma1"`` for ``single_soma``). With
        ``return_df=True``, the same members as one DataFrame row each (same
        order).

    Raises
    ------
    FileNotFoundError
        If the subgroup does not exist.
    AttributeError
        If the subgroup's generator script does not expose ``members()``.
    ValueError
        If a requested classification value is not part of that
        classification's declared scheme.

    Notes
    -----
    Membership is recomputed on every call by running the set's inclusion
    criteria against the current data — there is no cached member list, so a
    subgroup can never go stale. For ``single_soma`` this takes a few seconds.
    """
    members = _set_members(set_name, verbose)

    summary = [f"subgroup '{set_name}': {len(members)} members"]

    if classification is not None:
        members = _apply_classification_filter(
            members, classification, set_name, summary
        )

    if verbose:
        print(" | ".join(summary))
    if return_df:
        return members_to_df(members, set_name)
    return members


def members_to_df(members: list[str], set_name: str) -> pl.DataFrame:
    """Split member IDs into one row per member with one column per field.

    The first four ``--``-separated fields become ``subject``, ``exp``,
    ``loc``, ``acq``; any remaining suffix becomes a single extra column named
    per :data:`UNIT_COLUMN_NAMES` (only present when at least one member has
    a suffix). Column order is preserved from the IDs.
    """
    unit_col = UNIT_COLUMN_NAMES.get(set_name, "unit")
    rows: list[dict[str, str | None]] = []
    for member_id in members:
        parts = member_id.split("--")
        _acq_key(member_id, set_name)  # validates the four-field prefix
        row: dict[str, str | None] = dict(
            zip(ACQ_ID_FIELDS, parts[:4], strict=True)
        )
        row[unit_col] = "--".join(parts[4:]) if len(parts) > 4 else None
        rows.append(row)
    schema = {**{f: pl.String for f in ACQ_ID_FIELDS}, unit_col: pl.String}
    df = pl.DataFrame(rows, schema=schema)
    if df.get_column(unit_col).is_null().all():
        df = df.drop(unit_col)
    return df


def _apply_classification_filter(
    members: list[str],
    classification: dict[str, str | list[str]],
    set_name: str,
    summary: list[str],
) -> list[str]:
    """Keep only members whose acquisition matches every classification."""
    if not isinstance(classification, dict) or not classification:
        raise ValueError(
            "classification must be a non-empty dict, e.g. "
            "{'sleep_condition': 'sd_early'}; got "
            f"{classification!r}."
        )
    for cls_name, wanted in classification.items():
        wanted_values = [wanted] if isinstance(wanted, str) else list(wanted)
        if not wanted_values:
            raise ValueError(
                f"No values requested for classification '{cls_name}'."
            )
        declared = _load_classification_values(cls_name)
        unknown = [v for v in wanted_values if v not in declared]
        if unknown:
            raise ValueError(
                f"Unknown value(s) {unknown} for classification "
                f"'{cls_name}'. Declared values: {declared}"
            )

        labels = _load_classification(cls_name)
        label_by_acq = {
            (row["subject"], row["exp"], row["loc"], row["acq"]): row["value"]
            for row in labels.iter_rows(named=True)
        }
        kept: list[str] = []
        n_unlabeled = 0
        for member_id in members:
            value = label_by_acq.get(_acq_key(member_id, set_name))
            if value is None or value == "NO_VALUE":
                n_unlabeled += 1
            elif value in wanted_values:
                kept.append(member_id)
        unlabeled_note = (
            f", {n_unlabeled} unlabeled dropped" if n_unlabeled else ""
        )
        summary.append(
            f"{cls_name} in {wanted_values}: {len(kept)} kept{unlabeled_note}"
        )
        members = kept
    return members


def _acq_key(member_id: str, set_name: str) -> tuple[str, ...]:
    """The ``(subject, exp, loc, acq)`` prefix of a member identifier."""
    parts = member_id.split("--")
    if len(parts) < 4:
        raise ValueError(
            f"Malformed member ID {member_id!r} in subgroup '{set_name}': "
            "expected at least 'subject--exp--loc--acq'."
        )
    return tuple(parts[:4])
