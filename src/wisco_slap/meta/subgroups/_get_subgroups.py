"""Loading of experimental subgroup sets.

See ``subgroup_info.md`` in this directory for the full description of what a
subgroup set is, the folder layout every set must follow, and the
generate -> review -> load workflow.
"""

import json
import warnings
from pathlib import Path
from typing import Literal

from ..get import classification as _load_classification
from ..get import classification_values as _load_classification_values

SUBGROUPS_ROOT = Path(__file__).resolve().parent


def list_subgroup_sets() -> list[str]:
    """Names of all subgroup sets that have a generated candidate set.

    A set counts as present when
    ``<set_name>/candidate_set/candidate_set.json`` exists under the
    subgroups package directory.
    """
    return sorted(
        path.parent.parent.name
        for path in SUBGROUPS_ROOT.glob("*/candidate_set/candidate_set.json")
    )


def load_subgroup_set(
    set_name: str,
    version: Literal["valid", "candidates"] = "valid",
    classification: dict[str, str | list[str]] | None = None,
    verbose: bool = True,
) -> list[str]:
    """Load the member identifiers of an experimental subgroup.

    Parameters
    ----------
    set_name : str
        Name of the subgroup directory, such as ``"single_soma"`` (see
        :func:`list_subgroup_sets`).
    version : {"valid", "candidates"}, optional
        Which set to load. ``"candidates"`` returns everything the generator
        script found (``candidate_set.json``). ``"valid"`` (the default)
        returns only candidates that a human has marked ``"valid"`` during
        Lighttable review (``.lt-annotations/validity.json``); candidates
        marked anything else, or not yet reviewed, are dropped.
    classification : dict, optional
        Further filter the returned set on acquisition-level classifications
        (the JSON files in ``annotation_materials/_CLASSIFICATIONS/``, see
        ``wis.meta.get.classification``). Keys are classification names,
        values are one allowed value or a list of allowed values, e.g.
        ``{"sleep_condition": "sd_early"}`` or
        ``{"sleep_condition": ["sd_early", "sd_late"]}``. Multiple keys are
        combined with AND. Requested values are checked against the
        classification's declared value scheme, so typos raise immediately.
        Candidates whose acquisition is unlabeled (absent or ``"NO_VALUE"``)
        are dropped.
    verbose : bool, optional
        Print a one-line summary of how many candidates survived each
        filtering stage. Defaults to ``True``.

    Returns
    -------
    list[str]
        Member identifiers in candidate-set order, formatted as
        ``"subject--exp--loc--acq"`` plus any set-specific suffix
        (e.g. ``"--soma1"`` for ``single_soma``).

    Raises
    ------
    FileNotFoundError
        If the subgroup (or, for ``version="valid"``, its validity
        annotations) does not exist.
    ValueError
        If ``version`` is unrecognized, or a requested classification value
        is not part of that classification's declared scheme.
    """
    set_dir = SUBGROUPS_ROOT / set_name
    candidate_set_path = set_dir / "candidate_set" / "candidate_set.json"
    if not candidate_set_path.is_file():
        raise FileNotFoundError(
            f"No candidate set for subgroup '{set_name}' at {candidate_set_path}. "
            f"Available subgroup sets: {list_subgroup_sets()}"
        )
    with candidate_set_path.open(encoding="utf-8") as file:
        candidates: list[str] = json.load(file)

    summary = [f"subgroup '{set_name}': {len(candidates)} candidates"]

    if version == "candidates":
        members = list(candidates)
    elif version == "valid":
        members = _validated_subset(candidates, set_dir, set_name, summary)
    else:
        raise ValueError(
            f"version must be 'valid' or 'candidates', got {version!r}."
        )

    if classification is not None:
        members = _apply_classification_filter(
            members, classification, set_name, summary
        )

    if verbose:
        print(" | ".join(summary))
    return members


def _validated_subset(
    candidates: list[str],
    set_dir: Path,
    set_name: str,
    summary: list[str],
) -> list[str]:
    """Restrict ``candidates`` to those marked ``"valid"`` in Lighttable."""
    validity_path = set_dir / "candidate_set" / ".lt-annotations" / "validity.json"
    if not validity_path.is_file():
        raise FileNotFoundError(
            f"No validity annotations for subgroup '{set_name}' at "
            f"{validity_path}. Review the main_overview images in Lighttable "
            "to create them, or load the unreviewed set with "
            "version='candidates'."
        )
    with validity_path.open(encoding="utf-8") as file:
        marks = {
            member_id: item.get("mark")
            for member_id, item in json.load(file)["items"].items()
        }

    stale = sorted(set(marks) - set(candidates))
    if stale:
        warnings.warn(
            f"validity.json for subgroup '{set_name}' has {len(stale)} mark(s) "
            f"for IDs not in candidate_set.json (left over from an older "
            f"candidate set?): {stale}. Re-review after regenerating the "
            "candidate set.",
            stacklevel=3,
        )

    kept = [c for c in candidates if marks.get(c) == "valid"]
    n_not_valid = sum(1 for c in candidates if c in marks and marks[c] != "valid")
    n_unreviewed = sum(1 for c in candidates if c not in marks)
    summary.append(
        f"validated: {len(kept)} kept "
        f"({n_not_valid} marked not-valid, {n_unreviewed} unreviewed)"
    )
    return kept


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
