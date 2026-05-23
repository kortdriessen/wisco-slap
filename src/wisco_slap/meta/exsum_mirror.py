"""Mirror ExperimentSummary files from raw data into analysis_materials.

The mirror filename is copied verbatim from the upstream preprocessing
pipeline; we do not rename. The modern convention is
``SummaryLoCo-YYMMDD-HHMMSS.mat`` but a handful of older acquisitions
(e.g. ``alkaid/exp_2``) still carry the older ``Summary-YYMMDD-HHMMSS.mat``
prefix without the ``LoCo`` tag. All readers in this project extract the
version identifier by stripping only the ``.mat`` extension
(``os.path.basename(p).split(".mat")[0]``), so both formats are handled
transparently. Do NOT rename legacy files to match the modern format -
the identifier stored in flag/marker files is the exact filename-minus-
``.mat`` and must remain byte-identical for freshness comparisons.

Public API:

- :func:`mirror_one_acq` — per-acq mirror; raises on failure. The unit
  called by :func:`wis.meta.update`.
- :func:`update_exsum_mirror` — bulk sweep that iterates every acq in
  ``acquisition_master.yaml`` and prints warnings rather than raising.
  Retained for backward compatibility; new code should prefer
  ``wis.meta.update()``.
"""

from __future__ import annotations

import os
import shutil

import wisco_slap as wis
from wisco_slap.defs import data_root, exsum_mirror_root
from wisco_slap.meta.get import acq_master


def mirror_one_acq(subject: str, exp: str, loc: str, acq: str) -> str:
    """Mirror a single acq's ExperimentSummary file.

    Copies the upstream ``<data_root>/<subject>/<exp>/<loc>/<acq>/
    ExperimentSummary/Summary*.mat`` into
    ``<exsum_mirror_root>/<subject>/<exp>/<loc>--<acq>/`` if the mirror
    is empty or out-of-date, and updates ``prepro_info.yaml`` to record
    the version identifier (or ``"NO"`` if no ExSum has been produced
    yet). Is a no-op when the mirror is already up-to-date.

    Raises
    ------
    FileNotFoundError
        If the raw acq directory is missing (e.g. NAS not mounted, or
        the acq was added to ``acquisition_master.yaml`` before raw data
        existed).
    AssertionError
        If more than one ``.mat`` file is present in the mirror dir
        (requires manual cleanup).

    Returns
    -------
    str
        One of ``"up_to_date"``, ``"copied"``, ``"updated"``,
        ``"no_exsum"``. Purely informational.
    """
    acq_id = f"{loc}--{acq}"
    raw_acq_root = os.path.join(data_root, subject, exp, loc, acq)
    if not os.path.exists(raw_acq_root):
        raise FileNotFoundError(
            f"No raw data directory for {subject}/{exp}/{loc}--{acq}: "
            f"{raw_acq_root}"
        )

    mirror_dir = os.path.join(exsum_mirror_root, subject, exp, acq_id)
    os.makedirs(mirror_dir, exist_ok=True)

    esum_path = wis.meta.get.esum_path_raw(subject, exp, loc, acq)
    if esum_path is None:
        wis.meta.prepro_info.update_prepro_info_acqid(
            subject, exp, loc, acq, value="NO"
        )
        return "no_exsum"

    expected_mirror_file = os.path.basename(esum_path)
    prepro_name = expected_mirror_file.split(".mat")[0]

    current_mirror_files = [
        f for f in os.listdir(mirror_dir) if f.endswith(".mat")
    ]

    # Empty mirror: copy ExSum in.
    if not current_mirror_files:
        mirror_esum_path = os.path.join(mirror_dir, expected_mirror_file)
        shutil.copyfile(esum_path, mirror_esum_path)
        wis.meta.prepro_info.update_prepro_info_acqid(
            subject, exp, loc, acq, value=prepro_name
        )
        return "copied"

    assert len(current_mirror_files) == 1, (
        f"More than one .mat file in mirror directory {mirror_dir} — "
        f"manual cleanup required. Files: {current_mirror_files}"
    )
    current_mirror_file = current_mirror_files[0]

    wis.meta.prepro_info.update_prepro_info_acqid(
        subject, exp, loc, acq, value=prepro_name
    )

    if current_mirror_file == expected_mirror_file:
        return "up_to_date"

    # Mirror is stale: remove old .mat, copy new one. We only touch .mat
    # files, not the directory as a whole — preserves any sidecar files
    # that might have been placed there.
    old_path = os.path.join(mirror_dir, current_mirror_file)
    os.remove(old_path)
    new_path = os.path.join(mirror_dir, expected_mirror_file)
    shutil.copyfile(esum_path, new_path)
    return "updated"


def update_exsum_mirror() -> list[str]:
    """Bulk sweep: mirror every acq in ``acquisition_master.yaml``.

    Errors per acq are printed rather than raised — this preserves the
    previous behavior of the function. For per-acq error isolation with
    a machine-readable trail (``meta_issues/`` files), use
    :func:`wis.meta.update` instead.

    Returns
    -------
    list[str]
        Identifiers of acquisitions whose raw data directory was missing,
        formatted as ``"<subject>--<exp>--<loc>--<acq>"``.
    """
    acqs = acq_master()
    missing_raw_data: list[str] = []
    for subject in acqs:
        for exp in acqs[subject]:
            for acq_id in acqs[subject][exp]:
                loc, acq = acq_id.split("--")
                try:
                    status = mirror_one_acq(subject, exp, loc, acq)
                except FileNotFoundError:
                    print(f"No data directory for {subject} {exp} {loc}--{acq}!")
                    missing_raw_data.append(
                        f"{subject}--{exp}--{loc}--{acq}"
                    )
                    continue
                except AssertionError as exc:
                    print(f"[{subject} {exp} {loc}--{acq}] {exc}")
                    continue
                if status == "up_to_date":
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} is up to "
                        f"date, skipping."
                    )
                elif status == "copied":
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} was empty, "
                        f"copied esum file into mirror directory."
                    )
                elif status == "updated":
                    print(
                        f"Mirror for {subject} {exp} {loc} {acq} was "
                        f"mismatched, updated to proper file."
                    )
    wis.meta.prepro_info.update_prepro_info()
    return missing_raw_data
