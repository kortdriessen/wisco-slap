# Experimental subgroups (`wis.meta.subgroups`)

A **subgroup set** is a named, version-controlled, human-reviewed cohort of data
units that satisfy explicit inclusion criteria. It is the answer to "which data
goes into this analysis?" — defined once, reviewed once, and then loaded
identically by every analysis that uses it.

**If you are making a summary plot or running an analysis across subjects,
experiments, or cells, you almost certainly should be drawing your data from one
of these subgroups** rather than assembling an ad-hoc list. That is what makes
cross-analysis results comparable and reproducible.

Example: `single_soma` scans all available data and keeps identified single
somas that were recorded for at least 20 minutes and have at least 50 labeled
synapses.

## The three-stage lifecycle

1. **Generate** — the set's generator script (`<set_name>/<set_name>.py`) scans
   all available data, applies the inclusion criteria, and writes the
   *candidate set*: a JSON list of member IDs plus one overview image per
   candidate. This is automated and re-runnable; it makes **no** judgment about
   data quality beyond the mechanical criteria.
2. **Review** — a human (you) walks through the `main_overview/` images in
   **Lighttable** and marks each candidate (`valid`, or a rejection mark such
   as `exclude`, optionally with notes). Lighttable saves these marks to
   `candidate_set/.lt-annotations/validity.json`. This is the validation layer:
   a candidate is only *in* the subgroup once it carries `mark: "valid"`.
3. **Load** — analysis code calls
   `wis.meta.subgroups.load_subgroup_set(set_name)` and gets back the validated
   member IDs (see "Loading" below).

Because the candidate sets and validity marks live inside the `wisco-slap`
package (not out in `analysis_materials/`), cohort membership is git-tracked
alongside the code — you can always reconstruct exactly which cells an old
analysis ran on.

## Folder contract

Every subgroup set must follow this layout (this *is* the spec — the loader and
the review tooling both rely on it):

```
subgroups/
├── subgroup_info.md               ← this file
├── __init__.py
├── _get_subgroups.py              ← the loader (load_subgroup_set, list_subgroup_sets)
└── <set_name>/                    ← one folder per subgroup set, e.g. single_soma/
    ├── <set_name>.py              ← generator script, SAME NAME as the folder
    └── candidate_set/
        ├── candidate_set.json     ← flat JSON list of member IDs (generator output)
        ├── main_overview/         ← required: the plots reviewed for validation
        │   └── <member-id>.png    ← exactly one per candidate
        ├── <other_plot_dir>/      ← optional: any number of extra per-candidate
        │   └── <member-id>.png       plot directories, same one-PNG-per-ID rule
        └── .lt-annotations/
            └── validity.json      ← Lighttable review marks (created by review, step 2)
```

### Member IDs

Every member ID starts with the acquisition identifier and may append a
set-specific unit suffix:

```
subject--exp--loc--acq[--unit]
e.g.  kaus--exp_1--loc_E--acq_1--soma1     (single_soma: the unit is a soma)
```

The first four `--`-separated fields are always `subject`, `exp`, `loc`, `acq`
— the loader relies on this to join against acquisition-level metadata such as
classifications. IDs double as plot filenames, so they must stay
filesystem-safe (no spaces or slashes).

### `candidate_set.json`

A flat JSON list of member IDs, in whatever order the generator produced them
(the loader preserves this order):

```json
[
  "avior--exp_1--loc_I--acq_1--soma1",
  "kaus--exp_1--loc_E--acq_1--soma1"
]
```

### `.lt-annotations/validity.json`

Written by Lighttable during review; never edited by hand or by the generator:

```json
{
  "version": 1,
  "class": "validity",
  "updated": "2026-08-12T19:24:06.215Z",
  "items": {
    "kaus--exp_1--loc_E--acq_1--soma1": { "mark": "valid" },
    "elnath--exp_3--loc_D--acq_1--soma1": {
      "mark": "exclude",
      "notes": "#CHECK dendrite damage, short recording"
    }
  }
}
```

Interpretation rules (enforced by the loader):

- `mark == "valid"` → in the validated subgroup.
- any other mark (e.g. `"exclude"`) → reviewed and rejected.
- absent from `items` → **not yet reviewed**; treated as not-validated. The
  loader reports the unreviewed count so silent gaps are visible.
- marks for IDs that are no longer in `candidate_set.json` are *stale*
  (candidate set was regenerated since review) → the loader emits a warning
  naming them; re-review after regenerating.

### Generator script contract

`<set_name>.py` must:

- Rebuild `candidate_set.json` from scratch on every run (deterministic given
  the current state of the data).
- Write one overview PNG per candidate into `candidate_set/main_overview/`,
  named `<member-id>.png`. If plot generation fails for a candidate, write a
  blank PNG rather than nothing, so the review set stays complete.
- Skip already-existing PNGs by default and support a `--replace` flag to
  regenerate them (plots are expensive; membership recomputation is not).
- Never touch `.lt-annotations/` — review marks belong to the human.

It should be runnable both as a script (`python <set_name>.py [--replace]`)
from the `slap_mi_2_sleep` venv, and expose its selection function for
interactive use.

The `single_soma` generator gets its `main_overview/` images from the
`single_soma` **plot set** of `wis.viz.overview_and_summary_plots`
(`single_soma_overview_plot`, saved under
`annotation_materials/OVERVIEW_FIGURES/single_soma/single_soma_overview/`):
`write_main_overviews` runs
`wis.pns.overview_plot_gen.generate_plot_set("single_soma", overwrite=replace)`
— which iterates the same `single_soma_acqs()` selection and writes a blank
placeholder + `--ERROR.txt` sidecar for any failure — and copies each
candidate's PNG into `main_overview/`. Adding another per-candidate plot is
therefore a matter of adding a module to that plot-set subpackage.

## Loading

```python
import wisco_slap as wis

# what sets exist?
wis.meta.subgroups.list_subgroup_sets()          # ["single_soma", ...]

# the validated subgroup (the default — use this for real analyses)
ids = wis.meta.subgroups.load_subgroup_set("single_soma")

# everything the generator found, un-reviewed (use for generating review
# materials, e.g. lighttable plot suites)
ids = wis.meta.subgroups.load_subgroup_set("single_soma", version="candidates")

# validated AND labeled sd_early in the sleep_condition classification
ids = wis.meta.subgroups.load_subgroup_set(
    "single_soma", classification={"sleep_condition": "sd_early"}
)

# multiple allowed values (OR within a key), multiple keys (AND across keys)
ids = wis.meta.subgroups.load_subgroup_set(
    "single_soma",
    classification={"sleep_condition": ["sd_early", "sd_late"]},
)
```

Classification filtering joins each member's `subject--exp--loc--acq` prefix
against the acquisition classifications in
`analysis_materials/annotation_materials/_CLASSIFICATIONS/` (created via
WISynaptic's "Classify" button; see `wis.meta.get.classification`). Two
guardrails to know about:

- Requested values are validated against the classification's declared value
  scheme, so a typo (`"early_sd"` instead of `"sd_early"`) raises a
  `ValueError` instead of silently returning an empty list.
- Members whose acquisition is unlabeled (missing or `"NO_VALUE"`) are dropped
  and counted in the printed summary.

Every call prints a one-line accounting of the filtering (disable with
`verbose=False`), e.g.:

```
subgroup 'single_soma': 32 candidates | validated: 25 kept (5 marked not-valid, 2 unreviewed) | sleep_condition in ['sd_early']: 17 kept
```

## Creating a new subgroup set

1. `mkdir <set_name>` here, and write `<set_name>/<set_name>.py` following the
   generator contract above (use `single_soma/single_soma.py` as the
   template).
2. Run it from the `slap_mi_2_sleep` venv → produces `candidate_set/` with
   `candidate_set.json` and `main_overview/`.
3. Review the `main_overview` images in Lighttable, marking each candidate →
   produces `.lt-annotations/validity.json`.
4. (Optional) add richer per-candidate plot suites as sibling directories of
   `main_overview/` for deeper review passes.
5. Analyses load it with `load_subgroup_set("<set_name>")`.

When the underlying data changes (new subjects onboarded, criteria adjusted),
re-run the generator and re-review: the loader will warn about stale marks and
report unreviewed candidates until the review catches up.
