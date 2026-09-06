# Experimental subgroups (`wis.meta.subgroups`)

A **subgroup set** is a named cohort of data units that satisfy explicit,
code-defined inclusion criteria. It is the answer to "which data goes into
this analysis?" — defined once, and then loaded identically by every analysis
that uses it.

**If you are making a summary plot or running an analysis across subjects,
experiments, or cells, you almost certainly should be drawing your data from one
of these subgroups** rather than assembling an ad-hoc list. That is what makes
cross-analysis results comparable and reproducible.

Example: `single_soma` scans all available data and keeps identified single
somas that were recorded for at least 20 minutes and have at least 40 labeled
synapses.

## How membership works

Each set is defined by a single script, `<set_name>/<set_name>.py`, which
exposes a `members()` function applying the inclusion criteria to the current
data. `wis.meta.subgroups.load_subgroup_set(set_name)` calls that function on
every load.

**Nothing is cached on disk.** There is no stored member list, so a subgroup
can never go stale: add a subject, fix an annotation, or mark an acquisition
invalid, and the next load reflects it. The criteria in code *are* the
definition of membership — there is exactly one source of truth.

The cost is a few seconds per load (~4 s for `single_soma`, which reads
`valid_acquisitions()` plus one synapse-label table per acquisition). Call it
once at the top of a notebook and reuse the result.

If you ever need a frozen cohort — e.g. pinning the exact N behind a published
figure — snapshot the returned list into that analysis's own outputs, where it
sits next to the numbers it produced. Do not reintroduce a member list in this
package: an uncommitted cache next to the criteria that generate it is the one
arrangement guaranteed to drift silently.

## Folder contract

Every subgroup set must follow this layout (this *is* the spec — the loader
relies on it):

```
subgroups/
├── subgroup_info.md               ← this file
├── __init__.py
├── _get_subgroups.py              ← the loader (load_subgroup_set, list_subgroup_sets)
└── <set_name>/                    ← one folder per subgroup set, e.g. single_soma/
    └── <set_name>.py              ← generator script, SAME NAME as the folder
```

The same-name rule is load-bearing: `list_subgroup_sets()` discovers sets by
looking for `<dir>/<dir>.py`, and the loader imports exactly that module.

### Member IDs

Every member ID starts with the acquisition identifier and may append a
set-specific unit suffix:

```
subject--exp--loc--acq[--unit]
e.g.  kaus--exp_1--loc_E--acq_1--soma1     (single_soma: the unit is a soma)
```

The first four `--`-separated fields are always `subject`, `exp`, `loc`, `acq`
— the loader relies on this to join against acquisition-level metadata such as
classifications. IDs are used as filenames elsewhere, so they must stay
filesystem-safe (no spaces or slashes).

### Generator script contract

`<set_name>.py` must define:

```python
def members(verbose: bool = True) -> list[str]:
    """The current members of this subgroup."""
```

- It is called on **every** `load_subgroup_set` call, so it must be
  deterministic given the current state of the data, and cheap enough to run
  interactively (seconds, not minutes).
- The order it returns is the order the loader returns.
- `verbose=False` must not print. (The loader also suppresses stdout in that
  case, since generators call helpers like `valid_acquisitions()` that print
  unconditionally — but a generator's own logging should still respect the
  flag.)
- Keep the underlying criteria function public and parameterized (e.g.
  `single_soma_acqs(min_n_synapses=..., min_duration_seconds=...)`) so the
  thresholds can be varied interactively; `members()` is just that function
  called with the standard values.

Running the script directly (`python <set_name>.py` from the `slap_mi_2_sleep`
venv) should print the current membership.

## Loading

```python
import wisco_slap as wis

# what sets exist?
wis.meta.subgroups.list_subgroup_sets()          # ["single_soma", ...]

# the subgroup's members (recomputed live, ~4 s for single_soma)
ids = wis.meta.subgroups.load_subgroup_set("single_soma")

# members labeled sd_early in the sleep_condition classification
ids = wis.meta.subgroups.load_subgroup_set(
    "single_soma", classification={"sleep_condition": "sd_early"}
)

# multiple allowed values (OR within a key), multiple keys (AND across keys)
ids = wis.meta.subgroups.load_subgroup_set(
    "single_soma",
    classification={"sleep_condition": ["sd_early", "sd_late"]},
)
```

To get the members as a polars DataFrame instead of a list of IDs, pass
`return_df=True`. The IDs are split on `--` into `subject`, `exp`, `loc`, `acq`
plus one column for the set's unit suffix, named per `UNIT_COLUMN_NAMES` in
`_get_subgroups.py` (`soma_id` for `single_soma`; `unit` for sets without an
entry; omitted when the set has no suffix). Add your set there when it has a
suffix. The same split is available standalone as
`wis.meta.subgroups.members_to_df(ids, set_name)`.

```python
df = wis.meta.subgroups.load_subgroup_set("single_soma", return_df=True)
# shape: (n, 5) — subject | exp | loc | acq | soma_id
```

Classification filtering joins each member's `subject--exp--loc--acq` prefix
against the acquisition classifications in
`analysis_materials/annotation_materials/_CLASSIFICATIONS/` (created via
WISynaptic's "Classify" button; see `wis.meta.acq_types.classification`). Two
guardrails to know about:

- Requested values are validated against the classification's declared value
  scheme, so a typo (`"early_sd"` instead of `"sd_early"`) raises a
  `ValueError` instead of silently returning an empty list.
- Members whose acquisition is unlabeled (missing or `"NO_VALUE"`) are dropped
  and counted in the printed summary.

Every call prints a one-line accounting of the filtering (disable with
`verbose=False`), e.g.:

```
subgroup 'single_soma': 34 members | sleep_condition in ['sd_early']: 23 kept
```

## Creating a new subgroup set

1. `mkdir <set_name>` here, and write `<set_name>/<set_name>.py` following the
   generator contract above (use `single_soma/single_soma.py` as the
   template).
2. Add an entry to `UNIT_COLUMN_NAMES` in `_get_subgroups.py` if your member
   IDs carry a unit suffix.
3. Run it from the `slap_mi_2_sleep` venv to eyeball the membership, then load
   it from analyses with `load_subgroup_set("<set_name>")`.

There is no regeneration step: when the underlying data changes (new subjects
onboarded, annotations fixed, criteria adjusted), the next load picks it up.
