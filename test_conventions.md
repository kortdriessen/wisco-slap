# Testing conventions for `wisco-slap`

A short reference so testing patterns stay consistent across sessions. Read this
before writing tests in this repo.

## Why this doc exists

The user (Kort) has no formal background in software testing and has explicitly
delegated all testing decisions to whoever is editing the codebase. They will
not bring up tests, ever — not because tests are unwanted, but because they
don't have the vocabulary to ask. This doc records the conventions so we don't
have to rediscover them every session, and so test code stays uniform.

If you're tempted to introduce a new test pattern (different framework, mocking
library, parametrization style, etc.), update this doc *first* with the
rationale; otherwise stick to what's described below.

## Framework: `pytest`

Confirmed by the `pytest.ini` at the repo root and the existing tests in
`tests/`. No other framework. Don't reach for `unittest`, `nose`, etc.

The `pytest.ini` currently disables napari plugins (`-p no:napari -p
no:napari-plugin-engine`) — that's a workaround for plugin discovery issues
when napari is installed in the same environment, not a feature flag. Don't
remove it without understanding why.

## Layout

```
wisco-slap/
├── src/wisco_slap/
│   └── <package code>
└── tests/
    ├── test_<module>.py
    └── ...
```

- One test file per module-or-feature being tested. `test_<name>.py` mirrors
  the source structure but stays flat (no `tests/util/`, `tests/pns/` subdirs)
  unless the package has so many tests that flat layout becomes painful. As of
  writing, flat is fine.
- Helpers shared across files go in `tests/conftest.py`. Don't import from
  other test files.
- Tests import the package directly (`import wisco_slap`, `from wisco_slap.foo
  import bar`) — no `sys.path` munging.

## Style

- Start each test file with `from __future__ import annotations`.
- Test functions are plain `def test_<name>(...) -> None`. No classes unless a
  natural grouping emerges.
- Use `assert` directly. Don't reach for `assertEqual`-style helpers.
- Prefer `pytest`'s built-in fixtures (`tmp_path`, `monkeypatch`, `capsys`)
  over custom infrastructure.
- One assertion per concept where reasonable, but multi-assert is fine —
  prioritize readability over rigid one-assert-per-test.

## What to test (and what not to)

The repo is research code in active development. Tests exist to (1) prevent
silent miscomputation in code that's hard to spot-check by eye, and (2) lock
in algorithmic correctness for routines whose outputs feed multiple
downstream analyses. They do *not* exist for coverage's sake.

**Do write tests for:**

- Algorithmic routines whose correctness is non-obvious from reading the code
  (sample-index arithmetic, NaN-aware accounting, anything with edge cases at
  boundaries / empty inputs / off-by-one risk).
- Anything that produces a number that downstream science depends on (event
  rates, durations, fractions). Even one assert that pins the expected output
  for a known input prevents silent regressions.
- Public APIs that other modules consume — if changing the function would
  break callers across the repo, lock the contract.

**Skip tests for:**

- Pure plotting code (visual inspection is the test).
- Thin wrappers around well-tested third-party calls (xarray indexing,
  polars groupby) where the wrapper just forwards args.
- Code that's clearly going to be rewritten or deprecated in the next session.
- Code that requires real data files / a real ExperimentSummary / a real GPU.
  Use synthetic inputs constructed in the test instead.

## Synthetic-data pattern

For algorithmic tests, construct inputs in the test from `polars` /
`numpy` / `xarray` literals — don't read fixtures from disk. This keeps
tests fast, hermetic, and self-documenting (the test body shows what the
input looks like).

Example shape:

```python
def test_valid_state_epochs_spans_short_nan_gap() -> None:
    # NREM 0–13s with a 2s NaN at 6–8s, target = 10s of valid data.
    hypno = ...   # synthetic hypnogram
    mask  = ...   # 1-D boolean DataArray on a 200 Hz grid
    epochs = valid_state_epochs(hypno, mask, "NREM", epoch_length=10)
    assert epochs.height == 1
    assert epochs.row(0, named=True)["start_time"] == 0.0
    assert epochs.row(0, named=True)["end_time"]   == 12.0
```

When the table-of-cases pattern is natural (e.g. the worked-examples table
in a plan file), use `pytest.mark.parametrize` to expand them rather than
copy-pasting the test body.

## Running tests

From the repo root:

```bash
cd /data/slap_analysis/wisco-slap
/data/slap_analysis/slap_mi_2_sleep/.venv/bin/python -m pytest tests/ -v
```

Or scoped to one file:

```bash
/data/slap_analysis/slap_mi_2_sleep/.venv/bin/python -m pytest tests/test_validity.py -v
```

Always use the `slap_mi_2_sleep` venv — it's where all the deps are
installed (per project-wide convention).

## Tolerances and floating-point

- Many algorithmic results land on a sample grid (~200 Hz, dt = 0.005s). Use
  `pytest.approx` with `abs=1e-9` for sample-index-derived floats; use
  `abs=dt` (one sample) for boundaries that depend on rounding.
- Don't compare polars DataFrames with `==` directly across rows — use
  `df.sort(...)` then column-wise asserts, or `polars.testing.assert_frame_equal`.

## When tests fail in an unrelated way

Some tests use `monkeypatch` to stub out filesystem / network / metadata calls
(see `test_meta_status.py`). If those break after a refactor, the test body
shows what the stub assumed — fix the stub, not the production code, unless
the production behavior actually changed.
