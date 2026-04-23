# Handoff: feat/parametric-sympy

Short note for the next Claude session picking up this work. Read
[`parametric-sympy.md`](parametric-sympy.md) for the full design; this file
only records the current state and any decisions that aren't obvious from the
code itself.

## Branch state

- Working branch: `feat/parametric-sympy`.
- All feature changes are **staged but uncommitted** (18 files, +1263/-512
  including the design doc). Run `git diff --cached --stat` to see them.
- The previous session couldn't commit because the repo's `pre-commit` hook
  hardcodes Sebastian's local pyenv Python (`/Users/junges/.pyenv/versions/
  storm13/bin/python`) and the sandbox proxy blocked `pip install pre-commit`
  plus GitHub clones of the hook repos.
- Project rule (from Sebastian): **do not push**. Commits on feature branches
  are fine once hooks pass, but pushing is his call.

## What was done

Replaced the hand-rolled `Polynomial` / `RationalFunction` classes with a
pluggable backend system whose default backend is sympy:

- `stormvogel/parametric/` — new package. `_backend.py` defines the
  `ParametricBackend` `Protocol` and a registry. `__init__.py` exposes
  `functools.singledispatch` generics (`is_zero`, `evaluate`, `degree`,
  `free_symbol_names`, `to_str`, `numerator_denominator`) plus
  `is_parametric`, `symbol`, `constant`. `sympy_backend.py` registers itself
  on import.
- `Model` now stores parameters as an ordered `dict[str, Parametric]` with
  `declare_parameter`, `parameter_symbols`, `unused_parameters`,
  `prune_parameters`; `set_choices` / `add_choices` auto-declare any
  previously-unseen free symbols.
- stormpy bridge (`stormvogel/stormpy_utils/*.py`) delegates sympy ↔ pycarl
  conversion to `backend.to_pycarl` / `backend.from_pycarl`, iterating
  `model.parameters` for a deterministic variable ordering.
- Tutorials, the `knuth_yao_pmc` example, and tests rewritten to use sympy.
  No deprecation shim — old API is gone.

## Known sharp edges

- `Model.get_instantiated_model` supports **partial substitution** as of the
  last fix in the previous session (only substituted keys are removed from
  `_parameters`; `_is_parametric` is invalidated to `None`). Regression
  tests: `test_partial_instantiation_preserves_remaining_parameters` and
  `test_instantiation_ignores_unknown_keys` in `tests/test_parametric.py`.
- The sandbox that previous Claude ran in was Python 3.10, but stormvogel
  uses 3.12-only generic class syntax (`class Model[ValueType: Value]`).
  Smoke tests there used a `types.ModuleType("stormvogel")` shim to
  cherry-pick the parametric package. On Sebastian's local Python 3.12 the
  normal `pytest` path should work.
- `pytest` + `stormpy` aren't available in the Cowork sandbox, so none of the
  full test-suite paths were exercised there. `test_parametric.py` round-trip
  tests depend on stormpy and will be skipped otherwise.

## Suggested next steps

1. Run `pre-commit run --all-files` locally; fix anything it flags.
2. Run `pytest` locally with stormpy installed to exercise the round-trip
   tests.
3. Commit + push (per the project rule, only when Sebastian approves the
   push).
4. Optional: implement a pycarl backend as a second file in
   `stormvogel/parametric/` — the design doc §4.5 and the
   `ParametricBackend` Protocol describe the contract. Nothing outside the
   package should need to change.

## Context that isn't in the repo

Sebastian steered the design toward: (a) sympy as the leaf type (vs.
rolling our own AST), (b) string-based sympy↔carl conversion (vs. AST
walk), (c) no migration shim from 0.11, (d) storing parameters on the model
rather than recomputing, (e) keeping the design pycarl-ready from day one.
All five are reflected in the final code — mentioning them here only so you
don't second-guess them without reason.
