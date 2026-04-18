# Design: Proper parametric MDP support via `sympy` (with a pluggable backend)

**Status:** Proposal — for review before implementation
**Author:** drafted with Claude for Sebastian Junges
**Date:** 2026-04-16
**Scope:** Replace the hand-rolled `Polynomial` / `RationalFunction` classes in `stormvogel.parametric` with a clean representation backed by `sympy.Expr`, behind a small backend-abstraction layer so that `pycarl` (and potentially other libraries) can be plugged in later without touching the rest of stormvogel. Remove the legacy `add_term` / `.terms` / `Polynomial` / `RationalFunction` API and update all downstream uses (model, distribution, simulator, result, stormpy mapping, docs, tests).

## 1. Motivation

The current `stormvogel.parametric` module is a self-contained reimplementation of polynomials and rational functions:

- `Polynomial` stores terms as `dict[frozenset, float]` keyed on exponent tuples, with an `add_term(exponents, coefficient)` API.
- `RationalFunction` stores a pair of such polynomials.
- `Parametric = Polynomial | RationalFunction` is used as a leaf in the `Value` type and plumbed through distributions, the model, the simulator, results, and the stormpy bridge.

Problems:

1. **Semantic poverty.** No arithmetic (`+`, `*`, `-`, substitution of partial values), no simplification, no GCD/factorization, no differentiation, no coefficient-in-a-field notion. Two syntactically different but semantically equal polynomials are not equal, e.g. `x*y + y*x` cannot be constructed at all, and `add_term((1,1,1),4)` with variables `["x","y","z"]` silently loses variable order via `frozenset`.
2. **Fragile stormpy round-trip.** `convert_polynomial` in `stormpy_utils/mapping.py` parses pycarl polynomials via regex on `str(polynomial)` — brittle for any non-trivial term ordering and coefficient form.
3. **API clunkiness.** Users must pre-declare the variable list and then call `add_term(tuple, coef)` positionally. The tutorial (`docs/5-Advanced/1-Parametric-and-interval-models.py`) has to construct `x`, `1-x` with two `add_term` calls each.
4. **Float-only coefficients.** `add_term` casts coefficients to `float`; exact rationals are lost.
5. **`get_degree` bug.** `if self.terms is not {}` is always `True` (identity, not equality), so the early-return-on-empty branch is dead code; empty polynomials still iterate and return degree `0` instead of raising.
6. **Duplicate domain logic.** `is_zero`, `evaluate`, `get_variables`, `get_degree`, `__lt__`, `__eq__` already exist in `sympy` and are both more correct and faster there.

`sympy` is already a declared dependency in `pyproject.toml`, so there is no new third-party cost.

## 2. Design principles

- **No wrapper class; plain values carry their own type.** A parametric value is whatever object the active backend produces — a `sympy.Expr` today, a pycarl `Polynomial` / `RationalFunction` tomorrow. Users build them with ordinary arithmetic on backend-native symbols. The only stormvogel-specific surface is a small module of dispatching helpers.
- **Pluggable backend.** Everything stormvogel needs from a parametric value (`is_zero`, `free_symbol_names`, `evaluate`, `degree`, `numerator_denominator`, `to_pycarl`, `from_pycarl`, `to_str`) is exposed as top-level functions in `stormvogel.parametric` and implemented per backend via `functools.singledispatch`. Adding pycarl later is a new file — no edits to `model.py`, `distribution.py`, `value.py`, or the stormpy bridge.
- **Default backend: sympy.** `stormvogel.parametric.backend` names the active default. When stormvogel itself needs to *construct* a parametric value (rare — mostly for the stormpy → stormvogel direction and for internal constants), it goes through `backend.symbol(name)` / `backend.constant(n)` so swapping the default does not cascade into call sites.
- **Exact coefficients by default.** Integer and `Fraction` coefficients flow through as `sympy.Rational`. Floats are allowed but discouraged.
- **One conceptual leaf type in `Value`.** `Value = Number | Parametric | Interval`, where `Parametric` is a type alias that currently resolves to `sympy.Expr` and broadens to `sympy.Expr | pycarl.Polynomial | pycarl.RationalFunction` once the pycarl backend is registered. Downstream code uses `parametric.is_parametric(value)` / the helper functions — never an `isinstance` on a concrete backend class.
- **Round-tripping with stormpy is lossless for polynomials / rational functions.** For the sympy backend, sympy → pycarl walks the expression via `sympy.Poly` rather than doing regex parsing; pycarl → sympy parses the pycarl string with `sympy.sympify` (with a controlled `local_dict`). For a future pycarl backend, both directions are near-identity.
- **Backwards compatibility: none.** Per the agreed direction, the old API is removed.

## 3. Target API (`stormvogel/parametric/`)

The current single-file `stormvogel/parametric.py` becomes a package:

```
stormvogel/parametric/
    __init__.py            # public API, singledispatch generic functions, Parametric alias, backend selection
    _backend.py            # Backend protocol + registry + default-backend accessor
    sympy_backend.py       # sympy implementation (registered on import)
    # future: pycarl_backend.py
```

### 3.1 Public helpers (in `__init__.py`)

All take a parametric value and dispatch on its type:

```python
from functools import singledispatch
from fractions import Fraction
from typing import Protocol, TypeAlias, runtime_checkable
import sympy as sp

Number: TypeAlias = int | float | Fraction

# Type alias. Widened when further backends register their types.
Parametric: TypeAlias = sp.Expr  # extended at runtime; see _backend.py

@singledispatch
def is_zero(value) -> bool: ...

@singledispatch
def free_symbol_names(value) -> set[str]: ...

@singledispatch
def degree(value) -> int: ...

@singledispatch
def evaluate(value, values: dict[str, Number]) -> Number: ...

@singledispatch
def numerator_denominator(value) -> tuple["Parametric", "Parametric"]: ...

@singledispatch
def to_str(value) -> str: ...

# Construction helpers (dispatch via the currently-selected default backend):
def symbol(name: str) -> Parametric: ...
def constant(c: Number) -> Parametric: ...
def is_parametric(value) -> bool: ...
```

`is_parametric(value)` returns `True` iff `value`'s type is registered with any backend. This is what `Model.is_parametric`, `Distribution.is_stochastic`, `value.is_zero`, `result.maximum`, etc. should use — **never** a direct `isinstance(..., sp.Expr)`.

### 3.2 Backend protocol (in `_backend.py`)

```python
@runtime_checkable
class ParametricBackend(Protocol):
    name: str
    expr_types: tuple[type, ...]           # concrete types this backend owns
    def symbol(self, name: str) -> "Parametric": ...
    def constant(self, n: Number) -> "Parametric": ...
    # stormpy bridge, both directions (pycarl is optional at install time):
    def to_pycarl(self, value, var_map: dict[str, "pycarl.Variable"]) -> "pycarl.RationalFunction": ...
    def from_pycarl(self, pycarl_value) -> "Parametric": ...

_BACKENDS: list[ParametricBackend] = []
_default: ParametricBackend | None = None

def register(backend: ParametricBackend) -> None: ...
def set_default(name: str) -> None: ...
def get_default() -> ParametricBackend: ...
def types() -> tuple[type, ...]:    # union of all registered expr_types
    ...
```

The singledispatch helpers in `__init__.py` are registered by each backend for its own types. `is_parametric(value)` becomes `isinstance(value, types())`.

### 3.3 Sympy backend (in `sympy_backend.py`)

Implements the protocol and registers overloads for `sp.Expr`:

```python
import sympy as sp
from stormvogel.parametric import is_zero, free_symbol_names, degree, \
    evaluate, numerator_denominator, to_str
from stormvogel.parametric._backend import register, ParametricBackend

@is_zero.register(sp.Expr)
def _(value): return sp.simplify(value) == 0

@free_symbol_names.register(sp.Expr)
def _(value): return {s.name for s in value.free_symbols}

@degree.register(sp.Expr)
def _(value):
    return sp.Poly(value, *value.free_symbols).total_degree() if value.free_symbols else 0

@evaluate.register(sp.Expr)
def _(value, values):
    subs = {sp.Symbol(k) if isinstance(k, str) else k: v for k, v in values.items()}
    r = value.subs(subs)
    return float(r) if r.is_Float or r.is_Rational or r.is_Integer else r

@numerator_denominator.register(sp.Expr)
def _(value):
    return sp.fraction(sp.together(value))

@to_str.register(sp.Expr)
def _(value): return sp.sstr(value)

class SympyBackend:
    name = "sympy"
    expr_types = (sp.Expr,)
    def symbol(self, name): return sp.Symbol(name)
    def constant(self, n):  return sp.Rational(n) if isinstance(n, (int, Fraction)) else sp.Float(n)
    def to_pycarl(self, value, var_map): ...      # see §4.5
    def from_pycarl(self, v):             ...      # see §4.6

register(SympyBackend())
```

This backend is imported and registered by `stormvogel/parametric/__init__.py` on import, so the out-of-the-box behaviour matches the prior design draft.

### 3.4 Future: pycarl backend

`stormvogel/parametric/pycarl_backend.py` would register `@is_zero.register(pycarl.cln.Polynomial)` and friends, implement the protocol, and be opt-in via `stormvogel.parametric.set_default("pycarl")`. No call site in the rest of stormvogel changes. The stormpy bridge becomes near-identity for that backend.

## 4. Changes per file

### 4.1 `stormvogel/parametric/` (package)
New structure per §3. Old symbols (`Polynomial`, `RationalFunction`) are deleted — not even a deprecation shim, per the agreed direction. A `stormvogel.parametric.Parametric` alias remains as the authoritative leaf type used in type annotations.

### 4.2 `stormvogel/model/value.py`
- `Value = Number | parametric.Parametric | Interval`.
- `is_zero(value)`: branch `if parametric.is_parametric(value): return parametric.is_zero(value)`. **No backend-specific import here.**
- `value_to_string`: delegates to `parametric.to_str(value)` for parametric values.

### 4.3 `stormvogel/model/distribution.py`
- `is_stochastic`: the current check `isinstance(v, (Interval, Parametric))` becomes `isinstance(v, Interval) or parametric.is_parametric(v)`, returning `True` trivially for parametric distributions (same behaviour as before).
- `__add__` already uses `+=`, which works on all sensible backends.

### 4.4 `stormvogel/model/model.py`
- `is_parametric`: `parametric.is_parametric(value)`.
- `get_instantiated_model`: `new_distr[target] = parametric.evaluate(val, values)`. Keep the current deep-copy shape for now (see §6 for a potential follow-up).
- Parameter storage: see §4.4.1 below.

#### 4.4.1 Parameter storage

The current implementation recomputes `model.parameters` on every access by walking all transitions. We move to storing an ordered mapping on the model:

```python
class Model:
    ...
    def __init__(self, ...):
        ...
        # name -> backend-native symbol (sp.Symbol today, pycarl.Variable later)
        self._parameters: dict[str, Parametric] = {}
```

API:

```python
def declare_parameter(self, name: str, **kwargs) -> Parametric:
    """Return the symbol for `name`, creating it on first use.

    Extra keyword arguments are forwarded to the backend's symbol factory
    (e.g. sympy assumptions `positive=True, real=True`). Redeclaring with
    different kwargs raises.
    """

@property
def parameters(self) -> tuple[str, ...]:
    """Parameter names, in declaration order. Deterministic."""

def parameter_symbols(self) -> dict[str, Parametric]:
    """Insertion-ordered mapping from name to backend-native symbol."""

def unused_parameters(self) -> set[str]:
    """Declared parameters that no transition currently references."""

def prune_parameters(self) -> None:
    """Drop parameters not referenced by any transition."""
```

Motivation and properties:

- **Deterministic ordering.** `model.parameters` is used in `stormvogel_to_stormpy` to build pycarl variables; Python `set` iteration is hash-ordered, so the current code is potentially non-deterministic across runs. A `dict`-backed store fixes this at no cost.
- **Symbol identity.** The model owns exactly one `sp.Symbol` (or future `pycarl.Variable`) per name, so assumptions (`positive=True`, `real=True`) attach once and transitions that reference the same name reference the same object. This also matters for pycarl, which matches variables by identity in its pool — having the model own the pool mapping is the natural design.
- **No forced pre-declaration.** `set_choices` / `add_choices` automatically call `declare_parameter(name)` for any free symbol in a transition whose name is not yet known. That preserves the `bird.build_bird` workflow: users keep writing `sp.Symbol("x")` (or `model.declare_parameter("x")`) and the right thing happens.
- **No cache-invalidation sprawl.** Unlike `_is_parametric`, the parameters store is append-only on transition edits; only `prune_parameters` removes entries, and only when the user asks. So we don't need to touch all five `self._is_parametric = None` sites.
- **Authoritative source for the stormpy bridge.** §4.5 now reads `model.parameter_symbols()` directly — the bridge no longer re-walks transitions to collect parameter names, and pycarl `Variable`s are created in the deterministic order of the dict.

Edge case: a transition that used parameter `q` which is then replaced by a non-parametric transition leaves `q` in `_parameters` as "declared but unused". `unused_parameters()` surfaces this; `prune_parameters()` cleans it up on demand. No automatic pruning on mutation.

### 4.5 `stormvogel/stormpy_utils/stormvogel_to_stormpy.py`
The bridge consumes `model.parameter_symbols()` directly — an ordered `dict[str, Parametric]` — so pycarl variables are created in deterministic declaration order:

```python
stormpy.pycarl.clear_variable_pool()
var_map: dict[str, stormpy.pycarl.Variable] = {
    name: stormpy.pycarl.Variable(name)
    for name in model.parameter_symbols()
}
```

`value_to_stormpy` then delegates the parametric case to the value's backend:

```python
if model.is_parametric():
    if isinstance(value, Number):
        return _number_to_factorized_rational(value)
    backend = parametric.backend_for(value)     # looks the backend up by type
    return backend.to_pycarl(value, var_map)
```

The sympy backend's `to_pycarl` builds the pycarl polynomial via `sp.Poly(expr, *symbols).terms()` (exact rational coefficients via `stormpy.pycarl.cln.Rational(sp.Rational(c))`), using `model.parameter_symbols().values()` as the *ordered* symbol list so exponent tuples line up with the pycarl variable map. A future pycarl backend's `to_pycarl` is the identity (plus factorization wrapping).

### 4.6 `stormvogel/stormpy_utils/mapping.py`
`value_to_stormvogel` delegates to the default backend's `from_pycarl`:

```python
if sparsemodel.has_parameters:
    rf = value.rational_function()
    return parametric.get_default().from_pycarl(rf)
```

The sympy backend's `from_pycarl` parses the pycarl `str(...)` via `sp.sympify(s.replace("^", "**"), locals={name: sp.Symbol(name) for name in …})` — controlled `local_dict` avoids name clashes with sympy builtins (`E`, `I`, `S`, `N`, `O`, `Q`). If the denominator is 1 and the numerator is constant, it returns a plain `float`.

### 4.7 `stormvogel/result.py` and `stormvogel/stormpy_utils/convert_results.py`
- In `result.maximum`, the branch that raises for parametric results checks `parametric.is_parametric(v)` instead of `isinstance(v, Parametric)`.
- `convert_results.py` does not touch Parametric internals; only the type narrowing changes.

### 4.8 `stormvogel/simulator.py`
Any `isinstance(..., Parametric)` becomes `parametric.is_parametric(...)`. The simulator already errors on symbolic transitions; no behaviour change.

### 4.9 Docs: `docs/5-Advanced/1-Parametric-and-interval-models.py`
Rewrite the parametric section to the idiomatic form:

```python
import sympy as sp
from stormvogel import model, bird
from stormvogel.show import show

x = sp.symbols("x")
invx = 1 - x

def delta(s):
    ...
```

`get_instantiated_model({"x": 1/2})` keeps its current string-keyed signature. No migration note is needed — this tutorial was preliminary and unstable.

### 4.9a Docs: `docs/2-Orchard-Tutorial/4-uncertainty.py`
This is the bigger win. The current code is:

```python
one = stormvogel.parametric.Polynomial([])
one.add_term((0,), 1)

params = [f"p{i}" for i in range(2)]
parameters = [stormvogel.parametric.Polynomial([p]) for p in params]
for i in range(2):
    parameters[i].add_term((1,), 1)

parameters.append(stormvogel.parametric.Polynomial(params))
parameters[-1].add_term((0, 1), -1)
parameters[-1].add_term((1, 0), -2)
parameters[-1].add_term((0, 0), 1)
```

with a trailing note "(a parser is planned)". The rewrite is:

```python
import sympy as sp

p0, p1 = sp.symbols("p0 p1")
one = sp.Integer(1)                  # or just 1; the Value type accepts both
parameters = [p0, p0, p1, 1 - 2*p0 - p1]
```

The `delta_pmc` body is otherwise unchanged. The "(a parser is planned)" caveat is removed; sympy *is* the parser.

### 4.10 Tests
- `tests/test_parametric.py`: rewritten to construct sympy expressions directly; asserts unchanged in intent.
- `tests/test_value_to_string.py`, `tests/test_distribution.py`, `tests/test_model_methods.py`: adjusted to use sympy constructions where they referenced `Polynomial`.
- New test cases:
  - Round-trip `sp.Expr → pycarl → sp.Expr` produces an expression equal under `sp.simplify(... - ...) == 0` (structural equality is too strict after `Poly` normalisation).
  - `get_instantiated_model` on a rational-function transition.
  - `Model.parameters` on a model whose transitions mix constants, polynomials, and rational functions.
  - `is_stochastic` returns `True` for parametric distributions whose symbolic sum is not literally `1` (matches current trivial-True behaviour).

### 4.11 Example: `stormvogel/examples/knuth_yao_pmc.py`
Port to sympy in the same style as the tutorial.

## 5. Migration and compatibility

- **No shim.** Importing `stormvogel.parametric.Polynomial` raises `AttributeError` after the change. The existing parametric tutorials were preliminary and unstable, so no migration paragraph is included in user-facing docs — both tutorials simply show the new sympy idiom.
- **Pickle / persisted models.** The `model.html` / `.umb` formats do not serialize parametric values today (transitions are rendered as strings), so no data format change is needed. Worth grepping once to confirm.
- **stormpy optional.** The pycarl bridge remains gated on `try: import stormpy`. The sympy code path runs whether stormpy is present or not, so the non-stormpy test subset gets strictly better coverage.
- **Future pycarl backend.** A new file `stormvogel/parametric/pycarl_backend.py` can register pycarl types with the same singledispatch generics and implement the `ParametricBackend` protocol. The stormpy bridge will then pass pycarl values through without conversion, and `parametric.set_default("pycarl")` selects it for construction. This plan does not include that file, but everything above is shaped to admit it.

## 6. Out of scope (explicitly)

These are good follow-ups but not part of this change:

- Constraining symbols (e.g. `sp.symbols("x", positive=True, real=True)`) for stronger simplification of transition probabilities.
- Well-formedness checks on parametric distributions (e.g. "the sum of outgoing probabilities equals 1 as a symbolic identity after simplification").
- A region-checking helper on top of stormpy's parameter region API.
- Parametric reward models (already supported since rewards flow through the same `Value` type, but no dedicated tests yet).
- Avoiding `deepcopy` in `Model.get_instantiated_model`; we preserve the current semantics here.

## 7. Implementation plan

Proposed on a feature branch `feat/parametric-sympy`:

1. Introduce the `stormvogel/parametric/` package with the backend scaffolding and the sympy implementation (§3, §4.1). Delete the old `parametric.py`.
2. Update `stormvogel/model/value.py` (§4.2). Downstream call sites (`distribution.py`, `model.py`, `result.py`, `simulator.py`, `convert_results.py`) switch from concrete `isinstance` checks to `parametric.is_parametric` / `parametric.free_symbol_names` / `parametric.evaluate` (§4.3–§4.8). No pycarl changes yet, so this compiles and runs with stormpy absent.
3. Rewrite the stormpy bridge (§4.5, §4.6). Both directions go through backend methods, gated behind `try: import stormpy`.
4. Rewrite tests (§4.10). Run `pytest -q` locally; expect the stormpy-dependent tests to be skipped unless pycarl is installed. Add a test that stubs a second backend to confirm dispatch works independently of sympy.
5. Port `docs/5-Advanced/1-Parametric-and-interval-models.py` and `docs/2-Orchard-Tutorial/4-uncertainty.py` and `stormvogel/examples/knuth_yao_pmc.py` (§4.9, §4.9a, §4.11).
6. Open a PR; you decide on push (per the project instruction).

## 8. Risks

- **`sp.Poly` variable ordering** — we must always pass `*symbols` from `model.parameters` in a stable, sorted order so that the pycarl variable map matches. The plan does this.
- **`sp.sympify` on pycarl strings** — pycarl may emit Python-reserved-like tokens (e.g. a variable called `E` or `I` would clash with sympy constants). Mitigation: build a `local_dict = {name: sp.Symbol(name) for name in model.parameters}` and pass `locals=local_dict, evaluate=False` to `sympify`. This is cheap and robust.
- **Fraction preservation** — `sp.Rational(Fraction(...))` works; `sp.Rational(float)` can introduce tiny denominators. In `value_to_stormpy` we route via `sp.Rational(coeff, ...)` to stay exact.
- **Performance** — sympy is slower than the dict-of-terms representation for enormous models. The backend abstraction means a future pycarl-native backend can sidestep sympy entirely for performance-sensitive workflows without touching the rest of the codebase.
- **Dispatch on `sp.Expr` subclasses** — `singledispatch` uses MRO, so registering against `sp.Expr` covers all sympy expressions (`Add`, `Mul`, `Pow`, `Symbol`, `Rational`, …). We must double-check that plain `sp.Integer` / `sp.Rational` values (which are `sp.Expr` too) work for the "effectively constant" case — the planned `is_parametric` / `is_zero` / `evaluate` implementations all handle this.

---

Once this reads right to you, I'll cut `feat/parametric-sympy` and start with step 1.
