"""Backend registry for :mod:`stormvogel.parametric`.

A *backend* is anything that owns a concrete value type representing a
polynomial / rational function, knows how to construct its symbols and
constants, and can move values to and from pycarl (the representation
stormpy uses internally).

Backends are registered with :func:`register`. The first backend to register
also becomes the default; callers can change that with :func:`set_default`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a hard pycarl dependency at import time.
    pass


Number = int | float | Fraction


@runtime_checkable
class ParametricBackend(Protocol):
    """The contract a parametric-value backend must implement.

    A backend is responsible for *constructing* its values (symbols,
    constants) and for *bridging* them to pycarl. All *introspection*
    operations (``is_zero``, ``evaluate``, …) are registered as
    :func:`functools.singledispatch` overloads on the generics in
    :mod:`stormvogel.parametric`; the backend itself does not need to expose
    them on its instance.
    """

    #: Short name, e.g. ``"sympy"`` or ``"pycarl"``.
    name: str

    #: Concrete types this backend owns. Used for :func:`is_parametric` and
    #: :func:`backend_for` dispatch.
    expr_types: tuple[type, ...]

    def symbol(self, name: str, **kwargs: Any) -> Any:
        """Create a parameter symbol with the given name."""
        ...

    def constant(self, n: Number) -> Any:
        """Lift a Python number into a backend-native constant."""
        ...

    def to_pycarl(self, value: Any, var_map: dict[str, Any]) -> Any:
        """Convert ``value`` to a pycarl factorized rational function.

        ``var_map`` maps parameter names to the pycarl ``Variable`` objects
        created by the stormpy bridge. Callers guarantee that every free
        symbol in ``value`` has an entry.
        """
        ...

    def from_pycarl(self, pycarl_value: Any) -> Any:
        """Convert a pycarl rational function into a backend-native value."""
        ...


_BACKENDS: list[ParametricBackend] = []
_default: ParametricBackend | None = None


def register(backend: ParametricBackend) -> None:
    """Register a parametric backend.

    If no default has been set yet, the first backend registered becomes it.
    Re-registering the same backend (by ``name``) is a no-op — the existing
    entry is replaced in-place — so importing a backend module twice is
    harmless.
    """
    global _default
    for i, existing in enumerate(_BACKENDS):
        if existing.name == backend.name:
            _BACKENDS[i] = backend
            if _default is existing:
                _default = backend
            return
    _BACKENDS.append(backend)
    if _default is None:
        _default = backend


def set_default(name: str) -> None:
    """Select the backend used by :func:`stormvogel.parametric.symbol` /
    :func:`stormvogel.parametric.constant`."""
    global _default
    for backend in _BACKENDS:
        if backend.name == name:
            _default = backend
            return
    raise LookupError(
        f"No parametric backend registered with name {name!r}. "
        f"Registered: {[b.name for b in _BACKENDS]}"
    )


def get_default() -> ParametricBackend:
    """Return the currently-selected default backend.

    :raises RuntimeError: If no backend has been registered yet (should not
        happen in practice: importing :mod:`stormvogel.parametric`
        auto-registers the sympy backend).
    """
    if _default is None:
        raise RuntimeError(
            "No parametric backend registered. Import stormvogel.parametric "
            "or a custom backend before using this API."
        )
    return _default


def registered_types() -> tuple[type, ...]:
    """Return the union of all ``expr_types`` across registered backends.

    Used by :func:`stormvogel.parametric.is_parametric` as the
    :func:`isinstance` check.
    """
    types: list[type] = []
    for b in _BACKENDS:
        for t in b.expr_types:
            if t not in types:
                types.append(t)
    return tuple(types)


def backend_for(value: Any) -> ParametricBackend:
    """Return the backend that owns ``value``'s concrete type.

    :raises LookupError: If no registered backend claims the type.
    """
    for b in _BACKENDS:
        if isinstance(value, b.expr_types):
            return b
    raise LookupError(
        f"No parametric backend registered for type {type(value).__name__!r}. "
        f"Registered: {[b.name for b in _BACKENDS]}"
    )
