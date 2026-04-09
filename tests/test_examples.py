"""Smoke-tests all example builder functions: each must return a non-empty model."""

import importlib
import inspect
import pkgutil

import stormvogel.examples


def _example_builders():
    """Yield (name, callable) for every create_* / example_* function in stormvogel.examples."""
    pkg = stormvogel.examples
    for _, modname, _ in pkgutil.walk_packages(
        path=pkg.__path__,
        prefix=pkg.__name__ + ".",
        onerror=lambda x: None,
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith(("create_", "example_")) and obj.__module__ == modname:
                yield f"{modname}.{name}", obj


def test_all_examples_return_nonempty_model():
    builders = list(_example_builders())
    assert builders, "No example builder functions found"
    for qualname, fn in builders:
        try:
            result = fn()
        except Exception as e:
            raise AssertionError(f"{qualname} raised {type(e).__name__}: {e}") from e
        assert result is not None, f"{qualname} returned None"
        assert result.nr_states > 0, f"{qualname} returned a model with 0 states"
