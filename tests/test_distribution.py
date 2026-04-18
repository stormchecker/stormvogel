import pytest
from fractions import Fraction
from stormvogel.model.distribution import Distribution
from stormvogel.model.value import Interval


def test_is_stochastic_float():
    d = Distribution([(0.5, "a"), (0.5, "b")])
    assert d.is_stochastic()


def test_is_stochastic_int():
    d = Distribution([(1, "a")])
    assert d.is_stochastic()


def test_is_stochastic_fraction():
    d = Distribution([(Fraction(1, 2), "a"), (Fraction(1, 2), "b")])
    assert d.is_stochastic()


def test_is_stochastic_interval():
    d = Distribution([(Interval(0.1, 0.9), "a"), (Interval(0.1, 0.9), "b")])
    assert d.is_stochastic()


def test_is_stochastic_parametric():
    import sympy as sp

    p = sp.Symbol("p")
    q = sp.Symbol("q")
    d = Distribution([(p, "a"), (q, "b")])
    # For parametric distributions, is_stochastic is trivially True — the
    # probabilities do not even sum to a constant.
    assert d.is_stochastic()


def test_is_not_stochastic():
    d = Distribution([(0.5, "a"), (0.4, "b")])
    assert not d.is_stochastic()


def test_distribution_properties():
    d = Distribution([(0.6, "a"), (0.4, "b")])
    assert d.support == {"a", "b"}
    assert d.probabilities == [0.6, 0.4]
    assert len(d) == 2


def test_distribution_add():
    d1 = Distribution([(0.2, "a"), (0.5, "b")])
    d2 = Distribution([(0.3, "a"), (0.1, "c")])
    d_sum = d1 + d2
    assert d_sum.support == {"a", "b", "c"}

    # Convert to dict for easier value checking
    sum_dict = dict(zip((s for _, s in d_sum), (v for v, _ in d_sum)))
    assert sum_dict["a"] == 0.5
    assert sum_dict["b"] == 0.5
    assert sum_dict["c"] == 0.1


def test_distribution_add_type_error():
    d = Distribution([(1.0, "a")])
    with pytest.raises(TypeError):
        _ = d + 1  # type: ignore


def test_distribution_str():
    d = Distribution({"a": 0.5, "b": 0.5})
    assert str(d) == "Distribution({'a': 0.5, 'b': 0.5})"


def test_distribution_repr():
    d = Distribution({"a": 0.5, "b": 0.5})
    assert repr(d) == "Distribution({'a': 0.5, 'b': 0.5})"


def test_distribution_iter():
    d = Distribution([(0.5, "a"), (0.5, "b")])
    items = list(d)
    assert items == [(0.5, "a"), (0.5, "b")]


def test_distribution_from_distribution():
    """Distribution can be constructed from another Distribution."""
    d1 = Distribution([(0.3, "a"), (0.7, "b")])
    d2 = Distribution(d1)
    assert list(d2) == list(d1)
    # Must be a copy, not the same object
    d2["c"] = 0.1
    assert "c" not in d1


def test_distribution_invalid_type():
    """Distribution raises RuntimeError for unsupported input."""
    with pytest.raises(RuntimeError, match="Distribution expects"):
        Distribution(42)  # type: ignore


def test_distribution_eq_non_distribution():
    """Distribution.__eq__ returns NotImplemented when compared to a non-Distribution."""
    d = Distribution([(1.0, "a")])
    result = d.__eq__("not a distribution")
    assert result is NotImplemented


def test_distribution_add_merges_keys():
    """Distribution.__add__ when other is not a Distribution raises TypeError."""
    d1 = Distribution({"a": 0.5})
    d2 = Distribution({"b": 0.5})
    d3 = d1 + d2
    assert d3["a"] == 0.5
    assert d3["b"] == 0.5


def test_distribution_getitem():
    """Distribution.__getitem__ returns the value for a key."""
    d = Distribution([(0.4, "x"), (0.6, "y")])
    assert d["x"] == 0.4
    assert d["y"] == 0.6


def test_distribution_eq_equal_distributions():
    """Distribution.__eq__ returns True for two equal distributions."""
    d1 = Distribution([(0.5, "a"), (0.5, "b")])
    d2 = Distribution([(0.5, "a"), (0.5, "b")])
    assert d1 == d2


def test_distribution_eq_unequal_distributions():
    """Distribution.__eq__ returns False for different distributions."""
    d1 = Distribution([(0.5, "a")])
    d2 = Distribution([(0.5, "b")])
    assert d1 != d2
