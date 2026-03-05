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
    from stormvogel.parametric import Polynomial

    p1 = Polynomial(["p"])
    p2 = Polynomial(["q"])
    d = Distribution([(p1, "a"), (p2, "b")])
    assert d.is_stochastic()


def test_is_not_stochastic():
    d = Distribution([(0.5, "a"), (0.4, "b")])
    assert not d.is_stochastic()


def test_distribution_properties():
    d = Distribution([(0.6, "a"), (0.4, "b")])
    assert d.support == {"a", "b"}
    assert d.values == [0.6, 0.4]
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
        _ = d + 1


def test_distribution_str():
    d = Distribution([(0.5, "a"), (0.5, "b")])
    assert str(d) == "0.5 -> a, 0.5 -> b"


def test_distribution_iter():
    d = Distribution([(0.5, "a"), (0.5, "b")])
    items = list(d)
    assert items == [(0.5, "a"), (0.5, "b")]


def test_distribution_sort():
    class MockModel:
        def __init__(self, states):
            self.states = states

    class MockState:
        def __init__(self, name, model):
            self.name = name
            self.model = model

    states = []
    model = MockModel(states)

    s1 = MockState("s1", model)
    s2 = MockState("s2", model)
    s3 = MockState("s3", model)

    # the model.states order is s1, s2, s3
    states.extend([s1, s2, s3])

    d = Distribution([(0.3, s3), (0.2, s1), (0.5, s2)])
    d.sort()

    supports = [s for _, s in d.distribution]
    assert supports == [s1, s2, s3]
