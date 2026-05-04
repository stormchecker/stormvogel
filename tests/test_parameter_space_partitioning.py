"""Tests for teaching.parametric.parameter_space_partitioning."""

from fractions import Fraction

import pytest

from stormvogel.examples.knuth_yao_pmc import (
    create_knuth_yao_pmc,
    create_knuth_yao_pmc_twocoins,
)
from stormvogel.parametric.region import AnnotatedRegion, RectangularRegion
from stormvogel.teaching.parametric import parameter_space_partitioning

stormpy = pytest.importorskip("stormpy")

_PROP = 'P=? [F "rolled1"]'
_THRESHOLD = 1 / 6


@pytest.fixture(scope="module")
def pmc():
    return create_knuth_yao_pmc()


@pytest.fixture(scope="module")
def pmc_xy():
    return create_knuth_yao_pmc_twocoins()


# ---------------------------------------------------------------------------
# Single-coin (one parameter x)
# ---------------------------------------------------------------------------


def test_returns_list(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert isinstance(result, list)


def test_all_items_are_annotated_regions(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert all(isinstance(r, AnnotatedRegion) for r in result)


def test_decided_regions_have_correct_label(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        assert ar.classify(_THRESHOLD) in ("safe", "unsafe", "unknown", "neither")


def test_safe_regions_classify_safe(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        if ar.classify(_THRESHOLD) == "safe":
            assert ar.min_value[0] >= _THRESHOLD


def test_unsafe_regions_classify_unsafe(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=10)
    for ar in result:
        if ar.classify(_THRESHOLD) == "unsafe":
            assert ar.max_value[1] < _THRESHOLD


def test_more_splits_produce_more_regions(pmc):
    few = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    many = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=20)
    assert len(many) >= len(few)


def test_default_initial_region(pmc):
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=4)
    assert len(result) > 0


def test_explicit_initial_region(pmc):
    region = RectangularRegion({"x": (Fraction(1, 10), Fraction(9, 10))})
    result = parameter_space_partitioning(
        pmc, _PROP, _THRESHOLD, initial_region=region, max_iterations=4
    )
    assert len(result) > 0


def test_raises_for_non_graph_preserving_region(pmc):
    bad = RectangularRegion({"x": (Fraction(0), Fraction(9, 10))})
    with pytest.raises(ValueError, match="graph-preserving"):
        parameter_space_partitioning(pmc, _PROP, _THRESHOLD, initial_region=bad)


def test_zero_iterations_returns_annotated_initial_region(pmc):
    # No splits allowed: initial region is annotated and collected as-is.
    result = parameter_space_partitioning(pmc, _PROP, _THRESHOLD, max_iterations=0)
    assert len(result) == 1
    assert isinstance(result[0], AnnotatedRegion)


# ---------------------------------------------------------------------------
# Two-coin variant (parameters x and y)
# ---------------------------------------------------------------------------


def test_twocoins_returns_list(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert isinstance(result, list)


def test_twocoins_all_items_are_annotated_regions(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert all(isinstance(r, AnnotatedRegion) for r in result)


def test_twocoins_default_region_covers_both_params(pmc_xy):
    result = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    assert len(result) > 0
    for ar in result:
        assert "x" in ar.region.bounds
        assert "y" in ar.region.bounds


def test_twocoins_explicit_2d_region(pmc_xy):
    region = RectangularRegion(
        {
            "x": (Fraction(1, 10), Fraction(9, 10)),
            "y": (Fraction(1, 10), Fraction(9, 10)),
        }
    )
    result = parameter_space_partitioning(
        pmc_xy, _PROP, _THRESHOLD, initial_region=region, max_iterations=4
    )
    assert len(result) > 0


def test_twocoins_more_splits_produce_more_regions(pmc_xy):
    few = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=4)
    many = parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, max_iterations=20)
    assert len(many) >= len(few)


def test_twocoins_raises_for_non_graph_preserving_region(pmc_xy):
    bad = RectangularRegion(
        {"x": (Fraction(0), Fraction(9, 10)), "y": (Fraction(1, 10), Fraction(9, 10))}
    )
    with pytest.raises(ValueError, match="graph-preserving"):
        parameter_space_partitioning(pmc_xy, _PROP, _THRESHOLD, initial_region=bad)
