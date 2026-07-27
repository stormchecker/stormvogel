"""Tests for stormvogel.examples.premise_monitoring.

The PREMISE models are not shipped with stormvogel, so these run against the
POMDP that stormpy bundles.  Tests that need the benchmark itself skip unless
``STORMVOGEL_PREMISE_DIR`` points at a checkout.
"""

import os
from fractions import Fraction

import pytest

stormpy = pytest.importorskip("stormpy")

import stormpy.examples.files as stormpy_files  # noqa: E402

from stormvogel.examples.premise_monitoring import (  # noqa: E402
    PREMISE_MODELS,
    load_monitoring_hmm,
    monitoring_report,
    monitoring_trace,
)
from stormvogel.model.model import ModelType  # noqa: E402

PREMISE_DIR = os.environ.get("STORMVOGEL_PREMISE_DIR")
needs_premise = pytest.mark.skipif(
    PREMISE_DIR is None, reason="set STORMVOGEL_PREMISE_DIR to run on the benchmark"
)


@pytest.fixture(scope="module")
def maze():
    """stormpy's bundled maze POMDP, loaded as an HMM."""
    return load_monitoring_hmm(stormpy_files.prism_pomdp_maze)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_gives_an_hmm(maze):
    assert maze.model_type == ModelType.HMM
    assert maze.nr_states > 0


def test_every_state_has_exactly_one_choice(maze):
    """Uniformising leaves nothing to choose."""
    for state in maze.states:
        assert len(list(maze.transitions[state])) == 1


def test_distributions_sum_to_one(maze):
    for state in maze.states:
        _action, branch = next(iter(maze.transitions[state]))
        assert sum(float(p) for p, _ in branch) == pytest.approx(1.0)


def test_observations_are_shared_between_states(maze):
    """An HMM worth searching has fewer observations than states."""
    assert 0 < len(maze.observation_aliases) < maze.nr_states


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def test_trace_reaches_the_threshold(maze):
    label = next(iter(maze.state_labels))
    result = monitoring_trace(maze, label, Fraction(1, 2))
    if result.found:
        assert result.final_target_probability >= Fraction(1, 2)
        assert Fraction(0) < result.probability <= Fraction(1)


def test_trace_probability_is_the_product_of_its_steps(maze):
    label = next(iter(maze.state_labels))
    result = monitoring_trace(maze, label, Fraction(1, 2))
    product = Fraction(1)
    for step in result.steps:
        product *= step.probability
    if result.found:
        assert product == result.probability


def test_hmm_steps_have_no_action(maze):
    """There is no action to report in an HMM."""
    label = next(iter(maze.state_labels))
    result = monitoring_trace(maze, label, Fraction(1, 2))
    assert all(step.action == "" for step in result.steps)


def test_report_is_readable(maze):
    label = next(iter(maze.state_labels))
    report = monitoring_report(maze, label, Fraction(1, 2), step_bound=3)
    assert f"P({label}) >=" in report


def test_report_says_so_when_unreachable(maze):
    """No belief can be sure of a label no state carries."""
    maze.add_label("no_state_has_this")
    report = monitoring_report(maze, "no_state_has_this", Fraction(1, 2))
    assert "not reachable" in report


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


def test_premise_table_is_complete():
    assert len(PREMISE_MODELS) == 6
    for model in PREMISE_MODELS:
        assert model.filename.endswith(".nm")
        assert model.good_label in ("empty", "crash")
        assert model.step_bound > 0


@needs_premise
@pytest.mark.parametrize("model", PREMISE_MODELS, ids=lambda m: m.filename)
def test_premise_models_reach_ninety_percent(model):
    assert PREMISE_DIR is not None
    hmm = load_monitoring_hmm(f"{PREMISE_DIR}/{model.filename}", model.constants)
    result = monitoring_trace(hmm, model.good_label, Fraction(9, 10))
    assert result.found
    assert result.final_target_probability >= Fraction(9, 10)
