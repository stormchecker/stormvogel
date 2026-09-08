"""Tests for Model.to_dot, verified by running the graphviz ``dot`` binary.

These tests only run when graphviz is installed; the dot output itself is
produced without any optional dependency.
"""

import shutil
import subprocess

import pytest

import stormvogel.model
from stormvogel.examples.die import create_die_dtmc
from stormvogel.examples.monty_hall import create_monty_hall_mdp
from stormvogel.examples.nuclear_fusion_ctmc import create_nuclear_fusion_ctmc

pytestmark = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz 'dot' not found in PATH"
)


def render(dot: str) -> str:
    """Run graphviz on a dot string and return the rendered svg."""
    result = subprocess.run(
        ["dot", "-Tsvg"],
        input=dot,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"graphviz rejected the dot output:\n{result.stderr}\n\n{dot}"
    return result.stdout


@pytest.mark.parametrize(
    "create_model",
    [create_die_dtmc, create_monty_hall_mdp, create_nuclear_fusion_ctmc],
    ids=["dtmc", "mdp", "ctmc"],
)
def test_to_dot_is_valid_dot(create_model):
    render(create_model().to_dot())


def test_to_dot_uses_friendly_names():
    model = stormvogel.model.new_dtmc()
    init = model.initial_state
    init.set_friendly_name("start here")
    other = model.new_state(labels=["done"], friendly_name="the end")
    model.set_choices(init, [(1, other)])
    model.set_choices(other, [(1, other)])

    dot = model.to_dot()
    assert '"start here" -> "the end"' in dot
    svg = render(dot)
    assert "start here" in svg
    assert "the end" in svg


def test_to_dot_actions_are_valid_dot():
    """Action labels with spaces need quoting, both as nodes and as edge labels."""
    model = stormvogel.model.new_mdp()
    init = model.initial_state
    target = model.new_state(labels=["target"])
    model.set_choices(
        init,
        {
            model.action("go left"): [(0.5, init), (0.5, target)],
            model.action("go right"): [(1, target)],
        },
    )
    model.set_choices(target, {model.action("stay"): [(1, target)]})

    dot = model.to_dot()
    assert '"0_go left"' in dot
    render(dot)
