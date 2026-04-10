import pytest

import stormvogel.examples as examples
import stormvogel.model
import stormvogel.teaching.bellman as bellman


@pytest.mark.parametrize(
    "input",
    [
        (examples.create_monty_hall_mdp(), "target"),
        (examples.create_lion_mdp(), "full"),
    ],
)
def test_teaching_bellman(input):
    pytest.importorskip("stormpy")
    bellman.maxreachprob(*input)


def _simple_mdp():
    """Two-state MDP: init --(go)--> full (self-loop)."""
    mdp = stormvogel.model.new_mdp()
    full = mdp.new_state("full")
    act = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {act: [(1.0, full)]})
    mdp.add_self_loops()
    return mdp


def test_zero_value():
    mdp = _simple_mdp()
    v = bellman.zero_value(mdp)
    assert all(val == 0 for val in v.values())
    assert set(v.keys()) == set(mdp.sorted_states)


def test_one_value():
    mdp = _simple_mdp()
    v = bellman.one_value(mdp)
    assert all(val == 1 for val in v.values())


def test_value_to_latex():
    mdp = _simple_mdp()
    values: dict[stormvogel.model.State, stormvogel.model.Value] = {
        s: float(i) for i, s in enumerate(mdp.sorted_states)
    }
    latex_lines = bellman.value_to_latex(values)
    assert len(latex_lines) == len(mdp.states)
    for line in latex_lines:
        assert "=" in line
