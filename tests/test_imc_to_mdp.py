"""Tests for stormvogel.transformations.imc_to_mdp."""

import pytest
from fractions import Fraction

import stormvogel.model as sv_model
from stormvogel.model.action import EmptyAction
from stormvogel.model.choices import Choices
from stormvogel.model.distribution import Distribution
from stormvogel.model.value import Interval
from stormvogel.transformations.imc_to_mdp import _vertices, imc_to_mdp


# ---------------------------------------------------------------------------
# _vertices unit tests
# ---------------------------------------------------------------------------


def test_vertices_point_interval_single():
    """Single-entry point interval has exactly one vertex: (1,)."""
    vs = _vertices([Fraction(1)], [Fraction(1)])
    assert vs == [(Fraction(1),)]


def test_vertices_two_symmetric():
    """[0.3, 0.7] x [0.3, 0.7] yields exactly two vertices."""
    lows = [Fraction(3, 10), Fraction(3, 10)]
    highs = [Fraction(7, 10), Fraction(7, 10)]
    vs = _vertices(lows, highs)
    assert len(vs) == 2
    assert (Fraction(7, 10), Fraction(3, 10)) in vs
    assert (Fraction(3, 10), Fraction(7, 10)) in vs


def test_vertices_degenerate_two():
    """Point intervals yield a single vertex."""
    lows = [Fraction(2, 5), Fraction(3, 5)]
    highs = [Fraction(2, 5), Fraction(3, 5)]
    vs = _vertices(lows, highs)
    assert len(vs) == 1
    assert vs[0] == (Fraction(2, 5), Fraction(3, 5))


def test_vertices_three_entries_sum_to_one():
    """All vertices of a three-entry distribution sum to 1."""
    lows = [Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    highs = [Fraction(3, 5), Fraction(3, 5), Fraction(3, 5)]
    vs = _vertices(lows, highs)
    assert len(vs) > 0
    for v in vs:
        assert sum(v) == Fraction(1)


def test_vertices_each_in_bounds():
    """Every coordinate of every vertex lies within its interval."""
    lows = [Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    highs = [Fraction(3, 5), Fraction(3, 5), Fraction(3, 5)]
    vs = _vertices(lows, highs)
    for v in vs:
        for i, (lo, hi) in enumerate(zip(lows, highs)):
            assert lo <= v[i] <= hi


def test_vertices_no_duplicates():
    """_vertices never returns the same tuple twice."""
    lows = [Fraction(1, 5), Fraction(1, 5), Fraction(1, 5)]
    highs = [Fraction(3, 5), Fraction(3, 5), Fraction(3, 5)]
    vs = _vertices(lows, highs)
    assert len(vs) == len(set(vs))


# ---------------------------------------------------------------------------
# Helper: build a small interval MC
# ---------------------------------------------------------------------------
#
# init --[0.3,0.7]--> a
#      --[0.3,0.7]--> b
# a    --[1,1]--> a   (absorbing)
# b    --[1,1]--> b   (absorbing)


def _make_simple_imc():
    imc = sv_model.new_dtmc(create_initial_state=False)
    init = imc.new_state(["init"])
    a = imc.new_state(["a"])
    b = imc.new_state(["b"])

    imc.transitions[init] = Choices(
        {
            EmptyAction: Distribution(
                {
                    a: Interval(Fraction(3, 10), Fraction(7, 10)),
                    b: Interval(Fraction(3, 10), Fraction(7, 10)),
                }
            )
        }
    )
    imc.transitions[a] = Choices(
        {EmptyAction: Distribution({a: Interval(Fraction(1), Fraction(1))})}
    )
    imc.transitions[b] = Choices(
        {EmptyAction: Distribution({b: Interval(Fraction(1), Fraction(1))})}
    )
    return imc, init, a, b


# ---------------------------------------------------------------------------
# imc_to_mdp structural tests
# ---------------------------------------------------------------------------


def test_imc_to_mdp_returns_mdp():
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    assert mdp.model_type == sv_model.ModelType.MDP


def test_imc_to_mdp_state_count_preserved():
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    assert mdp.nr_states == imc.nr_states


def test_imc_to_mdp_labels_preserved():
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    assert "init" in mdp.state_labels
    assert "a" in mdp.state_labels
    assert "b" in mdp.state_labels


def test_imc_to_mdp_init_has_two_actions():
    """init has two actions (one per vertex of its interval distribution)."""
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    init_new = next(iter(mdp.state_labels["init"]))
    assert len(mdp.transitions[init_new]) == 2


def test_imc_to_mdp_absorbing_has_one_action():
    """Point-interval absorbing state produces exactly one action."""
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    a_new = next(iter(mdp.state_labels["a"]))
    assert len(mdp.transitions[a_new]) == 1


def test_imc_to_mdp_vertex_distributions_sum_to_one():
    """Every action distribution in the MDP sums to 1."""
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    for _, choices in mdp.transitions.items():
        for _, branch in choices:
            total = sum(p for p, _ in branch)
            assert abs(float(total) - 1.0) < 1e-12


def test_imc_to_mdp_vertex_probs_are_fractions():
    """Vertex probabilities are exact Fractions."""
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    for _, choices in mdp.transitions.items():
        for _, branch in choices:
            for prob, _ in branch:
                assert isinstance(prob, Fraction)


def test_imc_to_mdp_raises_for_non_interval():
    dtmc = sv_model.new_dtmc()
    with pytest.raises(ValueError, match="interval"):
        imc_to_mdp(dtmc)


def test_imc_to_mdp_state_rewards_preserved():
    imc, init, a, b = _make_simple_imc()
    rm = imc.new_reward_model("steps")
    rm.rewards[init] = 1
    rm.rewards[a] = 0
    rm.rewards[b] = 0

    mdp = imc_to_mdp(imc)
    assert len(mdp.rewards) == 1
    init_new = next(iter(mdp.state_labels["init"]))
    assert mdp.rewards[0].rewards[init_new] == 1


def test_imc_to_mdp_friendly_names_preserved():
    imc = sv_model.new_dtmc(create_initial_state=False)
    s = imc.new_state(["init"], friendly_name="my_state")
    imc.transitions[s] = Choices(
        {EmptyAction: Distribution({s: Interval(Fraction(1), Fraction(1))})}
    )
    mdp = imc_to_mdp(imc)
    init_new = next(iter(mdp.state_labels["init"]))
    assert mdp.friendly_names.get(init_new) == "my_state"


def test_imc_to_mdp_vertex_probabilities_match_expected():
    """The two vertices of init are (0.7, 0.3) and (0.3, 0.7) in some order."""
    imc, _, _, _ = _make_simple_imc()
    mdp = imc_to_mdp(imc)
    init_new = next(iter(mdp.state_labels["init"]))
    prob_sets = []
    for _, branch in mdp.transitions[init_new]:
        prob_sets.append(frozenset(p for p, _ in branch))
    # Both vertices have the same set of probabilities {0.7, 0.3}
    assert all(ps == frozenset({Fraction(7, 10), Fraction(3, 10)}) for ps in prob_sets)


def test_imc_to_mdp_fan_out_warning():
    """A state with >8 successors triggers a RuntimeWarning."""
    imc = sv_model.new_dtmc(create_initial_state=False)
    init = imc.new_state(["init"])
    successors = [imc.new_state([f"s{i}"]) for i in range(9)]

    total = Fraction(1)
    per = total / 9
    distr = {s: Interval(per, per) for s in successors}
    imc.transitions[init] = Choices({EmptyAction: Distribution(distr)})
    for s in successors:
        imc.transitions[s] = Choices(
            {EmptyAction: Distribution({s: Interval(Fraction(1), Fraction(1))})}
        )

    with pytest.warns(RuntimeWarning, match="fan-out"):
        imc_to_mdp(imc)
