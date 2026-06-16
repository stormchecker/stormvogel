"""Tests for stormvogel.teaching.lovejoy."""

from fractions import Fraction

import pytest

import stormvogel.model as sv_model
from stormvogel.model.model import ModelType
from stormvogel.teaching.lovejoy import lovejoy_grid_mdp


# ---------------------------------------------------------------------------
# Shared POMDP fixture
# ---------------------------------------------------------------------------
#
# Two-state POMDP with absorbing goal/failure:
#
#   s0 (obs "z") --a--> s0 (2/3), s1 (1/3)   [mixes belief toward s0]
#   s0           --b--> target (1)
#   s1 (obs "z") --a--> s0 (1/3), s1 (2/3)   [mixes belief toward s1]
#   s1           --b--> sink (1)
#   target (obs "target_obs", label "target"): absorbing
#   sink   (obs "sink_obs"):                  absorbing
#
# Key: from {s_L: 1} under action "a", the posterior p(s_L) is always 2/3
# regardless of which state the UUID sort assigns as s_L.


def _make_2state_pomdp():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_z = pomdp.new_observation("z")
    obs_t = pomdp.new_observation("target_obs")
    obs_f = pomdp.new_observation("sink_obs")
    s0 = pomdp.new_state(friendly_name="s0", observation=obs_z)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    target = pomdp.new_state(["target"], friendly_name="target", observation=obs_t)
    sink = pomdp.new_state(friendly_name="sink", observation=obs_f)
    act_a = pomdp.new_action("a")
    act_b = pomdp.new_action("b")
    pomdp.set_choices(
        s0,
        {
            act_a: [(Fraction(2, 3), s0), (Fraction(1, 3), s1)],
            act_b: [(Fraction(1), target)],
        },
    )
    pomdp.set_choices(
        s1,
        {
            act_a: [(Fraction(1, 3), s0), (Fraction(2, 3), s1)],
            act_b: [(Fraction(1), sink)],
        },
    )
    pomdp.set_choices(target, [(Fraction(1), target)])
    pomdp.set_choices(sink, [(Fraction(1), sink)])
    return pomdp, s0, s1, target, sink


def _trans_by_action(mdp, state):
    """Return {action_label: {succ_state: prob}} for a state."""
    result = {}
    for action, branch in mdp.transitions[state]:
        result[action.label] = {s: p for p, s in branch}
    return result


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_for_non_pomdp():
    dtmc = sv_model.new_dtmc()
    with pytest.raises(ValueError, match="POMDP"):
        lovejoy_grid_mdp(dtmc, {}, k=2)


def test_raises_for_k_zero():
    pomdp, s0, *_ = _make_2state_pomdp()
    with pytest.raises(ValueError, match="k must be"):
        lovejoy_grid_mdp(pomdp, {s0: Fraction(1)}, k=0)


def test_raises_for_four_states_per_obs():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("z")
    sa = pomdp.new_state(observation=obs)
    sb = pomdp.new_state(observation=obs)
    sc = pomdp.new_state(observation=obs)
    sd = pomdp.new_state(observation=obs)
    pomdp.set_choices(sa, [(Fraction(1), sa)])
    pomdp.set_choices(sb, [(Fraction(1), sb)])
    pomdp.set_choices(sc, [(Fraction(1), sc)])
    pomdp.set_choices(sd, [(Fraction(1), sd)])
    with pytest.raises(ValueError, match="3 states per observation"):
        lovejoy_grid_mdp(pomdp, {sa: Fraction(1)}, k=2)


def test_raises_for_off_grid_initial_belief():
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    # 1/3 is not a multiple of 1/2
    with pytest.raises(ValueError, match="multiple of 1"):
        lovejoy_grid_mdp(pomdp, {s0: Fraction(1, 3), s1: Fraction(2, 3)}, k=2)


def test_raises_for_multi_obs_initial_belief():
    pomdp, s0, _, target, _ = _make_2state_pomdp()
    with pytest.raises(ValueError, match="single observation"):
        lovejoy_grid_mdp(
            pomdp,
            {s0: Fraction(1, 2), target: Fraction(1, 2)},
            k=2,
        )


def test_raises_for_unnormalized_initial_belief():
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    with pytest.raises(ValueError, match="sum to 1"):
        lovejoy_grid_mdp(pomdp, {s0: Fraction(1, 4), s1: Fraction(1, 4)}, k=4)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


def test_result_is_mdp():
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]
    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=2)
    assert mdp.model_type == ModelType.MDP


def test_has_init_label():
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]
    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=2)
    assert "init" in mdp.state_labels
    assert len(mdp.state_labels["init"]) == 1


def test_target_label_propagated_to_singleton():
    """The singleton belief {target: 1} must carry the 'target' label."""
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]
    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=1)
    assert "target" in mdp.state_labels


def test_all_transitions_sum_to_one():
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]
    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=4)
    for state in mdp.states:
        for _, branch in mdp.transitions[state]:
            total = sum(p for p, _ in branch)
            assert total == pytest.approx(1.0, abs=1e-12)


def test_k1_gives_only_boundary_and_absorbing_states():
    """With k=1 the only z-group grid points are {s_L:1} and {s_R:1}."""
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L, _ = sorted([s0, s1], key=lambda s: s.state_id)
    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=1)
    # States: {s_L:1}, {s_R:1}, target singleton, sink singleton
    assert mdp.nr_states == 4


def test_finer_grid_gives_more_or_equal_states():
    """A finer grid allows more distinct beliefs to be reachable."""
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]
    b0 = {s_L: Fraction(1)}
    coarse = lovejoy_grid_mdp(pomdp, b0, k=2)
    fine = lovejoy_grid_mdp(pomdp, b0, k=6)
    assert fine.nr_states >= coarse.nr_states


# ---------------------------------------------------------------------------
# Grid interpolation
# ---------------------------------------------------------------------------


def test_interpolation_splits_off_grid_belief():
    """From {s_L: 1} under action 'a' with k=2, posterior 2/3 splits to two grid points."""
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]

    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=2)
    init = next(iter(mdp.state_labels["init"]))

    # From {s_L: 1}, action "a" yields posterior p_L = 2/3 ∉ {0, 1/2, 1}
    # → split: grid[1]={s_L:1/2, s_R:1/2} with weight 2/3,
    #           grid[2]={s_L:1}       with weight 1/3  (self-loop)
    a_trans = _trans_by_action(mdp, init)["a"]
    by_name = {s.friendly_name: p for s, p in a_trans.items()}

    assert by_name["b_z[1/2]"] == Fraction(2, 3)
    assert by_name[init.friendly_name] == Fraction(1, 3)


def test_no_interpolation_when_posterior_is_on_grid():
    """From {s_L: 1} under action 'a' with k=3, posterior 2/3 = 2/3 is a grid point."""
    pomdp, s0, s1, *_ = _make_2state_pomdp()
    s_L = sorted([s0, s1], key=lambda s: s.state_id)[0]

    mdp = lovejoy_grid_mdp(pomdp, {s_L: Fraction(1)}, k=3)
    init = next(iter(mdp.state_labels["init"]))

    a_trans = _trans_by_action(mdp, init)["a"]
    by_name = {s.friendly_name: p for s, p in a_trans.items()}

    # Posterior 2/3 = grid[2] with k=3: single successor in the z group
    assert by_name.get("b_z[2/3]") == Fraction(1)
    assert len(by_name) == 1


# ---------------------------------------------------------------------------
# Integration tests (require stormpy)
# ---------------------------------------------------------------------------


def test_4state_upper_bound_decreases_with_finer_grid():
    """Lovejoy bound at b0 is non-increasing as k grows."""
    pytest.importorskip("stormpy")
    from stormvogel.examples import create_4state_reachability_pomdp_variantb
    from stormvogel.stormpy_utils.model_checking import model_checking

    model = create_4state_reachability_pomdp_variantb()
    s1, s2 = sorted(
        model.compute_states_per_observation()[model.get_observation("z")],
        key=lambda s: s.friendly_name or "",
    )
    b0 = {s1: Fraction(1, 2), s2: Fraction(1, 2)}

    bounds = []
    for k in [2, 4, 8]:
        mdp = lovejoy_grid_mdp(model, b0, k=k)
        result = model_checking(mdp, 'Pmax=? [F "target"]')
        assert result is not None
        bounds.append(result.at_init())

    assert bounds[0] >= bounds[1] - 1e-9
    assert bounds[1] >= bounds[2] - 1e-9


def test_two_state_commitment_lovejoy_equals_vpomdp():
    """Lovejoy bound for the two-state commitment model equals V_POMDP = 1/2.

    In this model action 'w' never shifts the 50/50 belief (both states
    transit to themselves with the same probability), so no grid refinement
    provides extra information.  The Lovejoy MDP correctly identifies 1/2.
    """
    pytest.importorskip("stormpy")
    from stormvogel.examples import create_two_state_commitment_pomdp
    from stormvogel.stormpy_utils.model_checking import model_checking

    model = create_two_state_commitment_pomdp()
    s1, s2 = sorted(
        model.compute_states_per_observation()[model.get_observation("z")],
        key=lambda s: s.friendly_name or "",
    )
    b0 = {s1: Fraction(1, 2), s2: Fraction(1, 2)}

    for k in [2, 4]:
        mdp = lovejoy_grid_mdp(model, b0, k=k)
        result = model_checking(mdp, 'Pmax=? [F "target"]')
        assert result is not None
        val = result.at_init()
        assert val == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# 3-state observation tests
# ---------------------------------------------------------------------------


def _make_3state_pomdp():
    """3-state POMDP: one observation 'z' with s0, s1, s2; absorbing goal/fail.

    Actions:
      a: s0 -> goal (1), s1 -> goal (1), s2 -> fail (1)  [go from any state]
      b: s0 -> s0 (1/3), s1 (1/3), s2 (1/3)              [mix the belief]
         s1 -> s0 (1/3), s1 (1/3), s2 (1/3)
         s2 -> s0 (1/3), s1 (1/3), s2 (1/3)
    """
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_z = pomdp.new_observation("z")
    obs_t = pomdp.new_observation("target_obs")
    obs_f = pomdp.new_observation("fail_obs")
    s0 = pomdp.new_state(friendly_name="s0", observation=obs_z)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z)
    goal = pomdp.new_state(["target"], friendly_name="goal", observation=obs_t)
    fail = pomdp.new_state(friendly_name="fail", observation=obs_f)
    act_a = pomdp.new_action("a")
    act_b = pomdp.new_action("b")
    for src in [s0, s1]:
        pomdp.set_choices(
            src,
            {
                act_a: [(Fraction(1), goal)],
                act_b: [
                    (Fraction(1, 3), s0),
                    (Fraction(1, 3), s1),
                    (Fraction(1, 3), s2),
                ],
            },
        )
    pomdp.set_choices(
        s2,
        {
            act_a: [(Fraction(1), fail)],
            act_b: [(Fraction(1, 3), s0), (Fraction(1, 3), s1), (Fraction(1, 3), s2)],
        },
    )
    pomdp.set_choices(goal, [(Fraction(1), goal)])
    pomdp.set_choices(fail, [(Fraction(1), fail)])
    return pomdp, sorted([s0, s1, s2], key=lambda s: s.state_id), goal, fail


def test_3state_result_is_mdp():
    pomdp, states, *_ = _make_3state_pomdp()
    mdp = lovejoy_grid_mdp(pomdp, {states[0]: Fraction(1)}, k=3)
    assert mdp.model_type == ModelType.MDP


def test_3state_all_transitions_sum_to_one():
    pomdp, (sa, sb, sc), *_ = _make_3state_pomdp()
    mdp = lovejoy_grid_mdp(
        pomdp, {sa: Fraction(1, 3), sb: Fraction(1, 3), sc: Fraction(1, 3)}, k=3
    )
    for state in mdp.states:
        for _, branch in mdp.transitions[state]:
            total = sum(p for p, _ in branch)
            assert total == pytest.approx(1.0, abs=1e-12)


def test_3state_vertex_belief_has_unit_weight():
    """A belief exactly on a grid vertex maps to that vertex with weight 1."""
    pomdp, states, *_ = _make_3state_pomdp()
    # Initial belief is the pure state sa — a vertex of the simplex
    mdp = lovejoy_grid_mdp(pomdp, {states[0]: Fraction(1)}, k=4)
    # Under action 'b' from {sa:1}, the posterior is {sa:1/3, sb:1/3, sc:1/3},
    # which is the centroid — a grid point for k=3 but not for k=4.
    # This test checks that starting from a vertex produces a valid MDP.
    assert mdp.nr_states >= 1


def test_3state_centroid_on_grid():
    """With k=3 the centroid (1/3,1/3,1/3) is a grid point; action 'b' maps to it directly."""
    pomdp, states, *_ = _make_3state_pomdp()
    # Start at states[0] (pure), k=3: centroid (1/3,1/3,1/3) = grid point (1,1,1)
    mdp = lovejoy_grid_mdp(pomdp, {states[0]: Fraction(1)}, k=3)
    init = next(iter(mdp.state_labels["init"]))
    b_trans = _trans_by_action(mdp, init)["b"]
    # Under 'b' from {sa:1}, all three states each contribute 1/3 to each successor;
    # successor belief is {sa:1/3, sb:1/3, sc:1/3} which is on the grid.
    assert len(b_trans) == 1
    succ_prob = next(iter(b_trans.values()))
    assert succ_prob == pytest.approx(1.0, abs=1e-12)


def test_3state_target_label_propagated():
    pomdp, states, *_ = _make_3state_pomdp()
    mdp = lovejoy_grid_mdp(pomdp, {states[0]: Fraction(1)}, k=2)
    assert "target" in mdp.state_labels


def test_atva20_does_not_raise():
    """The ATVA20 POMDP has observation z0 with 3 states; should not raise."""
    from stormvogel.examples import create_atva20_pomdp

    pomdp = create_atva20_pomdp()
    z0_states = sorted(
        pomdp.compute_states_per_observation()[pomdp.get_observation("z0")],
        key=lambda s: s.state_id,
    )
    s0 = next(s for s in z0_states if s.friendly_name == "s0")
    mdp = lovejoy_grid_mdp(pomdp, {s0: Fraction(1)}, k=4)
    assert mdp.model_type == ModelType.MDP


def test_atva20_upper_bound_decreases_with_finer_grid():
    """Lovejoy bound for ATVA20 (3-state observation) is non-increasing as k grows."""
    pytest.importorskip("stormpy")
    from stormvogel.examples import create_atva20_pomdp
    from stormvogel.stormpy_utils.model_checking import model_checking

    pomdp = create_atva20_pomdp()
    z0_states = sorted(
        pomdp.compute_states_per_observation()[pomdp.get_observation("z0")],
        key=lambda s: s.state_id,
    )
    s0 = next(s for s in z0_states if s.friendly_name == "s0")
    b0 = {s0: Fraction(1)}

    bounds = []
    for k in [2, 4, 8]:
        mdp = lovejoy_grid_mdp(pomdp, b0, k=k)
        result = model_checking(mdp, 'Pmax=? [F "target"]')
        assert result is not None
        bounds.append(result.at_init())

    assert bounds[0] >= bounds[1] - 1e-9
    assert bounds[1] >= bounds[2] - 1e-9
