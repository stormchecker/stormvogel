"""Tests for stormvogel.transformations.policy_to_pmc."""

import pytest
import sympy as sp

import stormvogel.model as sv_model
from stormvogel.model.model import ModelType
from stormvogel.transformations.policy_to_pmc import policy_to_pmc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mdp():
    """Two-state MDP.

    init --a--> a (prob 1)
         --b--> b (prob 1)
    a, b: absorbing (single self-loop action)
    """
    mdp = sv_model.new_mdp(create_initial_state=False)
    init = mdp.new_state(["init"], friendly_name="init")
    a = mdp.new_state(["a"], friendly_name="a")
    b = mdp.new_state(["b"], friendly_name="b")
    act_a = mdp.new_action("a")
    act_b = mdp.new_action("b")
    mdp.set_choices(init, {act_a: [(1, a)], act_b: [(1, b)]})
    mdp.set_choices(a, [(1, a)])
    mdp.set_choices(b, [(1, b)])
    return mdp, init, a, b


def _make_pomdp():
    """Three-state POMDP.

    States s0, s1 share observation 'same'; s2 has observation 'other'.
    s0 --left--> s0 (prob 1)
       --right-> s2 (prob 1)
    s1 --left--> s1 (prob 1)
       --right-> s2 (prob 1)
    s2: absorbing
    """
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_same = pomdp.new_observation("same")
    obs_other = pomdp.new_observation("other")
    s0 = pomdp.new_state(["init"], observation=obs_same)
    s1 = pomdp.new_state(["s1"], observation=obs_same)
    s2 = pomdp.new_state(["s2"], observation=obs_other)
    left = pomdp.new_action("left")
    right = pomdp.new_action("right")
    pomdp.set_choices(s0, {left: [(1, s0)], right: [(1, s2)]})
    pomdp.set_choices(s1, {left: [(1, s1)], right: [(1, s2)]})
    pomdp.set_choices(s2, [(1, s2)])
    return pomdp, s0, s1, s2


# ---------------------------------------------------------------------------
# Basic MDP → pMC
# ---------------------------------------------------------------------------


def test_result_is_dtmc():
    mdp, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    assert pmc.model_type == ModelType.DTMC


def test_state_count_preserved():
    mdp, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    assert pmc.nr_states == mdp.nr_states


def test_labels_preserved():
    mdp, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    assert "init" in pmc.state_labels
    assert "a" in pmc.state_labels
    assert "b" in pmc.state_labels


def test_multi_action_state_has_policy_params():
    mdp, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    assert len(pmc.parameters) == 2  # y_init_a and y_init_b


def test_single_action_states_have_no_params():
    """Absorbing states with one action introduce no new parameter."""
    mdp, _, a, b = _make_mdp()
    pmc = policy_to_pmc(mdp)
    # Only init (2 actions) contributes parameters.
    param_names = set(pmc.parameters)
    assert not any(
        "_a_" in n or n.endswith("_a") for n in param_names if n.startswith("y_a")
    )
    # More directly: exactly 2 params total.
    assert len(param_names) == 2


def test_combined_expression_is_parametric():
    mdp, init, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    init_new = next(iter(pmc.state_labels["init"]))
    for _, branch in pmc.transitions[init_new]:
        for val, _ in branch:
            assert sv_model.parametric.is_parametric(val)


def test_combined_expressions_sum_to_one():
    """Σ val over successors of init equals the sum of the policy parameters."""
    mdp, init, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    init_new = next(iter(pmc.state_labels["init"]))
    for _, branch in pmc.transitions[init_new]:
        total = sum(sp.sympify(val) for val, _ in branch)
        # Use the exact declared symbols (carry positive=True assumption).
        param_sum = sum(pmc._parameters[p] for p in pmc.parameters)
        assert sp.expand(total - param_sum) == 0


def test_friendly_names_preserved():
    mdp, *_ = _make_mdp()
    pmc = policy_to_pmc(mdp)
    init_new = next(iter(pmc.state_labels["init"]))
    assert pmc.friendly_names.get(init_new) == "init"


def test_state_rewards_preserved():
    mdp, init, a, b = _make_mdp()
    rm = mdp.new_reward_model("steps")
    rm.rewards[init] = 1
    rm.rewards[a] = 0
    rm.rewards[b] = 0
    pmc = policy_to_pmc(mdp)
    init_new = next(iter(pmc.state_labels["init"]))
    assert pmc.rewards[0].rewards[init_new] == 1


# ---------------------------------------------------------------------------
# POMDP → pMC
# ---------------------------------------------------------------------------


def test_pomdp_result_is_dtmc():
    pomdp, *_ = _make_pomdp()
    pmc = policy_to_pmc(pomdp)
    assert pmc.model_type == ModelType.DTMC


def test_pomdp_shared_observation_shared_params():
    """s0 and s1 share observation 'same' → they use the same y parameters."""
    pomdp, s0, s1, s2 = _make_pomdp()
    pmc = policy_to_pmc(pomdp)

    s0_new = next(iter(pmc.state_labels["init"]))
    s1_new = next(iter(pmc.state_labels["s1"]))

    def _param_names(state):
        names = set()
        for _, branch in pmc.transitions[state]:
            for val, _ in branch:
                if sv_model.parametric.is_parametric(val):
                    names |= sv_model.parametric.free_symbol_names(val)
        return names

    assert _param_names(s0_new) == _param_names(s1_new)


def test_pomdp_absorbing_state_copied_directly():
    pomdp, s0, s1, s2 = _make_pomdp()
    pmc = policy_to_pmc(pomdp)
    s2_new = next(iter(pmc.state_labels["s2"]))
    for _, branch in pmc.transitions[s2_new]:
        for val, _ in branch:
            assert not sv_model.parametric.is_parametric(val)


def test_pomdp_param_count():
    """'same' obs has 2 actions → 2 params; 'other' obs has 1 action → 0."""
    pomdp, *_ = _make_pomdp()
    pmc = policy_to_pmc(pomdp)
    assert len(pmc.parameters) == 2


# ---------------------------------------------------------------------------
# Parametric MDP (pMDP) → pMC: freshness check
# ---------------------------------------------------------------------------


def _make_pmdp():
    """pMDP where transition probabilities contain parameter x.

    init --a--> goal  with prob x
         --a--> sink  with prob 1-x
         --b--> sink  with prob 1
    goal, sink: absorbing
    """
    pmdp = sv_model.new_mdp(create_initial_state=False)
    x = pmdp.declare_parameter("x", positive=True)
    init = pmdp.new_state(["init"])
    goal = pmdp.new_state(["goal"])
    sink = pmdp.new_state(["sink"])
    act_a = pmdp.new_action("a")
    act_b = pmdp.new_action("b")
    pmdp.set_choices(init, {act_a: [(x, goal), (1 - x, sink)], act_b: [(1, sink)]})
    pmdp.set_choices(goal, [(1, goal)])
    pmdp.set_choices(sink, [(1, sink)])
    return pmdp, x


def test_pmdp_existing_param_preserved():
    pmdp, x = _make_pmdp()
    pmc = policy_to_pmc(pmdp)
    assert "x" in pmc.parameters


def test_pmdp_policy_params_are_fresh():
    """New policy parameter names must not clash with pre-existing 'x'."""
    pmdp, x = _make_pmdp()
    pmc = policy_to_pmc(pmdp)
    policy_params = [p for p in pmc.parameters if p != "x"]
    assert "x" not in policy_params
    assert all(p.startswith("y_") for p in policy_params)


def test_pmdp_combined_expr_contains_x():
    """The combined transition expression must still reference x."""
    pmdp, x = _make_pmdp()
    pmc = policy_to_pmc(pmdp)
    init_new = next(iter(pmc.state_labels["init"]))
    all_free: set[str] = set()
    for _, branch in pmc.transitions[init_new]:
        for val, _ in branch:
            if sv_model.parametric.is_parametric(val):
                all_free |= sv_model.parametric.free_symbol_names(val)
    assert "x" in all_free


def test_pmdp_name_collision_resolved():
    """If existing params include a desired policy name, a suffix is used."""
    pmdp = sv_model.new_mdp(create_initial_state=False)
    pmdp.declare_parameter("y_init_a")  # pre-occupy the natural policy name
    x = pmdp.declare_parameter("x")
    init = pmdp.new_state(["init"], friendly_name="init")
    goal = pmdp.new_state(["goal"])
    act_a = pmdp.new_action("a")
    act_b = pmdp.new_action("b")
    pmdp.set_choices(init, {act_a: [(x, goal)], act_b: [(1 - x, goal)]})
    pmdp.set_choices(goal, [(1, goal)])

    pmc = policy_to_pmc(pmdp)
    assert "y_init_a" not in [
        p for p in pmc.parameters if p.startswith("y_") and p != "y_init_a"
    ]
    # The collision-resolved name must exist.
    assert "y_init_a_0" in pmc.parameters


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_for_dtmc():
    dtmc = sv_model.new_dtmc()
    with pytest.raises(ValueError, match="MDP"):
        policy_to_pmc(dtmc)


def test_raises_for_stochastic_observation():
    from stormvogel.model.distribution import Distribution as Dist

    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s = pomdp.new_state(["init"], observation=Dist({obs_a: 0.5, obs_b: 0.5}))
    pomdp.set_choices(s, [(1, s)])
    with pytest.raises(ValueError, match="stochastic"):
        policy_to_pmc(pomdp)
