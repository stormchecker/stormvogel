import stormvogel.model
from stormvogel.examples.minitown import create_minitown_mdp
from stormvogel.teaching.multiobjective import (
    goal_unfolding,
    weighted_multi_target_reachability,
)


def test_weighted_single_target():
    """Single target T1 on a chain; P(reach T1) = 0.7, weight 5 → reward 3.5.

    s0 --a,0.7--> s1(T1) --a,1--> s2(absorb)
         --a,0.3--> s2
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    s0.set_friendly_name("s0")
    s1 = mdp.new_state(labels=["T1"])
    s1.set_friendly_name("s1")
    s2 = mdp.new_state()
    s2.set_friendly_name("s2")
    a = mdp.action("a")
    s0.set_choices({a: [(0.7, s1), (0.3, s2)]})
    s1.set_choices({a: [(1.0, s2)]})
    s2.set_choices({a: [(1.0, s2)]})

    result = weighted_multi_target_reachability(mdp, ["T1"], [5.0])
    rw = result.rewards[0]

    # Exactly one entry state carrying reward 5
    rewarded = [
        (s.friendly_name, rw.get_state_reward(s))
        for s in result.states
        if rw.get_state_reward(s)
    ]
    assert len(rewarded) == 1
    assert rewarded[0][1] == 5.0
    assert rewarded[0][0] is not None and "s1" in rewarded[0][0]


def test_weighted_overlapping_targets():
    """s1 is in both T1 and T2; entry state must carry combined reward w1+w2=5."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    s0.set_friendly_name("s0")
    s1 = mdp.new_state(labels=["T1", "T2"])
    s1.set_friendly_name("s1")
    a = mdp.action("a")
    s0.set_choices({a: [(1.0, s1)]})
    s1.set_choices({a: [(1.0, s1)]})

    result = weighted_multi_target_reachability(mdp, ["T1", "T2"], [3.0, 2.0])
    rw = result.rewards[0]

    rewarded = [
        (s.friendly_name, rw.get_state_reward(s))
        for s in result.states
        if rw.get_state_reward(s)
    ]
    # First visit to s1: both T1 and T2 are new → reward = 5; subsequent visits: 0
    assert any(v == 5.0 for _, v in rewarded)
    assert all(v == 5.0 for _, v in rewarded)


def test_weighted_repeated_visits_reward_once():
    """Cycle through target s1; reward collected only on first visit.

    s0 --a,0.5--> s1(T1) --a,1--> s0
         --a,0.5--> s2(absorb)

    Expected total reward: 0.5 * 5 = 2.5 (visits cycle but bit prevents re-reward).
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    s0.set_friendly_name("s0")
    s1 = mdp.new_state(labels=["T1"])
    s1.set_friendly_name("s1")
    s2 = mdp.new_state()
    s2.set_friendly_name("s2")
    a = mdp.action("a")
    s0.set_choices({a: [(0.5, s1), (0.5, s2)]})
    s1.set_choices({a: [(1.0, s0)]})
    s2.set_choices({a: [(1.0, s2)]})

    result = weighted_multi_target_reachability(mdp, ["T1"], [5.0])
    rw = result.rewards[0]

    rewarded = [
        (s.friendly_name, rw.get_state_reward(s))
        for s in result.states
        if rw.get_state_reward(s)
    ]
    # e(s0,0 bits) gets reward 5; e(s0,1 bits) gets 0 → only one rewarded entry state
    assert len(rewarded) == 1
    assert rewarded[0][1] == 5.0


def test_goal_unfolding_minitown():
    """goal_unfolding tracks visited goal labels via a bit-vector.

    Minitown has 4 MDP states: home (no goal label), lib ("L"), sup ("S"),
    square (no goal label). Tracking goals ["L", "S"] yields 12 reachable
    product states:
      bits 00 — home, square
      bits 01 (S seen) — home, sup, square
      bits 10 (L seen) — home, lib, square
      bits 11 (both seen) — home, sup, lib, square
    """
    mdp = create_minitown_mdp()
    product = goal_unfolding(mdp, ["L", "S"])

    assert product.nr_states == 12

    # initial product state: home with no goals seen yet
    assert product.initial_state.friendly_name == "(home,00)"
    assert "init" in product.initial_state.labels

    # "L"-labelled product states are exactly the two lib copies
    l_states = list(product.get_states_with_label("L"))
    assert {s.friendly_name for s in l_states} == {"(lib,10)", "(lib,11)"}

    # "S"-labelled product states are exactly the two sup copies
    s_states = list(product.get_states_with_label("S"))
    assert {s.friendly_name for s in s_states} == {"(sup,01)", "(sup,11)"}
