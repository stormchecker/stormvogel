from stormvogel import extensions
from stormvogel.examples import create_die_dtmc, create_end_components_mdp


def test_naive_value_iteration():
    m = create_die_dtmc()
    # Find a target state
    target_state = m.states[-1]
    res = extensions.naive_value_iteration(m, epsilon=0.01, target_state=target_state)
    assert len(res) > 0


def test_dtmc_evolution():
    m = create_die_dtmc()
    res = extensions.dtmc_evolution(m, steps=5)
    assert len(res) == 5


def test_policy_iteration():
    m = create_end_components_mdp()
    extensions.policy_iteration(m, prop='P=? [F "mec1"]', visualize=False)
