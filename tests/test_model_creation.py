import stormvogel.model


def test_mdp_creation():
    dtmc = stormvogel.model.new_dtmc()

    init = dtmc.initial_state
    # roll die
    init.set_choices(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    # we add self loops to all states with no outgoing choices
    dtmc.add_self_loops()

    assert dtmc.nr_states == 7
    assert dtmc.nr_choices == 7
    # Check that all states 1..6 have self loops
    for i in range(1, 7):
        assert dtmc.get_successor_states(dtmc.states[i]) == {dtmc.states[i]}
