# content of test_sysexit.py
import stormvogel.model


def test_mdp_creation():
    dtmc = stormvogel.model.new_dtmc()

    init = dtmc.get_initial_state()

    # roll die
    init.set_choice(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    # we add self loops to all states with no outgoing choices
    dtmc.add_self_loops()

    assert len(dtmc.states) == 7
    assert len(dtmc.choices) == 7
    # Check that all states 1..6 have self loops
    for i in range(1, 7):
        # yeah we need transition getting syntax
        assert dtmc.get_branch(i).branch[0][1].id == i
