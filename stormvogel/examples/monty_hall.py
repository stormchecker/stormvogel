import stormvogel.model
from stormvogel.model.variable import Variable, IntDomain

# Positions range over {0, 1, 2}; -1 is used as a sentinel for "not yet assigned".
_POS_DOMAIN = IntDomain(-1, 2)


def create_monty_hall_mdp():
    mdp = stormvogel.model.new_mdp()

    car_pos = Variable("car_pos", _POS_DOMAIN)
    chosen_pos = Variable("chosen_pos", _POS_DOMAIN)
    reveal_pos = Variable("reveal_pos", _POS_DOMAIN)

    init = mdp.initial_state

    # first choose car position
    init.set_choices(
        [(1 / 3, mdp.new_state("carchosen", {car_pos: i})) for i in range(3)]
    )

    # we choose a door in each case
    for s in mdp.get_states_with_label("carchosen"):
        s.set_choices(
            [
                (
                    mdp.action(f"open{i}"),
                    mdp.new_state("open", s.valuations | {chosen_pos: i}),
                )
                for i in range(3)
            ]
        )

    # the other goat is revealed
    for s in mdp.get_states_with_label("open"):
        cp = s.valuations[car_pos]
        chp = s.valuations[chosen_pos]
        assert isinstance(cp, int) and isinstance(chp, int)
        other_pos = {0, 1, 2} - {cp, chp}
        s.set_choices(
            [
                (
                    1 / len(other_pos),
                    mdp.new_state("goatrevealed", s.valuations | {reveal_pos: i}),
                )
                for i in other_pos
            ]
        )

    # we must choose whether we want to switch
    for s in mdp.get_states_with_label("goatrevealed"):
        cp = s.valuations[car_pos]
        chp = s.valuations[chosen_pos]
        rp = s.valuations[reveal_pos]
        assert isinstance(rp, int) and isinstance(chp, int)
        other = list({0, 1, 2} - {rp, chp})[0]
        s.set_choices(
            [
                (
                    mdp.action("stay"),
                    mdp.new_state(
                        ["done"] + (["target"] if chp == cp else ["lost"]),
                        s.valuations | {chosen_pos: chp},
                    ),
                ),
                (
                    mdp.action("switch"),
                    mdp.new_state(
                        ["done"] + (["target"] if other == cp else ["lost"]),
                        s.valuations | {chosen_pos: other},
                    ),
                ),
            ]
        )

    # we add self loops to all states with no outgoing choices
    mdp.add_self_loops()

    # we set the value -1 to all unassigned variables in the states
    mdp.add_valuation_at_remaining_states(value=-1)

    return mdp


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_monty_hall_mdp().to_dot())
