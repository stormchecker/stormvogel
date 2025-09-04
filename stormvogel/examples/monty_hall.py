import stormvogel.model


def create_monty_hall_mdp():
    mdp = stormvogel.model.new_mdp()

    init = mdp.get_initial_state()

    # first choose car position
    init.set_choice(
        [(1 / 3, mdp.new_state("carchosen", {"car_pos": i})) for i in range(3)]
    )

    # we choose a door in each case
    for s in mdp.get_states_with_label("carchosen"):
        s.set_choice(
            [
                (
                    mdp.action(f"open{i}"),
                    mdp.new_state("open", s.valuations | {"chosen_pos": i}),
                )
                for i in range(3)
            ]
        )

    # the other goat is revealed
    for s in mdp.get_states_with_label("open"):
        car_pos = s.valuations["car_pos"]
        chosen_pos = s.valuations["chosen_pos"]
        assert isinstance(car_pos, int) and isinstance(chosen_pos, int)
        other_pos = {0, 1, 2} - {car_pos, chosen_pos}
        s.set_choice(
            [
                (
                    1 / len(other_pos),
                    mdp.new_state("goatrevealed", s.valuations | {"reveal_pos": i}),
                )
                for i in other_pos
            ]
        )

    # we must choose whether we want to switch
    for s in mdp.get_states_with_label("goatrevealed"):
        car_pos = s.valuations["car_pos"]
        chosen_pos = s.valuations["chosen_pos"]
        reveal_pos = s.valuations["reveal_pos"]
        assert isinstance(reveal_pos, int) and isinstance(chosen_pos, int)
        other_pos = list({0, 1, 2} - {reveal_pos, chosen_pos})[0]
        s.set_choice(
            [
                (
                    mdp.action("stay"),
                    mdp.new_state(
                        ["done"] + (["target"] if chosen_pos == car_pos else ["lost"]),
                        s.valuations | {"chosen_pos": chosen_pos},
                    ),
                ),
                (
                    mdp.action("switch"),
                    mdp.new_state(
                        ["done"] + (["target"] if other_pos == car_pos else ["lost"]),
                        s.valuations | {"chosen_pos": other_pos},
                    ),
                ),
            ]
        )

    # we add self loops to all states with no outgoing choices
    mdp.add_self_loops()

    # we set the value -1 to all unassigned variables in the states
    mdp.set_valuation_at_remaining_states(value=-1)

    return mdp


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_monty_hall_mdp().to_dot())
