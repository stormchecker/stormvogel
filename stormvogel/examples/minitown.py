import stormvogel


def create_minitown_mdp():
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    init.set_friendly_name("home")
    left = mdp.action("left")
    right = mdp.action("right")
    lib = mdp.new_state("L")
    lib.set_friendly_name("lib")
    supermarket = mdp.new_state("S")
    supermarket.set_friendly_name("sup")
    square = mdp.new_state()
    square.set_friendly_name("square")

    init.set_choices(
        {left: [(0.9, supermarket), (0.1, init)], right: [(0.9, square), (0.1, init)]}
    )
    supermarket.set_choices({right: [(0.9, init), (0.1, square)]})
    square.set_choices(
        {left: [(0.9, init), (0.1, square)], right: [(0.9, lib), (0.1, square)]}
    )
    lib.set_choices({left: [(0.9, square), (0.1, lib)]})

    return mdp
