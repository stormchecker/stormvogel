from fractions import Fraction

import stormvogel.model


def create_mixedpolicymo_example() -> stormvogel.model.Model:
    """Return the mixedpolicy-mo MDP.
    :returns: A stormvogel POMDP model.
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)

    # States
    start = mdp.new_state(["init"], friendly_name="s0")
    s1 = mdp.new_state(friendly_name="s1")
    s2 = mdp.new_state(["A"], friendly_name="s2")
    s3 = mdp.new_state(["B"], friendly_name="s3")

    # Actions
    a = mdp.action("a")
    b = mdp.action("b")

    # Initial distribution (no named action)
    mdp.set_choices(start, {a: [(1, s1)], b: [(1, s2)]})

    # Action a: commit to target/sink
    # Action b: belief evolution
    mdp.set_choices(s1, [(Fraction(1, 10), s3), (Fraction(9, 10), start)])
    # Absorbing terminal states
    mdp.add_self_loops()
    return mdp


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_mixedpolicymo_example().to_dot())
