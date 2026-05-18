"""4-State POMDP reachability example.

States s1 and s2 share observation ``z``, so the agent cannot distinguish them.
Action ``a`` (commit) attempts to reach the target with state-dependent
probability; action ``b`` (evolve) mixes the belief.  The initial belief puts
weight ``p`` on s1 and ``1 − p`` on s2.
"""

from fractions import Fraction

import stormvogel.model


def create_4state_reachability_pomdp(
    p: "Fraction | float" = Fraction(1, 2),
) -> stormvogel.model.Model:
    """Return the 4-state reachability POMDP.

    :param p: Initial probability of being in s1 (must be in (0, 1)).
    :returns: A stormvogel POMDP model.
    :raises ValueError: If *p* is not in (0, 1).
    """
    p = Fraction(p).limit_denominator(10**9)
    if not (Fraction(0) < p < Fraction(1)):
        raise ValueError(f"p must be in (0, 1); got {p}.")

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_z = pomdp.new_observation("z")
    obs_z_target = pomdp.new_observation("z_target")
    obs_z_sink = pomdp.new_observation("z_sink")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z)
    target = pomdp.new_state(
        ["target"], friendly_name="target", observation=obs_z_target
    )
    sink = pomdp.new_state(friendly_name="sink", observation=obs_z_sink)

    # Actions
    a = pomdp.action("a")
    b = pomdp.action("b")

    # Initial distribution (no named action)
    pomdp.set_choices(start, [(p, s1), (1 - p, s2)])

    # Action a: commit to target/sink
    # Action b: belief evolution
    pomdp.set_choices(
        s1,
        {
            a: [(Fraction(7, 10), target), (Fraction(3, 10), sink)],
            b: [(Fraction(8, 10), s1), (Fraction(2, 10), s2)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a: [(Fraction(2, 10), target), (Fraction(8, 10), sink)],
            b: [(Fraction(3, 10), s1), (Fraction(7, 10), s2)],
        },
    )

    # Absorbing terminal states
    pomdp.set_choices(target, [(Fraction(1), target)])
    pomdp.set_choices(sink, [(Fraction(1), sink)])

    return pomdp


def create_4state_reachability_pomdp_variantb(
    p: "Fraction | float" = Fraction(1, 2),
) -> stormvogel.model.Model:
    """Return the 4-state reachability POMDP.

    :param p: Initial probability of being in s1 (must be in (0, 1)).
    :returns: A stormvogel POMDP model.
    :raises ValueError: If *p* is not in (0, 1).
    """
    p = Fraction(p).limit_denominator(10**9)
    if not (Fraction(0) < p < Fraction(1)):
        raise ValueError(f"p must be in (0, 1); got {p}.")

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_z = pomdp.new_observation("z")
    obs_z_target = pomdp.new_observation("z_target")
    obs_z_sink = pomdp.new_observation("z_sink")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z)
    target = pomdp.new_state(
        ["target"], friendly_name="target", observation=obs_z_target
    )
    sink = pomdp.new_state(friendly_name="sink", observation=obs_z_sink)

    # Actions
    a = pomdp.action("a")
    b = pomdp.action("b")
    c = pomdp.action("c")

    # Initial distribution (no named action)
    pomdp.set_choices(start, [(p, s1), (1 - p, s2)])

    # Action a: commit to target/sink
    # Action b: belief evolution
    pomdp.set_choices(
        s1,
        {
            a: [(1, target)],
            b: [(Fraction(3, 4), s1), (Fraction(1, 4), s2)],
            c: [(Fraction(1, 4), target), (Fraction(3, 4), sink)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a: [(1, sink)],
            b: [(Fraction(2, 3), s1), (Fraction(1, 3), s2)],
            c: [(Fraction(4, 5), target), (Fraction(1, 5), sink)],
        },
    )

    # Absorbing terminal states
    pomdp.set_choices(target, [(Fraction(1), target)])
    pomdp.set_choices(sink, [(Fraction(1), sink)])

    return pomdp
