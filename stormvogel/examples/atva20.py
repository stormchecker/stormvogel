"""POMDP example from ATVA 2020.

States ``s0``, ``s5``, ``s6`` share observation ``z0`` (white);
states ``s1``, ``s2`` share observation ``z1`` (orange);
states ``s3``, ``s4`` share observation ``z2`` (yellow).

The agent starts at ``s0`` and can explore within the white cluster or move
into the orange cluster.  From the orange cluster it can commit to the yellow
cluster, where action ``b`` leads directly to the goal from ``s3`` and
directly to failure from ``s4``.
"""

from fractions import Fraction

import stormvogel.model


def create_atva20_pomdp() -> stormvogel.model.Model:
    """Return the ATVA 2020 POMDP example.

    Transitions::

        s0  --a-->  s1 (3/5),  s2 (1/5),  s0 (1/5)
        s0  --b-->  s5 (1/6),  s6 (1/3),  s0 (1/2)

        s5  --a-->  s1 (1)
        s5  --b-->  s5 (1/4),  s6 (3/4)

        s6  --a-->  s2 (1)
        s6  --b-->  s5 (2/3),  s6 (1/3)

        s1  --a-->  s3 (1)
        s1  --b-->  s3 (2/3),  s4 (1/3)

        s2  --a-->  s3 (3/4),  s4 (1/4)
        s2  --b-->  s4 (1)

        s3  --a-->  goal (2/5),  fail (3/5)
        s3  --b-->  goal (1)

        s4  --a-->  goal (3/4),  fail (1/4)
        s4  --b-->  fail (1)

    :returns: A stormvogel POMDP model.
    """
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_z0 = pomdp.new_observation("z0")  # white:  s0, s5, s6
    obs_z1 = pomdp.new_observation("z1")  # orange: s1, s2
    obs_z2 = pomdp.new_observation("z2")  # yellow: s3, s4
    obs_target = pomdp.new_observation("z_target")
    obs_sink = pomdp.new_observation("z_sink")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s0 = pomdp.new_state(friendly_name="s0", observation=obs_z0)
    s5 = pomdp.new_state(friendly_name="s5", observation=obs_z0)
    s6 = pomdp.new_state(friendly_name="s6", observation=obs_z0)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z1)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z1)
    s3 = pomdp.new_state(friendly_name="s3", observation=obs_z2)
    s4 = pomdp.new_state(friendly_name="s4", observation=obs_z2)
    goal = pomdp.new_state(["target"], friendly_name="goal", observation=obs_target)
    fail = pomdp.new_state(friendly_name="fail", observation=obs_sink)

    # Actions
    a = pomdp.action("a")
    b = pomdp.action("b")

    # Initial state
    pomdp.set_choices(start, [(Fraction(1), s0)])

    # White cluster
    pomdp.set_choices(
        s0,
        {
            a: [(Fraction(3, 5), s1), (Fraction(1, 5), s2), (Fraction(1, 5), s0)],
            b: [(Fraction(1, 6), s5), (Fraction(1, 3), s6), (Fraction(1, 2), s0)],
        },
    )
    pomdp.set_choices(
        s5,
        {
            a: [(Fraction(1), s1)],
            b: [(Fraction(1, 4), s5), (Fraction(3, 4), s6)],
        },
    )
    pomdp.set_choices(
        s6,
        {
            a: [(Fraction(1), s2)],
            b: [(Fraction(2, 3), s5), (Fraction(1, 3), s6)],
        },
    )

    # Orange cluster
    pomdp.set_choices(
        s1,
        {
            a: [(Fraction(1), s3)],
            b: [(Fraction(2, 3), s3), (Fraction(1, 3), s4)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a: [(Fraction(3, 4), s3), (Fraction(1, 4), s4)],
            b: [(Fraction(1), s4)],
        },
    )

    # Yellow cluster
    pomdp.set_choices(
        s3,
        {
            a: [(Fraction(2, 5), goal), (Fraction(3, 5), fail)],
            b: [(Fraction(1), goal)],
        },
    )
    pomdp.set_choices(
        s4,
        {
            a: [(Fraction(3, 4), goal), (Fraction(1, 4), fail)],
            b: [(Fraction(1), fail)],
        },
    )

    # Absorbing terminal states
    pomdp.set_choices(goal, [(Fraction(1), goal)])
    pomdp.set_choices(fail, [(Fraction(1), fail)])

    return pomdp
