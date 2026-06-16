"""Simplified POMDP derived from the z0 cluster of the ATVA 2020 example.

States ``s0``, ``s1``, ``s2`` share observation ``z0``.  Action ``b`` mixes
beliefs within the cluster (same transitions as s0/s5/s6 in the full ATVA
model); action ``a`` commits directly to goal or failure with state-dependent
probabilities.

Transitions::

    s0  --a-->  goal (1/2),  fail (1/2)
    s0  --b-->  s1  (1/6),   s2  (1/3),  s0  (1/2)

    s1  --a-->  goal (3/4),  fail (1/4)
    s1  --b-->  s1  (1/4),   s2  (3/4)

    s2  --a-->  goal (1/4),  fail (3/4)
    s2  --b-->  s1  (2/3),   s2  (1/3)
"""

from fractions import Fraction

import stormvogel.model


def create_atva20_z0_pomdp() -> stormvogel.model.Model:
    """Return the simplified ATVA z0-cluster POMDP.

    All three states share observation ``z0``, so the belief over them forms a
    2-simplex (triangle).  The ``b`` action mixes within the cluster; the ``a``
    action commits to an absorbing goal or fail state with probabilities that
    differ per state (``s1`` is the best state, ``s2`` the worst).

    :returns: A stormvogel POMDP model.
    """
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    obs_z0 = pomdp.new_observation("z0")
    obs_target = pomdp.new_observation("z_target")
    obs_sink = pomdp.new_observation("z_sink")

    s0 = pomdp.new_state(["init"], friendly_name="s0", observation=obs_z0)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z0)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z0)
    goal = pomdp.new_state(["target"], friendly_name="goal", observation=obs_target)
    fail = pomdp.new_state(friendly_name="fail", observation=obs_sink)

    a = pomdp.action("a")
    b = pomdp.action("b")

    pomdp.set_choices(
        s0,
        {
            a: [(Fraction(1, 2), goal), (Fraction(1, 2), fail)],
            b: [(Fraction(1, 6), s1), (Fraction(1, 3), s2), (Fraction(1, 2), s0)],
        },
    )
    pomdp.set_choices(
        s1,
        {
            a: [(Fraction(3, 4), goal), (Fraction(1, 4), fail)],
            b: [(Fraction(1, 4), s1), (Fraction(3, 4), s2)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a: [(Fraction(1, 4), goal), (Fraction(3, 4), fail)],
            b: [(Fraction(2, 3), s1), (Fraction(1, 3), s2)],
        },
    )

    pomdp.set_choices(goal, [(Fraction(1), goal)])
    pomdp.set_choices(fail, [(Fraction(1), fail)])

    return pomdp
