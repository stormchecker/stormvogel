"""Two-state commitment POMDP.

This POMDP illustrates the strict gap  V_POMDP < V_QMDP < V_MDP::

    V_POMDP(b_0) = 1/2,   V_QMDP(b_0) = 4/5,   V_MDP(b_0) = 1.

States ``s1`` and ``s2`` share observation ``z``; the agent can never
distinguish them.  Action ``a1`` succeeds (reaches goal ``g``) iff the
hidden state is ``s1``; action ``a2`` succeeds iff it is ``s2``.  Action
``w`` keeps the current state with probability 4/5 but risks failure with
1/5, and is never informative.

Because ``w`` is strictly risky and reveals no information, the optimal
POMDP policy commits immediately with ``a1`` (or ``a2``) at the 50/50
initial belief, giving value 1/2.  The QMDP heuristic over-estimates at
4/5 by assuming the state becomes known after one step.  The MDP oracle
always picks the right action and achieves value 1.
"""

from fractions import Fraction

import stormvogel.model


def create_two_state_commitment_pomdp(
    p: "Fraction | float" = Fraction(1, 2),
) -> stormvogel.model.Model:
    """Return the two-state commitment POMDP.

    The initial distribution puts weight *p* on ``s1`` and ``1 − p`` on
    ``s2``.

    Transitions::

        s1  --a1-->  g (1)
        s1  --a2-->  f (1)
        s1  --w -->  s1 (4/5),  f (1/5)

        s2  --a1-->  f (1)
        s2  --a2-->  g (1)
        s2  --w -->  s2 (4/5),  f (1/5)

    Analytical values at the uniform initial belief ``b_0 = {s1:1/2, s2:1/2}``::

        V_MDP(b_0)   = 1     (oracle always picks the correct action)
        V_QMDP(b_0)  = 4/5   (w-alpha dominates; assumes revelation after one step)
        V_POMDP(b_0) = 1/2   (commit immediately; waiting is never informative)

    :param p: Initial probability of being in ``s1`` (must be in (0, 1)).
    :returns: A stormvogel POMDP model.
    :raises ValueError: If *p* is not in (0, 1).
    """
    p = Fraction(p).limit_denominator(10**9)
    if not (Fraction(0) < p < Fraction(1)):
        raise ValueError(f"p must be in (0, 1); got {p}.")

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    obs_start = pomdp.new_observation("start")
    obs_z = pomdp.new_observation("z")
    obs_g = pomdp.new_observation("g")
    obs_f = pomdp.new_observation("f")

    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z)
    g = pomdp.new_state(["target"], friendly_name="g", observation=obs_g)
    f = pomdp.new_state(friendly_name="f", observation=obs_f)

    a1 = pomdp.action("a1")
    a2 = pomdp.action("a2")
    w = pomdp.action("w")

    pomdp.set_choices(start, [(p, s1), (1 - p, s2)])

    pomdp.set_choices(
        s1,
        {
            a1: [(Fraction(1), g)],
            a2: [(Fraction(1), f)],
            w: [(Fraction(4, 5), s1), (Fraction(1, 5), f)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a1: [(Fraction(1), f)],
            a2: [(Fraction(1), g)],
            w: [(Fraction(4, 5), s2), (Fraction(1, 5), f)],
        },
    )

    pomdp.set_choices(g, [(Fraction(1), g)])
    pomdp.set_choices(f, [(Fraction(1), f)])

    return pomdp
