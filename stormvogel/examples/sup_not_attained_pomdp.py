"""POMDP where the supremum reachability probability is not attained.

Hidden states are ``L`` (left) and ``R`` (right).  Action ``listen`` does not
change the hidden state but produces a noisy observation: from ``L`` the agent
hears ``l`` with probability 9/10 and ``r`` with probability 1/10; from ``R``
the probabilities are reversed.  Actions ``guessL`` and ``guessR`` commit:
``guessL`` transitions to goal from ``L`` and to failure from ``R``; ``guessR``
is symmetric.

**Supremum = 1.**
By listening *n* times and guessing the majority observation, the agent is
correct with probability approaching 1 (law of large numbers), so the
supremum over all observation-based policies is 1.

**Not attained.**
Every finite observation history has positive probability under both hidden
states, so whenever the agent guesses there is positive probability of
failure.  No policy achieves probability 1.

Because stormvogel uses deterministic state observations, the stochastic
observation function is encoded by splitting each hidden state into
observation-suffixed variants: ``L_l``/``R_l`` (last obs was ``l``) and
``L_r``/``R_r`` (last obs was ``r``).  States sharing an observation suffix
are indistinguishable to the agent.  The start state distributes uniformly
over all four variants, giving initial belief ``{L: 1/2, R: 1/2}``
regardless of the first observed observation.
"""

from fractions import Fraction

import stormvogel.model


def create_sup_not_attained_pomdp() -> stormvogel.model.Model:
    """Return the "supremum not attained" POMDP.

    The start state distributes uniformly over ``L_l``, ``R_l``, ``L_r``,
    ``R_r`` (probability 1/4 each), so the initial belief is
    ``{L: 1/2, R: 1/2}`` under either first observation.

    Transitions::

        L_l / L_r  --listen-->  L_l (9/10),  L_r (1/10)
        R_l / R_r  --listen-->  R_l (1/10),  R_r (9/10)

        L_l / L_r  --guessL-->  g (1)
        L_l / L_r  --guessR-->  f (1)
        R_l / R_r  --guessL-->  f (1)
        R_l / R_r  --guessR-->  g (1)

    :returns: A stormvogel POMDP model.
    """
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_l = pomdp.new_observation("l")  # last obs was l: L_l / R_l
    obs_r = pomdp.new_observation("r")  # last obs was r: L_r / R_r
    obs_g = pomdp.new_observation("g")
    obs_f = pomdp.new_observation("f")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    L_l = pomdp.new_state(friendly_name="L_l", observation=obs_l)
    R_l = pomdp.new_state(friendly_name="R_l", observation=obs_l)
    L_r = pomdp.new_state(friendly_name="L_r", observation=obs_r)
    R_r = pomdp.new_state(friendly_name="R_r", observation=obs_r)
    g = pomdp.new_state(["target"], friendly_name="g", observation=obs_g)
    f = pomdp.new_state(friendly_name="f", observation=obs_f)

    # Actions
    listen = pomdp.action("listen")
    guessL = pomdp.action("guessL")
    guessR = pomdp.action("guessR")

    # Uniform initial distribution over all four (hidden state, obs) variants
    pomdp.set_choices(
        start,
        [
            (Fraction(1, 4), L_l),
            (Fraction(1, 4), R_l),
            (Fraction(1, 4), L_r),
            (Fraction(1, 4), R_r),
        ],
    )

    # L states: listen is noisy toward l; guessL succeeds
    for L_state in (L_l, L_r):
        pomdp.set_choices(
            L_state,
            {
                listen: [(Fraction(9, 10), L_l), (Fraction(1, 10), L_r)],
                guessL: [(Fraction(1), g)],
                guessR: [(Fraction(1), f)],
            },
        )

    # R states: listen is noisy toward r; guessR succeeds
    for R_state in (R_l, R_r):
        pomdp.set_choices(
            R_state,
            {
                listen: [(Fraction(1, 10), R_l), (Fraction(9, 10), R_r)],
                guessL: [(Fraction(1), f)],
                guessR: [(Fraction(1), g)],
            },
        )

    # Absorbing terminal states
    pomdp.set_choices(g, [(Fraction(1), g)])
    pomdp.set_choices(f, [(Fraction(1), f)])

    return pomdp
