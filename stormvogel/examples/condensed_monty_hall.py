"""Condensed Monty Hall POMDP.

States: start, d1/d2/d3 (prize location), g/b (correct/incorrect pick),
g'/b' (after host reveals), win/lose.

Observations collapse the hidden state: all door states share "blue",
g and b share "green", g' and b' share "teal".
"""

from fractions import Fraction

import stormvogel.model


def create_condensed_monty_hall() -> stormvogel.model.Model:
    """Return the condensed Monty Hall POMDP.

    The model has 10 states.  Observations do not distinguish prize location
    (d1/d2/d3), nor whether the initial pick was correct (g vs b), nor the
    post-reveal state (g' vs b').  Switching is the optimal final action.

    :returns: A stormvogel POMDP model.
    """
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_pick = pomdp.new_observation("pick?")
    obs_switch = pomdp.new_observation("switch?")
    obs_show = pomdp.new_observation("show!")
    obs_won = pomdp.new_observation("won")
    obs_lost = pomdp.new_observation("lost")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)

    d1 = pomdp.new_state(friendly_name="d1", observation=obs_pick)
    d2 = pomdp.new_state(friendly_name="d2", observation=obs_pick)
    d3 = pomdp.new_state(friendly_name="d3", observation=obs_pick)

    g = pomdp.new_state(friendly_name="g", observation=obs_switch)
    b = pomdp.new_state(friendly_name="b", observation=obs_switch)

    gp = pomdp.new_state(friendly_name="g'", observation=obs_show)
    bp = pomdp.new_state(friendly_name="b'", observation=obs_show)

    win = pomdp.new_state(["win"], friendly_name="win", observation=obs_won)
    lose = pomdp.new_state(["lose"], friendly_name="lose", observation=obs_lost)

    # Actions
    pick1 = pomdp.action("pick1")
    pick2 = pomdp.action("pick2")
    pick3 = pomdp.action("pick3")
    wait = pomdp.action("wait")
    stay = pomdp.action("stay")
    switch = pomdp.action("switch")

    # start → d1/d2/d3 uniformly (no named action)
    pomdp.set_choices(
        start,
        [(Fraction(1, 3), d1), (Fraction(1, 3), d2), (Fraction(1, 3), d3)],
    )

    # d_i: pick_i → g (correct), pick_j → b (incorrect)
    pomdp.set_choices(d1, [(pick1, g), (pick2, b), (pick3, b)])
    pomdp.set_choices(d2, [(pick1, b), (pick2, g), (pick3, b)])
    pomdp.set_choices(d3, [(pick1, b), (pick2, b), (pick3, g)])

    # host step: g → g', b → b'
    pomdp.set_choices(g, [(wait, gp)])
    pomdp.set_choices(b, [(wait, bp)])

    # final decision
    pomdp.set_choices(gp, [(stay, win), (switch, lose)])
    pomdp.set_choices(bp, [(stay, lose), (switch, win)])

    # terminal self-loops
    pomdp.set_choices(win, [(Fraction(1), win)])
    pomdp.set_choices(lose, [(Fraction(1), lose)])

    return pomdp
