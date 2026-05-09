"""6-State POMDP reachability example.

A two-level extension of the 4-state reachability POMDP.

States ``s1`` and ``s2`` share observation ``z`` and behave exactly as in the
4-state model: action ``a`` commits towards the target, action ``b`` mixes
the belief over ``{s1, s2}``.

States ``s3`` and ``s4`` share observation ``z2`` and mirror the structure of
``s1``/``s2``, but action ``a`` sends them to ``s1``/``s2`` rather than to
target/sink, and action ``b`` mixes within ``{s3, s4}``.

The agent therefore faces two indistinguishable pairs of states at different
levels of the problem: it must first decide (at obs ``z2``) when to transition
to the inner level, and then (at obs ``z``) when to commit to the target.
"""

from fractions import Fraction

import stormvogel.model


def create_6state_reachability_pomdp(
    p: "Fraction | float" = Fraction(1, 2),
) -> stormvogel.model.Model:
    """Return the 6-state reachability POMDP.

    The initial distribution puts weight *p* on ``s3`` and ``1 − p`` on ``s4``.

    Transitions at the outer level (obs ``z2``)::

        s3  --a-->  s1 (7/10),  s2 (3/10)
        s3  --b-->  s3 (8/10),  s4 (2/10)
        s4  --a-->  s1 (2/10),  s2 (8/10)
        s4  --b-->  s3 (3/10),  s4 (7/10)

    Transitions at the inner level (obs ``z``) are identical to the 4-state
    model::

        s1  --a-->  target (7/10),  sink (3/10)
        s1  --b-->  s1 (8/10),     s2 (2/10)
        s2  --a-->  target (2/10),  sink (8/10)
        s2  --b-->  s1 (3/10),     s2 (7/10)

    :param p: Initial probability of being in ``s3`` (must be in (0, 1)).
    :returns: A stormvogel POMDP model.
    :raises ValueError: If *p* is not in (0, 1).
    """
    p = Fraction(p).limit_denominator(10**9)
    if not (Fraction(0) < p < Fraction(1)):
        raise ValueError(f"p must be in (0, 1); got {p}.")

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_z2 = pomdp.new_observation("z2")  # outer level: s3, s4
    obs_z = pomdp.new_observation("z")  # inner level: s1, s2
    obs_z_target = pomdp.new_observation("z_target")
    obs_z_sink = pomdp.new_observation("z_sink")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s3 = pomdp.new_state(friendly_name="s3", observation=obs_z2)
    s4 = pomdp.new_state(friendly_name="s4", observation=obs_z2)
    s1 = pomdp.new_state(friendly_name="s1", observation=obs_z)
    s2 = pomdp.new_state(friendly_name="s2", observation=obs_z)
    target = pomdp.new_state(
        ["target"], friendly_name="target", observation=obs_z_target
    )
    sink = pomdp.new_state(friendly_name="sink", observation=obs_z_sink)

    # Actions
    a = pomdp.action("a")
    b = pomdp.action("b")

    # Initial distribution over outer states
    pomdp.set_choices(start, [(p, s3), (1 - p, s4)])

    # Outer level: a sends to inner level, b mixes within {s3, s4}
    pomdp.set_choices(
        s3,
        {
            a: [(Fraction(7, 10), s1), (Fraction(3, 10), s2)],
            b: [(Fraction(8, 10), s3), (Fraction(2, 10), s4)],
        },
    )
    pomdp.set_choices(
        s4,
        {
            a: [(Fraction(2, 10), s1), (Fraction(8, 10), s2)],
            b: [(Fraction(3, 10), s3), (Fraction(7, 10), s4)],
        },
    )

    # Inner level: a commits to target/sink, b mixes within {s1, s2}
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


def create_6state_reachability_pomdp_variantb(
    p: "Fraction | float" = Fraction(1, 2),
) -> stormvogel.model.Model:
    """Return the 6-state reachability POMDP, variant b (three actions).

    Mirrors :func:`create_4state_reachability_pomdp_variantb`: the inner level
    uses three actions (``a``, ``b``, ``c``), and the outer level (s3/s4)
    mirrors the same structure with s1/s2 in place of target/sink.

    Transitions at the outer level (obs ``z2``)::

        s3  --a-->  s1 (1)
        s3  --b-->  s3 (3/4),  s4 (1/4)
        s3  --c-->  s1 (1/4),  s2 (3/4)
        s4  --a-->  s2 (1)
        s4  --b-->  s3 (2/3),  s4 (1/3)
        s4  --c-->  s1 (4/5),  s2 (1/5)

    Transitions at the inner level (obs ``z``) are identical to the 4-state
    variant b::

        s1  --a-->  target (1)
        s1  --b-->  s1 (3/4),     s2 (1/4)
        s1  --c-->  target (1/4), sink (3/4)
        s2  --a-->  sink (1)
        s2  --b-->  s1 (2/3),     s2 (1/3)
        s2  --c-->  target (4/5), sink (1/5)

    :param p: Initial probability of being in ``s3`` (must be in (0, 1)).
    :returns: A stormvogel POMDP model.
    :raises ValueError: If *p* is not in (0, 1).
    """
    p = Fraction(p).limit_denominator(10**9)
    if not (Fraction(0) < p < Fraction(1)):
        raise ValueError(f"p must be in (0, 1); got {p}.")

    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)

    # Observations
    obs_start = pomdp.new_observation("start")
    obs_z2 = pomdp.new_observation("z2")
    obs_z = pomdp.new_observation("z")
    obs_z_target = pomdp.new_observation("z_target")
    obs_z_sink = pomdp.new_observation("z_sink")

    # States
    start = pomdp.new_state(["init"], friendly_name="start", observation=obs_start)
    s3 = pomdp.new_state(friendly_name="s3", observation=obs_z2)
    s4 = pomdp.new_state(friendly_name="s4", observation=obs_z2)
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

    # Initial distribution over outer states
    pomdp.set_choices(start, [(p, s3), (1 - p, s4)])

    # Outer level
    pomdp.set_choices(
        s3,
        {
            a: [(Fraction(1), s1)],
            b: [(Fraction(3, 4), s3), (Fraction(1, 4), s4)],
            c: [(Fraction(1, 4), s1), (Fraction(3, 4), s2)],
        },
    )
    pomdp.set_choices(
        s4,
        {
            a: [(Fraction(1), s2)],
            b: [(Fraction(2, 3), s3), (Fraction(1, 3), s4)],
            c: [(Fraction(4, 5), s1), (Fraction(1, 5), s2)],
        },
    )

    # Inner level (identical to variantb)
    pomdp.set_choices(
        s1,
        {
            a: [(Fraction(1), target)],
            b: [(Fraction(3, 4), s1), (Fraction(1, 4), s2)],
            c: [(Fraction(1, 4), target), (Fraction(3, 4), sink)],
        },
    )
    pomdp.set_choices(
        s2,
        {
            a: [(Fraction(1), sink)],
            b: [(Fraction(2, 3), s1), (Fraction(1, 3), s2)],
            c: [(Fraction(4, 5), target), (Fraction(1, 5), sink)],
        },
    )

    # Absorbing terminal states
    pomdp.set_choices(target, [(Fraction(1), target)])
    pomdp.set_choices(sink, [(Fraction(1), sink)])

    return pomdp
