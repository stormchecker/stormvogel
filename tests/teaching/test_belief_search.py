"""Tests for stormvogel.teaching.belief_search."""

from fractions import Fraction

import pytest

import stormvogel.model as sv_model
from stormvogel.examples.cheese_maze import create_cheese_maze
from stormvogel.teaching.belief import Belief, BeliefTransitions, belief_trace
from stormvogel.teaching.belief_search import (
    BeliefSearchResult,
    astar_belief_search,
    dijkstra_belief_search,
    make_lookahead_heuristic,
    make_value_bound_heuristic,
    target_probability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def maze():
    """The default 3-corridor cheese maze."""
    return create_cheese_maze()


@pytest.fixture(scope="module")
def maze_belief(maze):
    """Uniform belief over the three corridor cells the start state leads to."""
    init = next(s for s in maze.states if "init" in s.labels)
    _, branch = next(iter(maze.transitions[init]))
    return {target: Fraction(1, 3) for _, target in branch}


@pytest.fixture(scope="module")
def commitment():
    """The two-state commitment POMDP: ``(model, initial belief, s1, goal)``.

    ``s1`` and ``s2`` share observation ``z`` and nothing ever tells them
    apart, so the belief stays 50/50 no matter what the agent does.
    """
    from stormvogel.examples.two_state_commitment_pomdp import (
        create_two_state_commitment_pomdp,
    )

    pomdp = create_two_state_commitment_pomdp()
    s1 = _state(pomdp, "s1")
    s2 = _state(pomdp, "s2")
    belief = {s1: Fraction(1, 2), s2: Fraction(1, 2)}
    return pomdp, belief, s1, _state(pomdp, "g")


def _state(model, name):
    return next(s for s in model.states if s.friendly_name == name)


def _alias(observation):
    assert observation is not None
    return observation.alias


def _split_chain(levels: int = 4):
    """POMDP on which the lookahead heuristic strictly beats Dijkstra.

    From every chain state ``u_i`` / ``v_i``::

        commit --> goal (2/5), junkA (3/10), junkB (3/10)
        wait   --> u_{i+1} (1/2), v_{i+1} (1/2)

    Every state has its own observation, so ``commit`` splits the belief over
    three observations and ``wait`` over two.  The best observation
    probability from any chain state is ``1/2`` (from ``wait``), so the
    depth-1 bound is ``h = 1/2``.  Committing immediately is optimal at 2/5,
    while waiting first caps the path at ``1/2 · 2/5 = 1/5``.
    """
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    commit = pomdp.action("commit")
    wait = pomdp.action("wait")

    goal = pomdp.new_state(
        ["target"], friendly_name="goal", observation=pomdp.observation("G")
    )
    junk_a = pomdp.new_state(friendly_name="junkA", observation=pomdp.observation("JA"))
    junk_b = pomdp.new_state(friendly_name="junkB", observation=pomdp.observation("JB"))
    for absorbing in (goal, junk_a, junk_b):
        pomdp.set_choices(absorbing, {commit: [(1, absorbing)], wait: [(1, absorbing)]})

    chain = [
        [
            pomdp.new_state(
                friendly_name=f"{side}{i}", observation=pomdp.observation(f"{side}{i}")
            )
            for side in ("u", "v")
        ]
        for i in range(levels + 1)
    ]
    for i, level in enumerate(chain):
        following = chain[min(i + 1, levels)]
        for state in level:
            pomdp.set_choices(
                state,
                {
                    commit: [
                        (Fraction(2, 5), goal),
                        (Fraction(3, 10), junk_a),
                        (Fraction(3, 10), junk_b),
                    ],
                    wait: [
                        (Fraction(1, 2), following[0]),
                        (Fraction(1, 2), following[1]),
                    ],
                },
            )
    pomdp.add_label("init")
    pomdp.state_labels["init"].add(chain[0][0])
    return pomdp, chain[0][0], goal


def _split_chain_mdp_values(pomdp):
    """Hand-computed ``Pmax=? [F "target"]`` for :func:`_split_chain`.

    Committing pays 2/5 and waiting only ever leads to another chain state
    with the same choice, so every chain state is worth exactly 2/5.
    """
    values = {}
    for state in pomdp.states:
        name = state.friendly_name or ""
        values[state] = (
            Fraction(1)
            if name == "goal"
            else Fraction(0)
            if name.startswith("junk")
            else Fraction(2, 5)
        )
    return values


def _threshold_trap():
    """POMDP where the *undivided* MDP-value bound sends A* down the wrong path.

    Returns ``(model, initial belief, mdp values)``.  With ``threshold=1/2``::

        s0 --cheap--> g1 (3/5, obs G1) | x (2/5, obs X)
        s0 --detour-> w  (1,   obs W)
        w  --probe--> t  (1/2, obs C) | u (1/2, obs C)

    ``g1`` and ``t`` are the targets.  ``cheap`` reaches the certain belief
    ``{g1}`` with probability 3/5.  ``detour`` reaches ``{t: 1/2, u: 1/2}``
    with probability 1, and that belief is good too because it puts exactly
    1/2 on a target — so the optimum is 1, not 3/5.

    The MDP values are ``V(w) = 1/2`` and ``V(s0) = 3/5``.  The raw weighted
    sum therefore scores ``{w}`` at 1/2, below the 3/5 of the ``{g1}`` goal,
    and A* commits to the worse path.  Dividing by the threshold lifts
    ``h({w})`` back to 1.
    """
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    cheap = pomdp.action("cheap")
    detour = pomdp.action("detour")
    probe = pomdp.action("probe")

    obs_c = pomdp.observation("C")
    s0 = pomdp.new_state(
        ["init"], friendly_name="s0", observation=pomdp.observation("S0")
    )
    w = pomdp.new_state(friendly_name="w", observation=pomdp.observation("W"))
    g1 = pomdp.new_state(
        ["target"], friendly_name="g1", observation=pomdp.observation("G1")
    )
    x = pomdp.new_state(friendly_name="x", observation=pomdp.observation("X"))
    t = pomdp.new_state(["target"], friendly_name="t", observation=obs_c)
    u = pomdp.new_state(friendly_name="u", observation=obs_c)

    pomdp.set_choices(
        s0,
        {
            cheap: [(Fraction(3, 5), g1), (Fraction(2, 5), x)],
            detour: [(Fraction(1), w)],
        },
    )
    pomdp.set_choices(w, {probe: [(Fraction(1, 2), t), (Fraction(1, 2), u)]})
    for absorbing in (g1, x, t, u):
        pomdp.set_choices(absorbing, {probe: [(1, absorbing)]})

    values = {
        s0: Fraction(3, 5),
        w: Fraction(1, 2),
        g1: Fraction(1),
        t: Fraction(1),
        x: Fraction(0),
        u: Fraction(0),
    }
    return pomdp, {s0: Fraction(1)}, values


def _brute_force(pomdp, initial_belief, targets, threshold, max_depth):
    """Reference implementation: best path probability within *max_depth* steps.

    Enumerates every action/observation sequence exhaustively, which is only
    feasible for tiny models and shallow depths, and returns the probability
    of the most probable one that ends in a good belief.
    """
    index = BeliefTransitions(pomdp)
    target_set = frozenset(targets)
    best = Fraction(0)

    def recurse(belief, probability, depth):
        nonlocal best
        if target_probability(belief, target_set) >= threshold:
            best = max(best, probability)
            return
        # Nothing deeper can beat what we already have: probabilities only fall.
        if depth == max_depth or probability <= best:
            return
        for action in index.actions(belief):
            for obs_prob, _obs, successor in index.successors(belief, action):
                recurse(successor, probability * obs_prob, depth + 1)

    recurse(Belief({s: Fraction(p) for s, p in initial_belief.items()}), Fraction(1), 0)
    return best


# ---------------------------------------------------------------------------
# target_probability
# ---------------------------------------------------------------------------


def test_target_probability_sums_target_mass(maze, maze_belief):
    belief = Belief(maze_belief)
    corridors = set(belief)
    assert target_probability(belief, corridors) == Fraction(1)
    assert target_probability(belief, list(corridors)[:1]) == Fraction(1, 3)
    assert target_probability(belief, []) == Fraction(0)


# ---------------------------------------------------------------------------
# Basic search behaviour
# ---------------------------------------------------------------------------


def test_finds_cheese_in_one_step(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    assert result.found
    assert result.actions == ["south"]
    assert result.probability == Fraction(1, 3)
    assert result.final_target_probability == Fraction(1)


def test_reports_the_ending_observation(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    assert result.final_observation is not None
    assert _alias(result.final_observation) == "cheese"
    assert "ending observation: cheese" in str(result)


def test_lower_threshold_can_be_met_by_a_vaguer_belief(maze, maze_belief):
    """At threshold 1/3 the initial belief already puts 1/3 on the dragons' row."""
    dragons = maze.get_states_with_label("dragon")
    strict = dijkstra_belief_search(maze, maze_belief, dragons, 1)
    loose = dijkstra_belief_search(maze, maze_belief, dragons, Fraction(2, 3))
    assert loose.probability >= strict.probability
    assert len(loose.steps) <= len(strict.steps)


def test_initial_belief_already_good(maze, maze_belief):
    corridors = set(maze_belief)
    result = dijkstra_belief_search(maze, maze_belief, corridors, 1)
    assert result.found
    assert result.steps == []
    assert result.probability == Fraction(1)
    assert result.final_observation is None
    assert result.final_belief == Belief(maze_belief)


def test_localisation_path_to_a_single_dragon(maze, maze_belief):
    """Reaching one specific dragon for sure needs a detour past a unique corner."""
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.found
    # 1/3 is the prior probability of standing in the left corridor; no
    # observation sequence can do better than confirming that.
    assert result.probability == Fraction(1, 3)
    assert result.actions[0] == "north"
    assert _alias(result.final_observation) == "dragon"


def test_unreachable_target_is_not_found(commitment):
    """s1 and s2 are indistinguishable forever, so no belief ever singles s1 out."""
    pomdp, belief, s1, _goal = commitment
    result = dijkstra_belief_search(pomdp, belief, [s1], 1, max_expansions=200)
    assert not result.found
    assert result.steps == []
    assert result.probability == Fraction(0)
    assert result.final_belief == Belief(belief)


# ---------------------------------------------------------------------------
# Optimality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_name", ["(3,0)", "(3,2)", "(3,4)"])
def test_dijkstra_matches_brute_force(maze, maze_belief, target_name):
    targets = [_state(maze, target_name)]
    result = dijkstra_belief_search(maze, maze_belief, targets, 1)
    reference = _brute_force(maze, maze_belief, targets, Fraction(1), max_depth=6)
    assert result.found
    assert len(result.steps) <= 6
    assert result.probability == reference


def test_path_probability_is_the_product_of_the_steps(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    product = Fraction(1)
    for step in result.steps:
        product *= step.probability
    assert product == result.probability


def test_steps_chain_up(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.steps[0].source == Belief(maze_belief)
    for previous, following in zip(result.steps, result.steps[1:]):
        assert following.source == previous.belief


def test_path_agrees_with_belief_trace(maze, maze_belief):
    """Replaying the found trace through belief_update yields the same beliefs."""
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    replayed = belief_trace(maze, Belief(maze_belief), result.trace)
    assert replayed == result.beliefs


# ---------------------------------------------------------------------------
# A* agrees with Dijkstra
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_name", ["(3,0)", "(3,2)", "(3,4)"])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_astar_finds_the_same_optimum(maze, maze_belief, target_name, depth):
    targets = [_state(maze, target_name)]
    reference = dijkstra_belief_search(maze, maze_belief, targets, 1)
    result = astar_belief_search(maze, maze_belief, targets, 1, depth=depth)
    assert result.found == reference.found
    assert result.probability == reference.probability


def test_astar_with_trivial_heuristic_is_dijkstra(maze, maze_belief):
    targets = [_state(maze, "(3,0)")]
    reference = dijkstra_belief_search(maze, maze_belief, targets, 1)
    result = astar_belief_search(
        maze, maze_belief, targets, 1, heuristic=lambda _b: Fraction(1)
    )
    assert result.probability == reference.probability
    assert result.expansions == reference.expansions
    assert list(result.visited) == list(reference.visited)


def test_astar_expands_less_than_dijkstra_when_the_bound_bites():
    pomdp, start, _goal = _split_chain()
    belief = {start: Fraction(1)}
    reference = dijkstra_belief_search(pomdp, belief, "target", 1)
    result = astar_belief_search(pomdp, belief, "target", 1, depth=1)
    assert result.probability == reference.probability == Fraction(2, 5)
    assert result.actions == ["commit"]
    assert result.expansions < reference.expansions


def test_lookahead_heuristic_is_an_upper_bound():
    """h(b) must never be below the true remaining probability, at any depth."""
    pomdp, start, _goal = _split_chain()
    belief = Belief({start: Fraction(1)})
    truth = dijkstra_belief_search(pomdp, belief, "target", 1).probability
    for depth in range(4):
        heuristic = make_lookahead_heuristic(pomdp, "target", 1, depth=depth)
        assert heuristic(belief) >= truth


def test_deeper_lookahead_is_never_weaker():
    pomdp, start, _goal = _split_chain()
    belief = Belief({start: Fraction(1)})
    bounds = [
        make_lookahead_heuristic(pomdp, "target", 1, depth=d)(belief) for d in range(4)
    ]
    assert bounds == sorted(bounds, reverse=True)


def test_value_bound_heuristic_matches_dijkstra():
    pomdp, start, _goal = _split_chain()
    belief = {start: Fraction(1)}
    reference = dijkstra_belief_search(pomdp, belief, "target", 1)
    heuristic = make_value_bound_heuristic(
        pomdp, "target", 1, values=_split_chain_mdp_values(pomdp)
    )
    result = astar_belief_search(pomdp, belief, "target", 1, heuristic=heuristic)
    assert result.probability == reference.probability
    assert result.expansions < reference.expansions


def test_value_bound_heuristic_is_an_upper_bound():
    """h(b) >= the true best path probability, for every belief the search sees."""
    pomdp, start, _goal = _split_chain()
    reference = dijkstra_belief_search(pomdp, {start: Fraction(1)}, "target", 1)
    heuristic = make_value_bound_heuristic(
        pomdp, "target", 1, values=_split_chain_mdp_values(pomdp)
    )
    for belief in reference.visited:
        best_from_here = dijkstra_belief_search(pomdp, belief, "target", 1)
        if best_from_here.found:
            assert heuristic(belief) >= best_from_here.probability


def test_value_bound_heuristic_survives_a_partial_threshold():
    """Dividing by the threshold keeps A* optimal where the raw bound fails."""
    pomdp, belief, values = _threshold_trap()
    threshold = Fraction(1, 2)
    reference = dijkstra_belief_search(pomdp, belief, "target", threshold)
    assert reference.probability == Fraction(1)

    scaled = make_value_bound_heuristic(pomdp, "target", threshold, values=values)
    result = astar_belief_search(pomdp, belief, "target", threshold, heuristic=scaled)
    assert result.probability == Fraction(1)
    assert _alias(result.final_observation) == "C"


def test_raw_value_bound_is_inadmissible_below_threshold_one():
    """The undivided weighted sum makes A* settle for a worse path."""
    pomdp, belief, values = _threshold_trap()
    threshold = Fraction(1, 2)

    def raw(b):
        return min(
            Fraction(1),
            sum((values[s] * p for s, p in b.items()), Fraction(0)),
        )

    result = astar_belief_search(pomdp, belief, "target", threshold, heuristic=raw)
    assert result.found
    # It reports the 3/5 shortcut and never discovers the certain detour.
    assert result.probability == Fraction(3, 5)
    assert result.actions == ["cheap"]


def test_value_bound_equals_the_raw_sum_at_threshold_one():
    pomdp, _belief, values = _threshold_trap()
    heuristic = make_value_bound_heuristic(pomdp, "target", 1, values=values)
    w = _state(pomdp, "w")
    assert heuristic(Belief({w: Fraction(1)})) == Fraction(1, 2)


def test_value_bound_needs_a_label_without_explicit_values():
    pomdp, _belief, _values = _threshold_trap()
    with pytest.raises(ValueError, match="label"):
        make_value_bound_heuristic(pomdp, [_state(pomdp, "t")], 1)


def test_value_bound_from_model_checking():
    """The hand-computed values agree with mdp_bound_alpha."""
    pytest.importorskip("stormpy")
    from stormvogel.teaching.pomdp_backup import mdp_bound_alpha

    pomdp, _belief, values = _threshold_trap()
    checked = mdp_bound_alpha(pomdp, "target").values
    for state, expected in values.items():
        assert Fraction(checked[state]) == expected


def test_heuristic_of_a_good_belief_is_one():
    pomdp, _start, goal = _split_chain()
    heuristic = make_lookahead_heuristic(pomdp, "target", 1, depth=2)
    assert heuristic(Belief({goal: Fraction(1)})) == Fraction(1)


# ---------------------------------------------------------------------------
# Remembering the explored belief space
# ---------------------------------------------------------------------------


def test_visited_records_every_expanded_belief(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert len(result.visited) > len(result.beliefs)
    assert Belief(maze_belief) in result.visited
    assert result.final_belief in result.visited
    assert all(isinstance(b, Belief) for b in result.visited)


def test_visited_starts_at_the_initial_belief(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert next(iter(result.visited)) == Belief(maze_belief)
    assert result.visited[Belief(maze_belief)] == Fraction(1)


def test_visited_probabilities_do_not_increase(maze, maze_belief):
    """Dijkstra settles beliefs in non-increasing order of path probability."""
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    probabilities = list(result.visited.values())
    assert probabilities == sorted(probabilities, reverse=True)


def test_frontier_is_disjoint_from_visited(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert set(result.frontier).isdisjoint(result.visited)


def test_path_to_any_visited_belief(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    for belief, probability in result.visited.items():
        steps = result.path_to(belief)
        product = Fraction(1)
        for step in steps:
            product *= step.probability
        assert product == probability
        assert (steps[-1].belief if steps else Belief(maze_belief)) == belief


def test_path_to_the_goal_equals_the_reported_path(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.path_to(result.final_belief) == result.steps


def test_path_to_unknown_belief_raises(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    stranger = Belief({_state(maze, "(0,1)"): Fraction(1)})
    with pytest.raises(KeyError):
        result.path_to(stranger)


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_expansion_limit_truncates(maze, maze_belief):
    result = dijkstra_belief_search(
        maze, maze_belief, [_state(maze, "(3,0)")], 1, max_expansions=2
    )
    assert not result.found
    assert result.truncated
    assert result.expansions == 2
    assert "expansion limit" in str(result)


def test_exhausted_search_is_not_truncated(commitment):
    pomdp, belief, s1, _goal = commitment
    result = dijkstra_belief_search(pomdp, belief, [s1], 1, max_expansions=10_000)
    assert not result.found
    assert not result.truncated
    assert "exhausted" in str(result)


# ---------------------------------------------------------------------------
# HMMs
# ---------------------------------------------------------------------------


def _hmm():
    """A three-state HMM whose emissions only gradually reveal the state.

    ``start`` splits evenly into ``bad`` and ``ok``, which share observation
    ``quiet``, so one step leaves the belief at 50/50.  From there ``bad``
    either raises the ``alarm`` (probability 1/5, pinning the state down) or
    quietly recovers to ``ok``.
    """
    hmm = sv_model.new_hmm(create_initial_state=False)
    boot = hmm.new_observation("boot")
    quiet = hmm.new_observation("quiet")
    alarm = hmm.new_observation("alarm")

    start = hmm.new_state(["init"], friendly_name="start", observation=boot)
    bad = hmm.new_state(["fault"], friendly_name="bad", observation=quiet)
    ok = hmm.new_state(friendly_name="ok", observation=quiet)
    ringing = hmm.new_state(["fault"], friendly_name="ringing", observation=alarm)

    hmm.set_choices(start, [(Fraction(1, 2), bad), (Fraction(1, 2), ok)])
    hmm.set_choices(bad, [(Fraction(1, 5), ringing), (Fraction(4, 5), ok)])
    hmm.set_choices(ok, [(Fraction(1), ok)])
    hmm.set_choices(ringing, [(Fraction(1), ringing)])
    return hmm, start


def test_hmm_is_accepted():
    """An HMM has one unlabelled choice per state; the search handles it."""
    hmm, start = _hmm()
    result = dijkstra_belief_search(hmm, {start: Fraction(1)}, "fault", 1)
    assert result.found
    # start -> {bad, ok} (quiet, p=1) -> {ringing} (alarm, p=1/2 * 1/5).
    assert result.probability == Fraction(1, 10)
    assert result.actions == ["", ""]
    assert [_alias(o) for o in result.observations] == ["quiet", "alarm"]


def test_hmm_partial_threshold_stops_earlier():
    """At threshold 1/2 the ambiguous belief after one step already qualifies."""
    hmm, start = _hmm()
    result = dijkstra_belief_search(hmm, {start: Fraction(1)}, "fault", Fraction(1, 2))
    assert result.found
    assert len(result.steps) == 1
    assert result.probability == Fraction(1)
    assert _alias(result.final_observation) == "quiet"
    assert result.final_target_probability == Fraction(1, 2)


def test_hmm_astar_agrees_with_dijkstra():
    hmm, start = _hmm()
    belief = {start: Fraction(1)}
    reference = dijkstra_belief_search(hmm, belief, "fault", 1)
    result = astar_belief_search(hmm, belief, "fault", 1)
    assert result.probability == reference.probability
    assert [_alias(o) for o in result.observations] == [
        _alias(o) for o in reference.observations
    ]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Approximate (float) arithmetic
# ---------------------------------------------------------------------------


def test_float_mode_uses_floats(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1, exact=False)
    assert isinstance(result.probability, float)
    assert all(isinstance(p, float) for p in result.final_belief.values())


def test_exact_mode_uses_fractions(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    assert isinstance(result.probability, Fraction)
    assert all(isinstance(p, Fraction) for p in result.final_belief.values())


@pytest.mark.parametrize("target_name", ["(3,0)", "(3,2)", "(3,4)"])
def test_float_agrees_with_exact(maze, maze_belief, target_name):
    targets = [_state(maze, target_name)]
    exact = dijkstra_belief_search(maze, maze_belief, targets, 1)
    approximate = dijkstra_belief_search(maze, maze_belief, targets, 1, exact=False)
    assert approximate.found == exact.found
    assert approximate.probability == pytest.approx(float(exact.probability))
    assert approximate.actions == exact.actions


def test_float_agrees_on_a_stochastic_model():
    """Slippery transitions are where float and Fraction could drift apart."""
    slippery = create_cheese_maze(slippery=0.1)
    init = next(s for s in slippery.states if "init" in s.labels)
    _, branch = next(iter(slippery.transitions[init]))
    belief = {t: Fraction(1, 3) for _, t in branch}

    exact = dijkstra_belief_search(slippery, belief, "cheese", 1, max_expansions=500)
    approximate = dijkstra_belief_search(
        slippery, belief, "cheese", 1, exact=False, max_expansions=500
    )
    assert approximate.probability == pytest.approx(float(exact.probability))


def test_float_astar_agrees_with_float_dijkstra(maze, maze_belief):
    targets = [_state(maze, "(3,0)")]
    reference = dijkstra_belief_search(maze, maze_belief, targets, 1, exact=False)
    result = astar_belief_search(maze, maze_belief, targets, 1, exact=False)
    assert result.probability == pytest.approx(reference.probability)


def test_float_beliefs_still_deduplicate():
    """Rounding is what lets the search recognise a belief it has seen before.

    Without it, arithmetically-equal beliefs would hash differently and the
    search would revisit the same belief forever.
    """
    slippery = create_cheese_maze(slippery=0.1)
    init = next(s for s in slippery.states if "init" in s.labels)
    _, branch = next(iter(slippery.transitions[init]))
    belief = {t: Fraction(1, 3) for _, t in branch}

    exact = dijkstra_belief_search(slippery, belief, "cheese", 1, max_expansions=500)
    approximate = dijkstra_belief_search(
        slippery, belief, "cheese", 1, exact=False, max_expansions=500
    )
    # Rounding may merge near-identical beliefs, never split them.
    assert approximate.explored <= exact.explored


def test_float_key_rounds_near_identical_beliefs():
    """Two beliefs agreeing to FLOAT_KEY_PRECISION places are the same node."""
    from stormvogel.teaching.belief import FLOAT_KEY_PRECISION

    pomdp, s0, s1 = _make_two_state_pomdp()
    wobble = 10 ** -(FLOAT_KEY_PRECISION + 3)
    a = Belief({s0: 0.5, s1: 0.5})
    b = Belief({s0: 0.5 + wobble, s1: 0.5 - wobble})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_float_key_keeps_genuinely_different_beliefs_apart():
    pomdp, s0, s1 = _make_two_state_pomdp()
    a = Belief({s0: 0.5, s1: 0.5})
    b = Belief({s0: 0.5001, s1: 0.4999})
    assert a != b
    assert len({a, b}) == 2


def test_exact_and_float_beliefs_are_not_confused():
    pomdp, s0, s1 = _make_two_state_pomdp()
    exact = Belief({s0: Fraction(1, 3), s1: Fraction(2, 3)})
    approximate = Belief({s0: 1 / 3, s1: 2 / 3})
    # 1/3 has no exact float, so the two keys differ.
    assert exact != approximate


def _make_two_state_pomdp():
    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs = pomdp.new_observation("o")
    s0 = pomdp.new_state(["init"], observation=obs)
    s1 = pomdp.new_state([], observation=obs)
    act = pomdp.new_action("a")
    pomdp.set_choices(s0, {act: [(Fraction(1, 2), s0), (Fraction(1, 2), s1)]})
    pomdp.set_choices(s1, {act: [(1, s1)]})
    return pomdp, s0, s1


# ---------------------------------------------------------------------------
# Search effort
# ---------------------------------------------------------------------------


def test_explored_matches_visited(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.explored == len(result.visited)


def test_discovered_counts_visited_plus_frontier(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.discovered == len(result.visited) + len(result.frontier)
    assert result.discovered >= result.explored


def test_timings_are_recorded(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.elapsed > 0
    assert result.setup_elapsed > 0
    assert result.total_elapsed == result.setup_elapsed + result.elapsed


def test_rates_are_counts_over_elapsed(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert result.beliefs_per_second == pytest.approx(result.explored / result.elapsed)
    assert result.expansions_per_second == pytest.approx(
        result.expansions / result.elapsed
    )


def test_rates_are_zero_without_a_measurement():
    """A hand-built result has no timing, and must not divide by zero."""
    empty = BeliefSearchResult(
        steps=[],
        probability=Fraction(1),
        found=False,
        initial_belief=Belief({}),
        targets=frozenset(),
        threshold=Fraction(1),
    )
    assert empty.beliefs_per_second == 0.0
    assert empty.expansions_per_second == 0.0
    assert empty.total_elapsed == 0.0


def test_more_search_explores_more(maze, maze_belief):
    """A harder target settles strictly more beliefs than an easy one."""
    easy = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    hard = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    assert hard.explored > easy.explored


def test_str_reports_effort(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    text = str(result)
    assert f"{result.explored} belief(s) explored" in text
    assert "beliefs/s" in text
    assert "building the index" in text


def test_str_lists_actions_and_observations(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    text = str(result)
    assert "north" in text
    assert "dragon" in text
    assert "belief(s) explored" in text


def test_html_has_a_row_per_belief(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, [_state(maze, "(3,0)")], 1)
    html = result._repr_html_()
    assert html.count("<tr>") == len(result.beliefs) + 1  # + header
    assert "dragon" in html


def test_repr_is_the_summary(maze, maze_belief):
    result = dijkstra_belief_search(maze, maze_belief, "cheese", 1)
    assert repr(result) == str(result)
    assert isinstance(result, BeliefSearchResult)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_for_non_pomdp():
    dtmc = sv_model.new_dtmc()
    with pytest.raises(ValueError, match="POMDP"):
        dijkstra_belief_search(dtmc, {}, [], 1)


def test_raises_for_unnormalised_belief(maze, maze_belief):
    bad = {s: Fraction(1, 4) for s in maze_belief}
    with pytest.raises(ValueError, match="sum to 1"):
        dijkstra_belief_search(maze, bad, "cheese", 1)


def test_raises_for_bad_threshold(maze, maze_belief):
    with pytest.raises(ValueError, match="threshold"):
        dijkstra_belief_search(maze, maze_belief, "cheese", Fraction(3, 2))


def test_raises_for_unknown_label(maze, maze_belief):
    with pytest.raises(ValueError, match="labelled"):
        dijkstra_belief_search(maze, maze_belief, "gouda", 1)


def test_raises_for_negative_depth(maze, maze_belief):
    with pytest.raises(ValueError, match="depth"):
        astar_belief_search(maze, maze_belief, "cheese", 1, depth=-1)


def test_raises_for_stochastic_observation():
    from stormvogel.model.distribution import Distribution

    pomdp = sv_model.new_pomdp(create_initial_state=False)
    obs_a = pomdp.new_observation("a")
    obs_b = pomdp.new_observation("b")
    s = pomdp.new_state(["init"], observation=Distribution({obs_a: 0.5, obs_b: 0.5}))
    pomdp.set_choices(s, [(1, s)])
    with pytest.raises(ValueError, match="stochastic"):
        dijkstra_belief_search(pomdp, {s: Fraction(1)}, [s], 1)


# ---------------------------------------------------------------------------
# Slippery maze: stochastic transitions
# ---------------------------------------------------------------------------


def test_slippery_maze_cheese_probability():
    """One southward step reaches the cheese with probability 1/3 · 9/10."""
    slippery = create_cheese_maze(slippery=0.1)
    init = next(s for s in slippery.states if "init" in s.labels)
    _, branch = next(iter(slippery.transitions[init]))
    belief = {target: Fraction(1, 3) for _, target in branch}

    result = dijkstra_belief_search(slippery, belief, "cheese", 1, max_expansions=500)
    assert result.found
    assert result.actions == ["south"]
    assert _alias(result.final_observation) == "cheese"
    assert float(result.probability) == pytest.approx(1 / 3 * 0.9)


def test_slippery_maze_astar_agrees():
    slippery = create_cheese_maze(slippery=0.1)
    init = next(s for s in slippery.states if "init" in s.labels)
    _, branch = next(iter(slippery.transitions[init]))
    belief = {target: Fraction(1, 3) for _, target in branch}

    reference = dijkstra_belief_search(
        slippery, belief, "cheese", 1, max_expansions=500
    )
    result = astar_belief_search(slippery, belief, "cheese", 1, max_expansions=500)
    assert result.probability == reference.probability
