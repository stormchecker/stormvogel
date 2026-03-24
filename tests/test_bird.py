from stormvogel import bird, model
import math
import pytest
import re
from model_testing import assert_models_equal
import stormvogel
from stormvogel.model.variable import Variable


def test_bird_mdp():
    # we build the model with bird:
    N = 2
    p = 0.5
    initial_state = bird.State(x=math.floor(N / 2))

    left = "left"
    right = "right"

    def available_actions(s: bird.State) -> list[bird.Action]:
        if s.x == N:
            return [right]
        elif s.x == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s: bird.State) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def labels(s: bird.State):
        return [str(s.x)]

    def delta(s: bird.State, action: bird.Action) -> list[tuple[float, bird.State]]:
        if action == left:
            return (
                [
                    (p, bird.State(x=s.x + 1)),
                    (1 - p, bird.State(x=s.x)),
                ]
                if s.x < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, bird.State(x=s.x - 1)),
                    (1 - p, bird.State(x=s.x)),
                ]
                if s.x > 0
                else []
            )
        else:
            return []

    bird_model = bird.build_bird(
        delta=delta,
        available_actions=available_actions,
        init=initial_state,
        labels=labels,
        rewards=rewards,
    )

    # we build the model in the regular way:
    regular_model = model.new_mdp(create_initial_state=False)
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    other_left = regular_model.new_action("left")
    other_right = regular_model.new_action("right")
    branch12 = model.Distribution([(0.5, state1), (0.5, state2)])
    branch10 = model.Distribution([(0.5, state1), (0.5, state0)])
    branch01 = model.Distribution([(0.5, state0), (0.5, state1)])
    branch21 = model.Distribution([(0.5, state2), (0.5, state1)])

    regular_model.add_choices(
        state1, model.Choices({other_left: branch12, other_right: branch10})
    )
    regular_model.add_choices(state2, model.Choices({other_right: branch21}))
    regular_model.add_choices(state0, model.Choices({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel.set_state_reward(state, 1)
    rewardmodel2 = regular_model.new_reward_model("r2")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel2.set_state_reward(state, 2)
    assert_models_equal(regular_model, bird_model)


def test_bird_mdp_int():
    # we build the model with bird:
    N = 2
    p = 0.5
    initial_state = math.floor(N / 2)

    left = "left"
    right = "right"

    def available_actions(s):
        if s == N:
            return [right]
        elif s == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def labels(s):
        return [str(s)]

    def delta(s, action: bird.Action):
        if action == left:
            return (
                [
                    (p, s + 1),
                    (1 - p, s),
                ]
                if s < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, s - 1),
                    (1 - p, s),
                ]
                if s > 0
                else []
            )

    bird_model = bird.build_bird(
        delta=delta,
        available_actions=available_actions,
        init=initial_state,
        labels=labels,
        rewards=rewards,
    )

    # we build the model in the regular way:
    regular_model = model.new_mdp(create_initial_state=False)
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    other_left = regular_model.new_action("left")
    other_right = regular_model.new_action("right")
    branch12 = model.Distribution([(0.5, state1), (0.5, state2)])
    branch10 = model.Distribution([(0.5, state1), (0.5, state0)])
    branch01 = model.Distribution([(0.5, state0), (0.5, state1)])
    branch21 = model.Distribution([(0.5, state2), (0.5, state1)])

    regular_model.add_choices(
        state1,
        model.Choices(
            {other_right: branch10, other_left: branch12}
        ),  # state1, model.Choices({left: branch12, right: branch10})
    )
    regular_model.add_choices(state2, model.Choices({other_right: branch21}))
    regular_model.add_choices(state0, model.Choices({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel.set_state_reward(state, 1)
    rewardmodel2 = regular_model.new_reward_model("r2")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel2.set_state_reward(state, 2)
    assert_models_equal(regular_model, bird_model)


def test_bird_dtmc():
    # we build the model with bird:
    p = 0.5
    initial_state = bird.State(s=0)

    def rewards(s: bird.State) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def delta(s: bird.State) -> list[tuple[float, bird.State]] | None:
        match s.s:
            case 0:
                return [(p, bird.State(s=1)), (1 - p, bird.State(s=2))]
            case 1:
                return [(p, bird.State(s=3)), (1 - p, bird.State(s=4))]
            case 2:
                return [(p, bird.State(s=5)), (1 - p, bird.State(s=6))]
            case 3:
                return [(p, bird.State(s=1)), (1 - p, bird.State(s=7, d=1))]
            case 4:
                return [
                    (p, bird.State(s=7, d=2)),
                    (1 - p, bird.State(s=7, d=3)),
                ]
            case 5:
                return [
                    (p, bird.State(s=7, d=4)),
                    (1 - p, bird.State(s=7, d=5)),
                ]
            case 6:
                return [(p, bird.State(s=2)), (1 - p, bird.State(s=7, d=6))]
            case 7:
                return [(1, bird.State(s=7, d=0))]

    def valuations(
        s: bird.State,
    ) -> dict[stormvogel.model.Variable, float | int | bool]:
        s_var = stormvogel.model.Variable("s")
        d_var = stormvogel.model.Variable("d")
        match s.s:
            case 0:
                return {s_var: 0, d_var: -1}
            case 1:
                return {s_var: 1, d_var: -1}
            case 2:
                return {s_var: 2, d_var: -1}
            case 3:
                return {s_var: 3, d_var: -1}
            case 4:
                return {s_var: 4, d_var: -1}
            case 5:
                return {s_var: 5, d_var: -1}
            case 6:
                return {s_var: 6, d_var: -1}
            case 7:
                match s.d:
                    case 0:
                        return {s_var: 7, d_var: 0}
                    case 1:
                        return {s_var: 7, d_var: 1}
                    case 2:
                        return {s_var: 7, d_var: 2}
                    case 3:
                        return {s_var: 7, d_var: 3}
                    case 4:
                        return {s_var: 7, d_var: 4}
                    case 5:
                        return {s_var: 7, d_var: 5}
                    case 6:
                        return {s_var: 7, d_var: 6}
                    case _:
                        return {s_var: -1, d_var: -1}
            case _:
                return {s_var: -1, d_var: -1}

    bird_model = bird.build_bird(
        delta=delta,
        init=initial_state,
        rewards=rewards,
        modeltype=model.ModelType.DTMC,
        valuations=valuations,
    )

    # we build the model in the regular way:
    regular_model = model.new_dtmc()
    regular_model.states[0].valuations = {"s": 0}
    init = regular_model.initial_state
    init.valuations = {"s": 0, "d": -1}
    regular_model.set_choices(
        init,
        [
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 1, Variable("d"): -1}
                ),
            ),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 2, Variable("d"): -1}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[1],
        [
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 3, Variable("d"): -1}
                ),
            ),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 4, Variable("d"): -1}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[2],
        [
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 5, Variable("d"): -1}
                ),
            ),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 6, Variable("d"): -1}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[3],
        [
            (1 / 2, regular_model.states[1]),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 1}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[4],
        [
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 2}
                ),
            ),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 3}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[5],
        [
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 4}
                ),
            ),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 5}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[6],
        [
            (1 / 2, regular_model.states[2]),
            (
                1 / 2,
                regular_model.new_state(
                    valuations={Variable("s"): 7, Variable("d"): 6}
                ),
            ),
        ],
    )
    regular_model.set_choices(
        regular_model.states[7],
        [(1, regular_model.new_state(valuations={Variable("s"): 7, Variable("d"): 0}))],
    )
    regular_model.set_choices(regular_model.states[8], [(1, regular_model.states[13])])
    regular_model.set_choices(regular_model.states[9], [(1, regular_model.states[13])])
    regular_model.set_choices(regular_model.states[10], [(1, regular_model.states[13])])
    regular_model.set_choices(regular_model.states[11], [(1, regular_model.states[13])])
    regular_model.set_choices(regular_model.states[12], [(1, regular_model.states[13])])
    regular_model.set_choices(regular_model.states[13], [(1, regular_model.states[13])])

    rewardmodel = regular_model.new_reward_model("r1")
    for state in regular_model:
        rewardmodel.set_state_reward(state, 1)
    rewardmodel = regular_model.new_reward_model("r2")
    for state in regular_model:
        rewardmodel.set_state_reward(state, 2)

    assert_models_equal(bird_model, regular_model)


def test_bird_dtmc_arbitrary():
    def delta(current_state):
        match current_state:
            case "hungry":
                return ["eating"]
            case "eating":
                return ["hungry"]

    bird_model = bird.build_bird(delta, init="hungry", modeltype=model.ModelType.DTMC)

    regular_model = model.new_dtmc()
    regular_model.set_choices(
        regular_model.initial_state, [(1, regular_model.new_state())]
    )
    regular_model.set_choices(
        regular_model.states[1], [(1, regular_model.initial_state)]
    )

    assert_models_equal(bird_model, regular_model)


def test_bird_mdp_empty_action():
    # we test if we can also provide empty actions
    def available_actions(s):
        return [""]

    def delta(current_state, action):
        match current_state:
            case "hungry":
                return ["eating"]
            case "eating":
                return ["hungry"]

    bird_model = bird.build_bird(
        delta,
        init="hungry",
        available_actions=available_actions,
        modeltype=model.ModelType.MDP,
    )

    regular_model = model.new_mdp()
    regular_model.set_choices(
        regular_model.initial_state, [(1, regular_model.new_state())]
    )
    regular_model.set_choices(
        regular_model.states[1], [(1, regular_model.initial_state)]
    )

    assert_models_equal(bird_model, regular_model)
    assert len(bird_model.states) == 2
    assert len(list(bird_model.actions)) == 1


def test_bird_mdp_empty_action_2():
    # we test if we can also provide empty actions
    def available_actions(s):
        return [""]

    def delta(current_state, action):
        match current_state:
            case "hungry":
                return ["eating"]
            case "eating":
                return ["hungry"]

    bird_model = bird.build_bird(
        delta,
        init="hungry",
        available_actions=available_actions,
        modeltype=model.ModelType.MDP,
    )

    regular_model = model.new_mdp()
    regular_model.set_choices(
        regular_model.initial_state, [(1, regular_model.new_state())]
    )
    regular_model.set_choices(
        regular_model.states[1], [(1, regular_model.initial_state)]
    )

    assert_models_equal(bird_model, regular_model)
    assert len(bird_model.states) == 2
    assert len(list(bird_model.actions)) == 1


def test_bird_mdp_empty_action_3():
    # we test if we can also provide empty actions
    def available_actions(s):
        return [""]

    def delta(current_state, action):
        match current_state:
            case "hungry":
                return ["eating"]
            case "eating":
                return ["hungry"]

    bird_model = bird.build_bird(
        delta,
        init="hungry",
        available_actions=available_actions,
        modeltype=model.ModelType.MDP,
    )

    regular_model = model.new_mdp()
    regular_model.set_choices(
        regular_model.initial_state, [(1, regular_model.new_state())]
    )
    regular_model.set_choices(
        regular_model.states[1], [(1, regular_model.initial_state)]
    )

    assert_models_equal(bird_model, regular_model)
    assert len(bird_model.states) == 2
    assert len(list(bird_model.actions)) == 1


def test_bird_endless():
    # we test if we get the correct error when the model gets too large
    init = bird.State(x="")

    def available_actions(s: bird.State):
        if s == init:  # If we are in the initial state, we have a choice.
            return ["study", "don't study"]
        else:  # Otherwise, we don't have any choice, we are just a Markov chain.
            return [""]

    def delta(s: bird.State, a: bird.Action):
        if a == "study":
            return [(1, bird.State(x=["studied"]))]
        elif a == "don't study":
            return [(1, bird.State(x=["didn't study"]))]
        elif "studied" in s.x:
            return [
                (9 / 10, bird.State(x=["pass test"])),
                (1 / 10, bird.State(x=["fail test"])),
            ]
        elif "didn't study" in s.x:
            return [
                (2 / 5, bird.State(x=["pass test"])),
                (3 / 5, bird.State(x=["fail test"])),
            ]
        else:
            return [(1, bird.State(x=[f"{s.x[0]}0"]))]

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The model you want to create has a very large amount of states (at least 10000). If you wish to proceed, set max_size to some larger number."
        ),
    ):
        bird.build_bird(
            delta=delta,
            init=init,
            available_actions=available_actions,
            modeltype=model.ModelType.MDP,
            max_size=10000,
        )


def test_bird_pomdp():
    # here we test if the observations function works
    # we build the pomdp model with bird:
    N = 2
    p = 0.5
    initial_state = bird.State(x=math.floor(N / 2))

    left = "left"
    right = "right"

    def available_actions(s: bird.State):
        if s.x == N:
            return [right]
        elif s.x == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s: bird.State) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def labels(s: bird.State):
        return [str(s.x)]

    def observations(s: bird.State):
        return 5

    def delta(s: bird.State, action: bird.Action):
        if action == left:
            return (
                [
                    (p, bird.State(x=s.x + 1)),
                    (1 - p, bird.State(x=s.x)),
                ]
                if s.x < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, bird.State(x=s.x - 1)),
                    (1 - p, bird.State(x=s.x)),
                ]
                if s.x > 0
                else []
            )

    bird_model = bird.build_bird(
        delta=delta,
        available_actions=available_actions,
        init=initial_state,
        labels=labels,
        rewards=rewards,
        observations=observations,
        modeltype=model.ModelType.POMDP,
    )

    # we build the pomdp model in the regular way:
    regular_model = model.new_pomdp(create_initial_state=False)
    state1 = regular_model.new_state(
        labels=["init", "1"], observation=regular_model.new_observation("obs1")
    )
    state2 = regular_model.new_state(
        labels=["2"], observation=regular_model.new_observation("obs2")
    )
    state0 = regular_model.new_state(
        labels=["0"], observation=regular_model.new_observation("obs0")
    )
    other_left = regular_model.new_action("left")
    other_right = regular_model.new_action("right")
    branch12 = model.Distribution([(0.5, state1), (0.5, state2)])
    branch10 = model.Distribution([(0.5, state1), (0.5, state0)])
    branch01 = model.Distribution([(0.5, state0), (0.5, state1)])
    branch21 = model.Distribution([(0.5, state2), (0.5, state1)])

    regular_model.add_choices(
        state1, model.Choices({other_left: branch12, other_right: branch10})
    )
    regular_model.add_choices(state2, model.Choices({other_right: branch21}))
    regular_model.add_choices(state0, model.Choices({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel.set_state_reward(state, 1)
    rewardmodel2 = regular_model.new_reward_model("r2")
    for state in regular_model.states:
        pair = regular_model.transitions.get(state)
        assert pair is not None
        if pair is not None:
            rewardmodel2.set_state_reward(state, 2)
    observation = regular_model.new_observation("5")
    for state in regular_model.states:
        state.observation = observation

    assert_models_equal(regular_model, bird_model)


def test_bird_ctmc():
    def delta(current_state):
        match current_state:
            case "hungry":
                return [(1.0, "eating")]
            case "eating":
                return [(1.0, "hungry")]

    def rates(s) -> float:
        match s:
            case "hungry":
                return 5
            case "eating":
                return 3
            case _:
                return 0

    bird_model = bird.build_bird(
        delta, init="hungry", rates=rates, modeltype=model.ModelType.CTMC
    )

    regular_model = model.new_ctmc()
    regular_model.set_choices(
        regular_model.initial_state, [(5, regular_model.new_state())]
    )
    regular_model.set_choices(
        regular_model.states[1], [(3, regular_model.initial_state)]
    )

    assert_models_equal(bird_model, regular_model)


def test_self_loops():
    # we test if self loops automatically get added if delta returns None
    def delta(current_state):
        match current_state:
            case "nonexistent":
                return ["also nonexistent"]

    bird_model = bird.build_bird(delta, init="hungry", modeltype=model.ModelType.DTMC)

    regular_model = model.new_dtmc()
    regular_model.add_self_loops()

    assert_models_equal(bird_model, regular_model)


def test_bird_stochastic_observations():
    """Test that bird builder wraps stochastic observations in Distribution."""
    from stormvogel.model.distribution import Distribution
    from stormvogel.model.observation import Observation

    initial_state = bird.State(x=0)

    def available_actions(s: bird.State):
        return ["a"]

    def delta(s: bird.State, action: bird.Action):
        if s.x == 0:
            return [(1, bird.State(x=1))]
        return []

    def observations(s: bird.State):
        if s.x == 0:
            return [(0.6, 0), (0.4, 1)]
        else:
            return 2

    bird_model = bird.build_bird(
        delta=delta,
        available_actions=available_actions,
        init=initial_state,
        observations=observations,
        modeltype=model.ModelType.POMDP,
    )

    # The initial state (x=0) has stochastic observations, so its
    # observation must be a Distribution, not a raw list.
    init_s = bird_model.initial_state
    assert isinstance(
        init_s.observation, Distribution
    ), f"Expected Distribution, got {type(init_s.observation)}"

    # Verify the distribution contents
    obs_aliases = {obs.alias for _, obs in init_s.observation}
    assert obs_aliases == {"0", "1"}
    probs = {obs.alias: float(p) for p, obs in init_s.observation}
    assert abs(probs["0"] - 0.6) < 1e-9
    assert abs(probs["1"] - 0.4) < 1e-9

    # The second state (x=1) has a deterministic observation.
    other_states = [s for s in bird_model.states if s != init_s]
    assert len(other_states) == 1
    assert isinstance(other_states[0].observation, Observation)
    assert other_states[0].observation.alias == "2"


def test_bird_stochastic_observations_make_deterministic():
    """Stochastic observations from bird must work with make_observations_deterministic."""
    from stormvogel.model.distribution import Distribution

    initial_state = bird.State(x=0)

    def available_actions(s: bird.State):
        return ["a"]

    def delta(s: bird.State, action: bird.Action):
        if s.x == 0:
            return [(1, bird.State(x=1))]
        return []

    def observations(s: bird.State):
        if s.x == 0:
            return [(0.3, 0), (0.7, 1)]
        else:
            return 2

    bird_model = bird.build_bird(
        delta=delta,
        available_actions=available_actions,
        init=initial_state,
        observations=observations,
        modeltype=model.ModelType.POMDP,
    )

    # Precondition: initial state has a Distribution observation
    assert isinstance(bird_model.initial_state.observation, Distribution)

    # This should succeed now that we have proper Distribution objects.
    # Before the fix, isinstance(..., Distribution) was False and this
    # method would skip the state entirely, producing wrong results.
    bird_model.make_observations_deterministic()

    # After determinization, no state should have a Distribution observation.
    for s in bird_model.states:
        assert not isinstance(
            s.observation, Distribution
        ), f"State still has Distribution observation after determinization: {s}"


def test_bird_ctmc_rates_preserve_distribution():
    """CTMC rate multiplication must keep Branches.branches as a Distribution."""
    from stormvogel.model.distribution import Distribution

    def delta(current_state):
        match current_state:
            case "a":
                return [(0.4, "b"), (0.6, "c")]
            case "b":
                return [(1.0, "a")]
            case "c":
                return [(1.0, "a")]

    def rates(s) -> float:
        match s:
            case "a":
                return 10
            case "b":
                return 5
            case "c":
                return 3
            case _:
                return 0

    bird_model = bird.build_bird(
        delta, init="a", rates=rates, modeltype=model.ModelType.CTMC
    )

    # Every branch's .branches must remain a Distribution, not a plain list.
    for s in bird_model.states:
        for action, branches in bird_model.transitions[s]:
            assert isinstance(
                branches, Distribution
            ), f"branches is {type(branches)}, expected Distribution"

    # Verify the rates were multiplied into the transition values.
    # State "a" had transitions (0.4, b) and (0.6, c) with rate 10,
    # so the resulting distribution should be (4.0, b) and (6.0, c).
    state_a = bird_model.initial_state
    outgoing = state_a.get_outgoing_transitions()
    assert outgoing is not None
    transitions = list(outgoing)
    vals = sorted(float(v) for v, _ in transitions)
    assert abs(vals[0] - 4.0) < 1e-9
    assert abs(vals[1] - 6.0) < 1e-9
