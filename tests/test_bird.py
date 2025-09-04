from stormvogel import bird, model
import math
import pytest
import re


def test_bird_mdp():
    # we build the model with bird:
    N = 2
    p = 0.5
    initial_state = bird.State(x=math.floor(N / 2))

    left = ["left"]
    right = ["right"]

    def available_actions(s: bird.State) -> list[bird.Action]:
        if s.x == N:
            return [right]
        elif s.x == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s: bird.State, a: bird.Action) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def labels(s: bird.State):
        return [str(s.x)]

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
    )

    # we build the model in the regular way:
    regular_model = model.new_mdp(create_initial_state=False)
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    other_left = regular_model.new_action(frozenset({"left"}))
    other_right = regular_model.new_action(frozenset({"right"}))
    branch12 = model.Branch([(0.5, state1), (0.5, state2)])
    branch10 = model.Branch([(0.5, state1), (0.5, state0)])
    branch01 = model.Branch([(0.5, state0), (0.5, state1)])
    branch21 = model.Branch([(0.5, state2), (0.5, state1)])

    regular_model.add_choice(
        state1, model.Choice({other_left: branch12, other_right: branch10})
    )
    regular_model.add_choice(state2, model.Choice({other_right: branch21}))
    regular_model.add_choice(state0, model.Choice({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.new_reward_model("r2")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    assert regular_model == bird_model


def test_bird_mdp_int():
    # we build the model with bird:
    N = 2
    p = 0.5
    initial_state = math.floor(N / 2)

    left = ["left"]
    right = ["right"]

    def available_actions(s):
        if s == N:
            return [right]
        elif s == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s, a: bird.Action) -> dict[str, model.Value]:
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
    other_left = regular_model.new_action(frozenset({"left"}))
    other_right = regular_model.new_action(frozenset({"right"}))
    branch12 = model.Branch([(0.5, state1), (0.5, state2)])
    branch10 = model.Branch([(0.5, state1), (0.5, state0)])
    branch01 = model.Branch([(0.5, state0), (0.5, state1)])
    branch21 = model.Branch([(0.5, state2), (0.5, state1)])

    regular_model.add_choice(
        state1,
        model.Choice(
            {other_right: branch10, other_left: branch12}
        ),  # state1, model.Choice({left: branch12, right: branch10})
    )
    regular_model.add_choice(state2, model.Choice({other_right: branch21}))
    regular_model.add_choice(state0, model.Choice({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.new_reward_model("r2")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    assert regular_model == bird_model


def test_bird_dtmc():
    # we build the model with bird:
    p = 0.5
    initial_state = bird.State(s=0)

    def rewards(s: bird.State) -> dict[str, model.Value]:
        return {"r1": 1, "r2": 2}

    def delta(s: bird.State):
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

    def valuations(s: bird.State) -> dict[str, float | int | bool]:
        match s.s:
            case 0:
                return {"s": 0, "d": -1}
            case 1:
                return {"s": 1, "d": -1}
            case 2:
                return {"s": 2, "d": -1}
            case 3:
                return {"s": 3, "d": -1}
            case 4:
                return {"s": 4, "d": -1}
            case 5:
                return {"s": 5, "d": -1}
            case 6:
                return {"s": 6, "d": -1}
            case 7:
                match s.d:
                    case 0:
                        return {"s": 7, "d": 0}
                    case 1:
                        return {"s": 7, "d": 1}
                    case 2:
                        return {"s": 7, "d": 2}
                    case 3:
                        return {"s": 7, "d": 3}
                    case 4:
                        return {"s": 7, "d": 4}
                    case 5:
                        return {"s": 7, "d": 5}
                    case 6:
                        return {"s": 7, "d": 6}
                    case _:
                        return {"s": -1, "d": -1}
            case _:
                return {"s": -1, "d": -1}

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
    init = regular_model.get_initial_state()
    init.valuations = {"s": 0, "d": -1}
    regular_model.set_choice(
        init,
        [
            (1 / 2, regular_model.new_state(valuations={"s": 1, "d": -1})),
            (1 / 2, regular_model.new_state(valuations={"s": 2, "d": -1})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(1),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 3, "d": -1})),
            (1 / 2, regular_model.new_state(valuations={"s": 4, "d": -1})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(2),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 5, "d": -1})),
            (1 / 2, regular_model.new_state(valuations={"s": 6, "d": -1})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(3),
        [
            (1 / 2, regular_model.get_state_by_id(1)),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 1})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(4),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 2})),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 3})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(5),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 4})),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 5})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(6),
        [
            (1 / 2, regular_model.get_state_by_id(2)),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 6})),
        ],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(7),
        [(1, regular_model.new_state(valuations={"s": 7, "d": 0}))],
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(8), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(9), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(10), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(11), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(12), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(13), [(1, regular_model.get_state_by_id(13))]
    )

    rewardmodel = regular_model.new_reward_model("r1")
    for _, state in regular_model:
        rewardmodel.set_state_reward(state, 1)
    rewardmodel = regular_model.new_reward_model("r2")
    for _, state in regular_model:
        rewardmodel.set_state_reward(state, 2)

    assert bird_model == regular_model


def test_bird_dtmc_arbitrary():
    def delta(current_state):
        match current_state:
            case "hungry":
                return ["eating"]
            case "eating":
                return ["hungry"]

    bird_model = bird.build_bird(delta, init="hungry", modeltype=model.ModelType.DTMC)

    regular_model = model.new_dtmc()
    regular_model.set_choice(
        regular_model.get_initial_state(), [(1, regular_model.new_state())]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(1), [(1, regular_model.get_initial_state())]
    )

    assert bird_model == regular_model


def test_bird_mdp_empty_action():
    # we test if we can also provide empty actions
    def available_actions(s):
        return [[]]

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
    regular_model.set_choice(
        regular_model.get_initial_state(), [(1, regular_model.new_state())]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(1), [(1, regular_model.get_initial_state())]
    )

    assert bird_model == regular_model


def test_bird_endless():
    # we test if we get the correct error when the model gets too large
    init = bird.State(x="")

    def available_actions(s: bird.State):
        if s == init:  # If we are in the initial state, we have a choice.
            return [["study"], ["don't study"]]
        else:  # Otherwise, we don't have any choice, we are just a Markov chain.
            return [[]]

    def delta(s: bird.State, a: bird.Action):
        if "study" in a:
            return [(1, bird.State(x=["studied"]))]
        elif "don't study" in a:
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
            "The model you want te create has a very large amount of states (at least 10000), if you wish to proceed, set max_size to some larger number."
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

    left = ["left"]
    right = ["right"]

    def available_actions(s: bird.State):
        if s.x == N:
            return [right]
        elif s.x == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s: bird.State, a: bird.Action) -> dict[str, model.Value]:
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
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    other_left = regular_model.new_action(frozenset({"left"}))
    other_right = regular_model.new_action(frozenset({"right"}))
    branch12 = model.Branch([(0.5, state1), (0.5, state2)])
    branch10 = model.Branch([(0.5, state1), (0.5, state0)])
    branch01 = model.Branch([(0.5, state0), (0.5, state1)])
    branch21 = model.Branch([(0.5, state2), (0.5, state1)])

    regular_model.add_choice(
        state1, model.Choice({other_left: branch12, other_right: branch10})
    )
    regular_model.add_choice(state2, model.Choice({other_right: branch21}))
    regular_model.add_choice(state0, model.Choice({other_left: branch01}))

    rewardmodel = regular_model.new_reward_model("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.new_reward_model("r2")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    for _, state in regular_model:
        state.set_observation(5)

    assert regular_model == bird_model


def test_bird_ctmc():
    def delta(current_state):
        match current_state:
            case "hungry":
                return [(5.0, "eating")]
            case "eating":
                return [(3.0, "hungry")]

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
    regular_model.set_choice(
        regular_model.get_initial_state(), [(5, regular_model.new_state())]
    )
    regular_model.set_choice(
        regular_model.get_state_by_id(1), [(3, regular_model.get_initial_state())]
    )
    regular_model.set_rate(regular_model.get_initial_state(), 5)
    regular_model.set_rate(list(regular_model.states.values())[1], 3)

    assert bird_model == regular_model


def test_self_loops():
    # we test if self loops automatically get added if delta returns None
    def delta(current_state):
        match current_state:
            case "nonexistent":
                return ["also nonexistent"]

    bird_model = bird.build_bird(delta, init="hungry", modeltype=model.ModelType.DTMC)

    regular_model = model.new_dtmc()
    regular_model.add_self_loops()

    assert bird_model == regular_model
