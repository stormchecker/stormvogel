import stormvogel.examples.die
import stormvogel.examples.monty_hall
import stormvogel.examples.nuclear_fusion_ctmc
import stormvogel.examples.monty_hall_pomdp
from stormvogel.examples.lion import create_lion_mdp
import stormvogel.model
import stormvogel.simulator as simulator


def test_simulate():
    # we make a die dtmc and run the simulator with it
    dtmc = stormvogel.examples.die.create_die_dtmc()
    rewardmodel = dtmc.new_reward_model("rewardmodel")
    for state in dtmc:
        rewardmodel.set_state_reward(state, 3)
    rewardmodel2 = dtmc.new_reward_model("rewardmodel2")
    for state in dtmc:
        rewardmodel2.set_state_reward(state, 2)
    rewardmodel3 = dtmc.new_reward_model("rewardmodel3")
    for state in dtmc:
        rewardmodel3.set_state_reward(state, 1)
    partial_model = simulator.simulate(dtmc, runs=5, steps=1, seed=3)

    # we make the partial model that should be created by the simulator
    other_dtmc = stormvogel.model.new_dtmc()
    init = other_dtmc.initial_state
    init.valuations = {"rolled": 0}
    init.set_choices(
        [
            (1 / 6, other_dtmc.new_state("rolled2", valuations={"rolled": 2})),
            (1 / 6, other_dtmc.new_state("rolled4", valuations={"rolled": 4})),
            (1 / 6, other_dtmc.new_state("rolled5", valuations={"rolled": 5})),
        ]
    )

    rewardmodel = other_dtmc.new_reward_model("rewardmodel")
    for state in other_dtmc:
        rewardmodel.set_state_reward(state, float(3))
    rewardmodel2 = other_dtmc.new_reward_model("rewardmodel2")
    for state in other_dtmc:
        rewardmodel2.set_state_reward(state, float(2))
    rewardmodel3 = other_dtmc.new_reward_model("rewardmodel3")
    for state in other_dtmc:
        rewardmodel3.set_state_reward(state, float(1))

    assert partial_model == other_dtmc
    ######################################################################################################################
    # we make a monty hall mdp and run the simulator with it
    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()
    rewardmodel = mdp.new_reward_model("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))
    rewardmodel2 = mdp.new_reward_model("rewardmodel2")
    rewardmodel2.set_from_rewards_vector(list(range(67)))

    taken_actions = {}
    for state in mdp:
        taken_actions[state] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # we make the partial model that should be created by the simulator
    other_mdp = stormvogel.model.new_mdp()
    init = other_mdp.initial_state
    init.valuations = {"reveal_pos": -1, "car_pos": -1, "chosen_pos": -1}
    init.set_choices(
        [
            (
                1 / 3,
                other_mdp.new_state(
                    "carchosen",
                    valuations={"car_pos": 0, "reveal_pos": -1, "chosen_pos": -1},
                ),
            )
        ]
    )
    branch = stormvogel.model.Branches(
        [
            (
                1,
                other_mdp.new_state(
                    "open", valuations={"car_pos": 0, "chosen_pos": 0, "reveal_pos": -1}
                ),
            )
        ]
    )
    action1 = other_mdp.new_action("open0")
    transition = stormvogel.model.Choices({action1: branch})
    other_mdp.states[1].set_choices(transition)
    other_mdp.states[2].add_choices(
        [
            (
                0.5,
                other_mdp.new_state(
                    "goatrevealed",
                    valuations={"car_pos": 0, "chosen_pos": 0, "reveal_pos": 1},
                ),
            )
        ]
    )

    rewardmodel = other_mdp.new_reward_model("rewardmodel")
    rewardmodel.rewards = {
        (0, stormvogel.model.EmptyAction): 0,
        (1, action1): 1,
        (4, stormvogel.model.EmptyAction): 10,
    }
    rewardmodel2 = other_mdp.new_reward_model("rewardmodel2")
    rewardmodel2.rewards = {
        (0, stormvogel.model.EmptyAction): 0,
        (1, action1): 1,
        (4, stormvogel.model.EmptyAction): 10,
    }

    assert partial_model == other_mdp
    ######################################################################################################################

    # we test the simulator for an mdp with a lambda as Scheduler

    def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
        actions = state.available_actions()
        return actions[0]

    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    partial_model = simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # we make the partial model that should be created by the simulator
    other_mdp = stormvogel.model.new_mdp()
    init = other_mdp.initial_state
    init.valuations = {"reveal_pos": -1, "chosen_pos": -1, "car_pos": -1}
    other_mdp.initial_state.set_choices(
        [
            (
                1 / 3,
                other_mdp.new_state(
                    "carchosen",
                    valuations={"car_pos": 0, "reveal_pos": -1, "chosen_pos": -1},
                ),
            )
        ]
    )
    branch = stormvogel.model.Branches(
        [
            (
                1,
                other_mdp.new_state(
                    "open", valuations={"car_pos": 0, "chosen_pos": 0, "reveal_pos": -1}
                ),
            )
        ]
    )
    action1 = other_mdp.new_action("open0")
    transition = stormvogel.model.Choices({action1: branch})
    other_mdp.states[1].set_choices(transition)
    other_mdp.states[2].set_choices(
        [
            (
                0.5,
                other_mdp.new_state(
                    "goatrevealed",
                    valuations={"car_pos": 0, "chosen_pos": 0, "reveal_pos": 1},
                ),
            )
        ]
    )

    assert partial_model == other_mdp

    # we do a more complicated mdp test to check if partial model choices are properly added:
    lion = create_lion_mdp()
    partial_model = simulator.simulate(lion, steps=100, seed=1, scheduler=scheduler)

    lion = stormvogel.model.new_mdp()
    init = lion.initial_state
    hungry = lion.new_state("hungry :(")
    satisfied = init
    full = lion.new_state("full")
    starving = lion.new_state("starving :((")
    hunt = lion.new_action("hunt >:D")

    satisfied.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Branches(
                    [(0.5, satisfied), (0.3, full), (0.2, hungry)]
                ),
            }
        )
    )

    hungry.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Branches(
                    [(0.2, full), (0.5, satisfied), (0.2, starving)]
                ),
            }
        )
    )

    full.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Branches(
                    [
                        (0.5, full),
                        (0.5, satisfied),
                    ]
                ),
            }
        )
    )

    starving.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Branches(0.2, hungry),
            }
        )
    )

    lion.add_self_loops()

    reward_model = lion.new_reward_model("R")
    reward_model.set_unset_rewards(0)

    assert lion == partial_model


def test_simulate_path():
    # we make the nuclear fusion ctmc and run simulate path with it
    ctmc = stormvogel.examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    path = simulator.simulate_path(ctmc, steps=5, seed=1)

    # we make the path that the simulate path function should create
    other_path = simulator.Path(
        [
            ctmc.states[1],
            ctmc.states[2],
            ctmc.states[3],
            ctmc.states[4],
        ],
        ctmc,
    )

    assert path == other_path
    ##############################################################################################
    # we make the monty hall pomdp and run simulate path with it
    pomdp = stormvogel.examples.monty_hall_pomdp.create_monty_hall_pomdp()
    taken_actions = {}
    for state in pomdp:
        taken_actions[state] = state.available_actions()[
            len(state.available_actions()) - 1
        ]
    scheduler = stormvogel.result.Scheduler(pomdp, taken_actions)
    path = simulator.simulate_path(pomdp, steps=4, seed=1, scheduler=scheduler)

    # we make the path that the simulate path function should create

    action0 = pomdp.action("open2")
    action1 = pomdp.action("switch")

    other_path = simulator.Path(
        [
            (stormvogel.model.EmptyAction, pomdp.states[1]),
            (
                action0,
                pomdp.states[6],
            ),
            (stormvogel.model.EmptyAction, pomdp.states[16]),
            (
                action1,
                pomdp.states[32],
            ),
        ],
        pomdp,
    )

    assert path == other_path

    ##############################################################################################
    # we test the monty hall pomdp with a lambda as scheduler
    def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
        actions = state.available_actions()
        return actions[0]

    pomdp = stormvogel.examples.monty_hall_pomdp.create_monty_hall_pomdp()
    path = simulator.simulate_path(pomdp, steps=4, seed=1, scheduler=scheduler)

    action0 = pomdp.action("open0")
    action1 = pomdp.action("stay")

    # we make the path that the simulate path function should create
    other_path = simulator.Path(
        [
            (stormvogel.model.EmptyAction, pomdp.states[1]),
            (
                action0,
                pomdp.states[4],
            ),
            (stormvogel.model.EmptyAction, pomdp.states[13]),
            (
                action1,
                pomdp.states[25],
            ),
        ],
        pomdp,
    )

    assert path == other_path


def test_step_reward_uses_current_state_dtmc():
    """step() returns exit-state rewards (reward of current state) for DTMC."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    s1 = dtmc.new_state("s1")
    init.set_choices([(1.0, s1)])
    dtmc.add_self_loops()

    rm = dtmc.new_reward_model("R")
    rm.set_state_reward(init, 10)
    rm.set_state_reward(s1, 20)

    next_state, rewards, _ = simulator.step(init, seed=42)
    assert next_state == s1
    # Reward must be from init (the current/exiting state), not s1
    assert rewards == [10]


def test_step_reward_uses_current_state_mdp():
    """step() returns exit-state rewards (reward of current state) for MDP."""
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    act = mdp.new_action("go")
    s1 = mdp.new_state("s1")
    init.set_choices({act: [(1.0, s1)]})
    mdp.add_self_loops()

    rm = mdp.new_reward_model("R")
    rm.set_state_reward(init, 10)
    rm.set_state_reward(s1, 20)

    next_state, rewards, _ = simulator.step(init, action=act, seed=42)
    assert next_state == s1
    assert rewards == [10]


def test_step_missing_reward_defaults_to_zero():
    """step() returns 0 for states with no reward set."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    s1 = dtmc.new_state("s1")
    init.set_choices([(1.0, s1)])
    dtmc.add_self_loops()

    rm = dtmc.new_reward_model("R")
    rm.set_state_reward(s1, 5)  # only set for s1, not init

    _, rewards, _ = simulator.step(init, seed=42)
    assert rewards == [0]  # init has no reward, defaults to 0


def test_simulate_rewards_all_states_dtmc():
    """simulate() populates rewards for every discovered state in a DTMC."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    s1 = dtmc.new_state("s1")
    s2 = dtmc.new_state("s2")
    init.set_choices([(0.5, s1), (0.5, s2)])
    dtmc.add_self_loops()

    rm = dtmc.new_reward_model("R")
    rm.set_state_reward(init, 100)
    rm.set_state_reward(s1, 200)
    rm.set_state_reward(s2, 300)

    partial = simulator.simulate(dtmc, steps=1, runs=20, seed=1)
    assert partial is not None

    assert len(partial.rewards) == 1
    rm_partial = partial.rewards[0]
    for state in partial:
        r = rm_partial.get_state_reward(state)
        assert r is not None, f"State {list(state.labels)} has no reward"

    # Verify specific reward values by label
    for state in partial:
        r = rm_partial.get_state_reward(state)
        if state.has_label("init"):
            assert r == 100
        elif state.has_label("s1"):
            assert r == 200
        elif state.has_label("s2"):
            assert r == 300


def test_simulate_rewards_all_states_mdp():
    """simulate() populates rewards for every discovered state in an MDP."""
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    act = mdp.new_action("go")
    s1 = mdp.new_state("s1")
    s2 = mdp.new_state("s2")
    init.set_choices({act: [(0.5, s1), (0.5, s2)]})
    s1.set_choices({act: [(1.0, s2)]})
    mdp.add_self_loops()

    rm = mdp.new_reward_model("R")
    rm.set_state_reward(init, 10)
    rm.set_state_reward(s1, 20)
    rm.set_state_reward(s2, 30)

    def pick_first(state):
        return state.available_actions()[0]

    partial = simulator.simulate(mdp, steps=2, runs=10, seed=1, scheduler=pick_first)
    assert partial is not None

    assert len(partial.rewards) == 1
    rm_partial = partial.rewards[0]
    for state in partial:
        r = rm_partial.get_state_reward(state)
        assert r is not None, f"State {list(state.labels)} has no reward"

    # Verify specific reward values by label
    for state in partial:
        r = rm_partial.get_state_reward(state)
        if state.has_label("init"):
            assert r == 10
        elif state.has_label("s1"):
            assert r == 20
        elif state.has_label("s2"):
            assert r == 30


def test_simulate_missing_initial_reward_mdp():
    """simulate() does not crash when the initial state has no reward in an MDP."""
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    act = mdp.new_action("go")
    s1 = mdp.new_state("s1")
    init.set_choices({act: [(1.0, s1)]})
    mdp.add_self_loops()

    rm = mdp.new_reward_model("R")
    rm.set_state_reward(s1, 5)  # no reward for init

    def pick_first(state):
        return state.available_actions()[0]

    partial = simulator.simulate(mdp, steps=1, runs=1, seed=1, scheduler=pick_first)
    assert partial is not None

    rm_partial = partial.rewards[0]
    # init should get default 0
    assert rm_partial.get_state_reward(partial.initial_state) == 0
    # s1 should get 5
    for state in partial:
        if state.has_label("s1"):
            assert rm_partial.get_state_reward(state) == 5


def test_simulate_no_rewards():
    """simulate() works fine on models without any reward models."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    s1 = dtmc.new_state("s1")
    init.set_choices([(1.0, s1)])
    dtmc.add_self_loops()

    partial = simulator.simulate(dtmc, steps=1, runs=1, seed=1)
    assert partial is not None
    assert len(partial.rewards) == 0


def test_simulate_multiple_reward_models():
    """simulate() handles multiple reward models correctly."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    s1 = dtmc.new_state("s1")
    init.set_choices([(1.0, s1)])
    dtmc.add_self_loops()

    rm1 = dtmc.new_reward_model("cost")
    rm1.set_state_reward(init, 1)
    rm1.set_state_reward(s1, 2)

    rm2 = dtmc.new_reward_model("time")
    rm2.set_state_reward(init, 10)
    rm2.set_state_reward(s1, 20)

    partial = simulator.simulate(dtmc, steps=1, runs=1, seed=1)
    assert partial is not None
    assert len(partial.rewards) == 2

    rm1_p = partial.rewards[0]
    rm2_p = partial.rewards[1]
    assert rm1_p.name == "cost"
    assert rm2_p.name == "time"

    for state in partial:
        if state.has_label("init"):
            assert rm1_p.get_state_reward(state) == 1
            assert rm2_p.get_state_reward(state) == 10
        elif state.has_label("s1"):
            assert rm1_p.get_state_reward(state) == 2
            assert rm2_p.get_state_reward(state) == 20


def test_simulate_mdp_revisited_state_has_reward():
    """In an MDP, a state discovered as next_state but revisited via different
    actions must still have its reward set."""
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    act_a = mdp.new_action("a")
    act_b = mdp.new_action("b")
    s1 = mdp.new_state("s1")
    # Both actions lead to s1 deterministically
    init.set_choices({act_a: [(1.0, s1)], act_b: [(1.0, s1)]})
    mdp.add_self_loops()

    rm = mdp.new_reward_model("R")
    rm.set_state_reward(init, 42)
    rm.set_state_reward(s1, 99)

    # Scheduler alternates -- but random with seed will pick something.
    # With enough runs both actions should be tried, s1 discovered once.
    partial = simulator.simulate(mdp, steps=1, runs=10, seed=7)
    assert partial is not None

    rm_partial = partial.rewards[0]
    for state in partial:
        r = rm_partial.get_state_reward(state)
        assert r is not None, f"State {list(state.labels)} has no reward"
        if state.has_label("s1"):
            assert r == 99
