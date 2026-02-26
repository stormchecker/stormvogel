import stormvogel.examples.die
import stormvogel.examples.monty_hall
import stormvogel.examples.nuclear_fusion_ctmc
import stormvogel.examples.monty_hall_pomdp
from stormvogel.examples.lion import create_lion_mdp
from stormvogel.model import EmptyAction
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

    action0 = pomdp.get_action_with_label("open2")
    assert action0 is not None
    action1 = pomdp.get_action_with_label("switch")
    assert action1 is not None

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

    action0 = pomdp.get_action_with_label("open0")
    assert action0 is not None
    action1 = pomdp.get_action_with_label("stay")
    assert action1 is not None

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
