import pytest

import stormvogel.examples.die
import stormvogel.examples.monty_hall
import stormvogel.examples.nuclear_fusion_ctmc
import stormvogel.examples.monty_hall_pomdp
from stormvogel.examples.lion import create_lion_mdp
import stormvogel.model
import stormvogel.result
from stormvogel.model.variable import Variable

gym = pytest.importorskip("gymnasium")
import stormvogel.simulator as simulator
from stormvogel.gym_env import ActionUnavailableError, ModelEnv
from model_testing import assert_models_equal, assert_paths_equal


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
    init.valuations = {Variable("rolled"): 0}
    init.set_choices(
        [
            (
                1 / 6,
                other_dtmc.new_state("rolled2", valuations={Variable("rolled"): 2}),
            ),
            (
                1 / 6,
                other_dtmc.new_state("rolled4", valuations={Variable("rolled"): 4}),
            ),
            (
                1 / 6,
                other_dtmc.new_state("rolled5", valuations={Variable("rolled"): 5}),
            ),
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

    assert_models_equal(partial_model, other_dtmc)
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
    sched_obj = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=sched_obj
    )

    # Verify structural properties of the partial model
    assert partial_model is not None
    assert len(partial_model.states) == 4
    assert any(s.has_label("init") for s in partial_model)
    assert any(s.has_label("carchosen") for s in partial_model)
    assert any(s.has_label("open") for s in partial_model)
    assert any(s.has_label("goatrevealed") for s in partial_model)
    # Verify rewards are correctly copied from original
    assert len(partial_model.rewards) == len(mdp.rewards)
    for ps in partial_model:
        # Find corresponding original state
        orig = next(
            s
            for s in mdp
            if set(s.labels) == set(ps.labels) and s.valuations == ps.valuations
        )
        for ri, rm in enumerate(partial_model.rewards):
            expected = mdp.rewards[ri].get_state_reward(orig)
            actual = rm.get_state_reward(ps)
            assert actual == (
                expected if expected is not None else 0
            ), f"Reward mismatch for {list(ps.labels)}: {actual} != {expected}"
    ######################################################################################################################

    # we test the simulator for an mdp with a lambda as Scheduler

    def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
        actions = state.available_actions()
        return actions[0]

    mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()

    partial_model = simulator.simulate(
        mdp, runs=1, steps=3, seed=1, scheduler=scheduler
    )

    # Verify structural properties of the partial model
    assert partial_model is not None
    assert len(partial_model.states) == 4
    assert any(s.has_label("init") for s in partial_model)
    assert any(s.has_label("carchosen") for s in partial_model)
    assert any(s.has_label("open") for s in partial_model)
    assert any(s.has_label("goatrevealed") for s in partial_model)

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
                hunt: stormvogel.model.Distribution(
                    [(0.5, satisfied), (0.3, full), (0.2, hungry)]
                ),
            }
        )
    )

    hungry.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Distribution(
                    [(0.2, full), (0.5, satisfied), (0.2, starving)]
                ),
            }
        )
    )

    full.set_choices(
        stormvogel.model.Choices(
            {
                hunt: stormvogel.model.Distribution(
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
                hunt: stormvogel.model.Distribution([(0.2, hungry)]),
            }
        )
    )

    lion.add_self_loops()

    reward_model = lion.new_reward_model("R")
    reward_model.set_state_reward(full, 100)
    reward_model.set_unset_rewards(0)

    assert_models_equal(lion, partial_model)


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

    assert_paths_equal(path, other_path)
    ##############################################################################################
    # we make the monty hall pomdp and run simulate path with it
    pomdp = stormvogel.examples.monty_hall_pomdp.create_monty_hall_pomdp()
    taken_actions = {}
    for state in pomdp:
        taken_actions[state] = state.available_actions()[
            len(state.available_actions()) - 1
        ]
    sched_obj = stormvogel.result.Scheduler(pomdp, taken_actions)
    path = simulator.simulate_path(pomdp, steps=4, seed=1, scheduler=sched_obj)

    # Verify the path is valid (each transition exists in the model)
    assert len(path.path) == 4  # steps=4
    for i, step in enumerate(path.path):
        s = step[1] if isinstance(step, tuple) else step
        assert s in pomdp.states, f"Step {i} state not in model"

    ##############################################################################################
    # we test the monty hall pomdp with a lambda as scheduler
    def scheduler(state: stormvogel.model.State) -> stormvogel.model.Action:
        actions = state.available_actions()
        return actions[0]

    pomdp = stormvogel.examples.monty_hall_pomdp.create_monty_hall_pomdp()
    path = simulator.simulate_path(pomdp, steps=4, seed=1, scheduler=scheduler)

    # Verify the path is valid (each transition exists in the model)
    assert len(path.path) == 4  # steps=4
    for i, step in enumerate(path.path):
        s = step[1] if isinstance(step, tuple) else step
        assert s in pomdp.states, f"Step {i} state not in model"


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


def _simple_dtmc():
    dtmc = stormvogel.model.new_dtmc()
    s1 = dtmc.new_state("s1")
    dtmc.initial_state.set_choices([(1.0, s1)])
    dtmc.add_self_loops()
    return dtmc


def _simple_mdp():
    mdp = stormvogel.model.new_mdp()
    act = mdp.new_action("go")
    s1 = mdp.new_state("s1")
    mdp.initial_state.set_choices([(act, s1)])
    mdp.add_self_loops()
    return mdp, act, s1


def test_path_init_ma_raises():
    """Path.__init__ raises NotImplementedError for MA models."""
    ma = stormvogel.model.new_ma()
    with pytest.raises(NotImplementedError):
        simulator.Path([], ma)


def test_path_get_state_in_step_dtmc():
    """Path.get_state_in_step for a DTMC path."""
    dtmc = _simple_dtmc()
    s1 = dtmc.states[1]
    path = simulator.Path([s1], dtmc)
    assert path.get_state_in_step(0) == s1


def test_path_get_state_in_step_mdp():
    """Path.get_state_in_step for an MDP path."""
    mdp, act, s1 = _simple_mdp()
    path = simulator.Path([(act, s1)], mdp)
    assert path.get_state_in_step(0) == s1


def test_path_get_action_in_step():
    """Path.get_action_in_step for an MDP path."""
    mdp, act, s1 = _simple_mdp()
    path = simulator.Path([(act, s1)], mdp)
    assert path.get_action_in_step(0) == act


def test_path_get_step():
    """Path.get_step returns the raw entry."""
    mdp, act, s1 = _simple_mdp()
    path = simulator.Path([(act, s1)], mdp)
    assert path.get_step(0) == (act, s1)


def test_path_to_state_action_sequence_mdp():
    """Path.to_state_action_sequence for an MDP path."""
    mdp, act, s1 = _simple_mdp()
    path = simulator.Path([(act, s1)], mdp)
    seq = path.to_state_action_sequence()
    assert mdp.initial_state in seq
    assert act in seq
    assert s1 in seq


def test_path_str_mdp():
    """Path.__str__ for an MDP path."""
    mdp, act, s1 = _simple_mdp()
    path = simulator.Path([(act, s1)], mdp)
    s = str(path)
    assert "action" in s


def test_path_str_dtmc():
    """Path.__str__ for a DTMC path."""
    dtmc = _simple_dtmc()
    s1 = dtmc.states[1]
    path = simulator.Path([s1], dtmc)
    s = str(path)
    assert "state" in s


def test_path_len():
    """Path.__len__."""
    dtmc = _simple_dtmc()
    s1 = dtmc.states[1]
    path = simulator.Path([s1, s1], dtmc)
    assert len(path) == 2


def test_get_action_at_state_type_error():
    """get_action_at_state raises TypeError for invalid scheduler."""
    dtmc = _simple_dtmc()
    with pytest.raises(TypeError):
        simulator.get_action_at_state(dtmc.initial_state, 42)  # type: ignore


# --- ModelEnv tests ---


def _gym_mdp():
    """Two-state MDP: init --(a)--> goal (absorbing), init --(b)--> init."""
    mdp = stormvogel.model.new_mdp()
    init = mdp.initial_state
    goal = mdp.new_state("goal")
    a = mdp.new_action("a")
    b = mdp.new_action("b")
    init.set_choices({a: [(1.0, goal)], b: [(1.0, init)]})
    r = mdp.new_reward_model("R")
    r.set_state_reward(init, 1.0)
    r.set_state_reward(goal, 0.0)
    return mdp


def _gym_mdp_with_valuations():
    """_gym_mdp extended with a domain-bearing variable 'x' on each state."""
    from stormvogel.model.variable import (
        Variable,
        IntDomain,
        BoolDomain,
        CategoricalDomain,
    )

    mdp = _gym_mdp()
    x = Variable("x", IntDomain(0, 1))
    done = Variable("done", BoolDomain())
    dir_ = Variable("dir", CategoricalDomain(("left", "right")))
    for i, s in enumerate(mdp.states):
        s.add_valuation(x, i)
        s.add_valuation(done, i == 1)
        s.add_valuation(dir_, "left" if i == 0 else "right")
    return mdp


def test_model_env_rejects_ctmc():
    ctmc = stormvogel.model.new_ctmc()
    with pytest.raises(ValueError, match="MDP, DTMC, POMDP, or HMM"):
        ModelEnv(ctmc)


def test_mdp_env_multiple_rewards_no_name():
    mdp = _gym_mdp()
    mdp.new_reward_model("R2")
    with pytest.raises(ValueError, match="multiple reward models"):
        ModelEnv(mdp)


def test_mdp_env_reset():
    mdp = _gym_mdp()
    env = ModelEnv(mdp)
    obs, info = env.reset(seed=0)
    assert obs == env._state_to_index[mdp.initial_state]
    assert info["state"] is mdp.initial_state


def test_mdp_env_step_valid():
    mdp = _gym_mdp()
    env = ModelEnv(mdp)
    env.reset(seed=0)
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    obs, _reward, terminated, truncated, _info = env.step(a_idx)
    assert not truncated
    assert terminated  # goal is absorbing
    assert isinstance(obs, int)
    assert 0 <= obs < env.observation_space.n


def test_mdp_env_action_unavailable_error():
    mdp = _gym_mdp()
    env = ModelEnv(mdp)
    env.reset()
    # "b" goes back to init; manually move to goal first by patching current state
    goal = next(s for s in mdp.states if s.has_label("goal"))
    env._current_state = goal
    b_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "b")
    ]
    with pytest.raises(ActionUnavailableError):
        env.step(b_idx)


def test_mdp_env_reward():
    mdp = _gym_mdp()
    env = ModelEnv(mdp, reward_model_name="R")
    env.reset()
    # Stepping from init with action "a" gives state-exit reward of init = 1.0
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    _, reward, _, _, _ = env.step(a_idx)
    assert reward == 1.0


# --- obs_type="valuations" tests ---


def test_mdp_env_valuations_obs_space():
    mdp = _gym_mdp_with_valuations()
    env = ModelEnv(mdp, obs_type="valuations")
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Dict)
    # dir: CategoricalDomain(("left","right")) → Discrete(2)
    assert obs_space["dir"].n == 2
    # done: BoolDomain → Discrete(2)
    assert obs_space["done"].n == 2
    # x: IntDomain(0,1) → Discrete(2)
    assert obs_space["x"].n == 2


def test_mdp_env_valuations_reset_returns_dict():
    mdp = _gym_mdp_with_valuations()
    env = ModelEnv(mdp, obs_type="valuations")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"x", "done", "dir"}
    # init state: x=0, done=False, dir="left"
    assert obs["x"] == 0  # 0 - lo(0) = 0
    assert obs["done"] == 0  # False → 0
    assert obs["dir"] == 0  # "left" is index 0


def test_mdp_env_valuations_step_returns_dict():
    mdp = _gym_mdp_with_valuations()
    env = ModelEnv(mdp, obs_type="valuations")
    env.reset()
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    obs, _, terminated, _, _ = env.step(a_idx)
    assert isinstance(obs, dict)
    assert terminated
    # goal state: x=1, done=True, dir="right"
    assert obs["x"] == 1
    assert obs["done"] == 1
    assert obs["dir"] == 1


def test_mdp_env_valuations_no_domain_vars_raises():
    mdp = _gym_mdp()  # no domain-bearing variables
    with pytest.raises(ValueError, match="domain"):
        ModelEnv(mdp, obs_type="valuations")


def test_mdp_env_index_unchanged():
    """obs_type='index' (default) still returns plain int observations."""
    mdp = _gym_mdp_with_valuations()
    env = ModelEnv(mdp, obs_type="index")
    obs, _ = env.reset()
    assert isinstance(obs, int)


# --- DTMC ModelEnv tests ---


def _gym_dtmc():
    """Two-state DTMC: init --(0.5)--> init, --(0.5)--> goal (absorbing)."""
    dtmc = stormvogel.model.new_dtmc()
    init = dtmc.initial_state
    goal = dtmc.new_state("goal")
    init.set_choices([(0.5, init), (0.5, goal)])
    return dtmc


def test_dtmc_env_action_space():
    dtmc = _gym_dtmc()
    env = ModelEnv(dtmc)
    assert env.action_space.n == 1


def test_dtmc_env_reset():
    dtmc = _gym_dtmc()
    env = ModelEnv(dtmc)
    obs, info = env.reset(seed=0)
    assert obs == env._state_to_index[dtmc.initial_state]
    assert info["state"] is dtmc.initial_state


def test_dtmc_env_step():
    dtmc = _gym_dtmc()
    env = ModelEnv(dtmc)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, int)
    assert reward == 0.0
    assert not truncated
    assert "state" in info


def test_dtmc_env_step_invalid_action():
    dtmc = _gym_dtmc()
    env = ModelEnv(dtmc)
    env.reset()
    with pytest.raises(ActionUnavailableError):
        env.step(1)


# --- POMDP / HMM ModelEnv tests ---


def _gym_pomdp():
    """Three-state POMDP with two observations (no valuations)."""
    pomdp = stormvogel.model.new_pomdp()
    obs_init = pomdp.get_observation("init")
    obs_goal = pomdp.new_observation("goal")

    init = pomdp.initial_state
    goal = pomdp.new_state("goal", observation=obs_goal)
    loop = pomdp.new_state("loop", observation=obs_init)  # shares obs with init

    a = pomdp.new_action("a")
    b = pomdp.new_action("b")
    init.set_choices({a: [(1.0, goal)], b: [(1.0, loop)]})
    loop.set_choices({a: [(1.0, goal)], b: [(1.0, loop)]})
    goal.set_choices({a: [(1.0, goal)], b: [(1.0, goal)]})
    return pomdp


def _gym_pomdp_with_obs_valuations():
    """Same POMDP but with a domain-bearing variable 'y' on each observation."""
    from stormvogel.model.variable import Variable, IntDomain

    pomdp = _gym_pomdp()
    y = Variable("y", IntDomain(0, 1))
    obs_init = pomdp.get_observation("init")
    obs_goal = pomdp.get_observation("goal")
    pomdp.observation_valuations[obs_init] = {y: 0}
    pomdp.observation_valuations[obs_goal] = {y: 1}
    return pomdp


def _gym_hmm():
    """Two-state HMM: init loops with prob 0.5, goes to goal with prob 0.5."""
    hmm = stormvogel.model.new_hmm()
    obs_goal = hmm.new_observation("goal")

    init = hmm.initial_state
    goal = hmm.new_state("goal", observation=obs_goal)
    init.set_choices([(0.5, init), (0.5, goal)])
    goal.set_choices([(1.0, goal)])
    return hmm


def test_pomdp_env_rejects_unsupported():
    """MA is not accepted."""
    ma = stormvogel.model.new_ma()
    with pytest.raises(ValueError, match="MDP, DTMC, POMDP, or HMM"):
        ModelEnv(ma)


def test_pomdp_env_index_obs_space():
    """index mode: observation_space size equals number of distinct observations."""
    pomdp = _gym_pomdp()
    env = ModelEnv(pomdp)
    # 2 observations: "init" and "goal"
    assert env.observation_space.n == 2


def test_pomdp_env_index_reset():
    """reset returns an integer observation index, not a state index."""
    pomdp = _gym_pomdp()
    env = ModelEnv(pomdp)
    obs, info = env.reset()
    assert isinstance(obs, int)
    assert 0 <= obs < env.observation_space.n
    assert info["state"] is pomdp.initial_state


def test_pomdp_env_index_step_maps_to_observation():
    """step returns the observation index of the next state."""
    pomdp = _gym_pomdp()
    env = ModelEnv(pomdp)
    env.reset()
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    obs, _reward, terminated, truncated, info = env.step(a_idx)
    assert isinstance(obs, int)
    assert not truncated
    # The returned index must equal the index of the next state's observation.
    next_obs = env._observation_to_index[info["state"].observation]
    assert obs == next_obs


def test_pomdp_env_shared_observation():
    """Two distinct states sharing one observation return the same obs index."""
    pomdp = _gym_pomdp()
    env = ModelEnv(pomdp)
    obs_init = pomdp.get_observation("init")
    init_idx = env._observation_to_index[obs_init]

    loop = next(s for s in pomdp.states if s.has_label("loop"))
    env._current_state = loop
    obs, _r, _term, _trunc, _info = env.step(
        env._action_to_index[next(x for x in env._index_to_action if x.label == "b")]
    )
    # loop --b--> loop, loop shares obs_init
    assert obs == init_idx


def test_pomdp_env_action_space():
    """POMDP has an action space reflecting its actions."""
    pomdp = _gym_pomdp()
    env = ModelEnv(pomdp)
    assert env.action_space.n == 2


def test_pomdp_env_valuations_obs_space():
    """valuations mode: observation_space is a Dict built from observation variables."""
    pomdp = _gym_pomdp_with_obs_valuations()
    env = ModelEnv(pomdp, obs_type="valuations")
    obs_space = env.observation_space
    assert isinstance(obs_space, gym.spaces.Dict)
    # y: IntDomain(0, 1) → Discrete(2)
    assert obs_space["y"].n == 2


def test_pomdp_env_valuations_reset():
    """valuations reset returns a dict encoding the initial observation."""
    pomdp = _gym_pomdp_with_obs_valuations()
    env = ModelEnv(pomdp, obs_type="valuations")
    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert obs["y"] == 0  # obs_init has y=0


def test_pomdp_env_valuations_step():
    """After stepping to goal, valuations encode obs_goal."""
    pomdp = _gym_pomdp_with_obs_valuations()
    env = ModelEnv(pomdp, obs_type="valuations")
    env.reset()
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    obs, _r, _term, _trunc, _info = env.step(a_idx)
    assert isinstance(obs, dict)
    assert obs["y"] == 1  # obs_goal has y=1


def test_pomdp_env_valuations_missing_var_raises():
    """Stepping into a state whose observation lacks a required variable raises ValueError."""
    from stormvogel.model.variable import Variable, IntDomain

    pomdp = _gym_pomdp()
    y = Variable("y", IntDomain(0, 1))
    # Only annotate obs_init; leave obs_goal empty so a step to goal triggers the error.
    obs_init = pomdp.get_observation("init")
    pomdp.observation_valuations[obs_init] = {y: 0}

    env = ModelEnv(pomdp, obs_type="valuations")
    env.reset()
    a_idx = env._action_to_index[
        next(x for x in env._index_to_action if x.label == "a")
    ]
    with pytest.raises(ValueError, match="has no value for variable"):
        env.step(a_idx)


def test_pomdp_env_valuations_no_domain_vars_raises():
    """valuations mode requires at least one domain-bearing observation variable."""
    pomdp = _gym_pomdp()  # observations have empty valuations
    with pytest.raises(ValueError, match="domain"):
        ModelEnv(pomdp, obs_type="valuations")


def test_hmm_env_action_space():
    """HMM has no actions — action space is Discrete(1)."""
    hmm = _gym_hmm()
    env = ModelEnv(hmm)
    assert env.action_space.n == 1


def test_hmm_env_index_obs_space():
    """HMM index mode observation space is Discrete(n_observations)."""
    hmm = _gym_hmm()
    env = ModelEnv(hmm)
    assert env.observation_space.n == 2


def test_hmm_env_reset_and_step():
    """HMM reset returns an observation index; step with action 0 advances."""
    hmm = _gym_hmm()
    env = ModelEnv(hmm)
    obs, info = env.reset()
    assert isinstance(obs, int)
    assert info["state"] is hmm.initial_state
    obs2, reward, _term, truncated, info2 = env.step(0)
    assert isinstance(obs2, int)
    assert reward == 0.0
    assert not truncated
    assert "state" in info2
