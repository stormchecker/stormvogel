from stormvogel.model.model import new_pomdp
from stormvogel.model.action import Action
from stormvogel.model.observation import Observation
from stormvogel.model.distribution import Distribution


def test_make_observations_deterministic():
    model = new_pomdp(create_initial_state=False)

    # Create initial state with an action leading to a state with multiple observations
    init_obs = model.new_observation("init")
    obs1 = model.new_observation("obs1")
    obs2 = model.new_observation("obs2")
    s0 = model.new_state(labels=["init"], observation=init_obs)

    obs_dist = Distribution([(0.3, obs1), (0.7, obs2)])
    s1 = model.new_state(observation=obs_dist)

    action1 = Action("action1")
    s0.set_choices({action1: [(1.0, s1)]})

    # Apply method
    model.make_observations_deterministic()

    # Check that s1 has been removed
    assert s1 not in model.states

    # Check that there are now states with deterministic observations instead
    new_s1_states = [s for s in model.states if s != s0]
    assert len(new_s1_states) == 2

    # Get values correctly
    obs_set = {s.observation.alias for s in new_s1_states if s.observation is not None}
    assert obs_set == {"obs1", "obs2"}

    # Check that transitions from s0 have been distributed correctly
    s0_transitions = list(model.transitions[s0][action1])
    assert len(s0_transitions) == 2

    prob_obs1 = next(
        prob
        for prob, target in s0_transitions
        if target.observation is not None and target.observation.alias == "obs1"
    )
    prob_obs2 = next(
        prob
        for prob, target in s0_transitions
        if target.observation is not None and target.observation.alias == "obs2"
    )

    assert abs(float(prob_obs1) - 0.3) < 1e-6
    assert abs(float(prob_obs2) - 0.7) < 1e-6


def test_observation_id_distinct():
    """Each Observation instance must get its own UUID, not a shared one."""
    model = new_pomdp(create_initial_state=False)
    obs_a = model.new_observation("a")
    obs_b = model.new_observation("b")
    assert obs_a.observation_id != obs_b.observation_id


def test_observation_id_explicit_preserved():
    """An explicitly supplied observation_id must not be overwritten."""
    from uuid import uuid4

    model = new_pomdp(create_initial_state=False)
    explicit = uuid4()
    obs = Observation(model=model, observation_id=explicit)
    assert obs.observation_id == explicit
