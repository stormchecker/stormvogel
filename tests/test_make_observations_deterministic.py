from stormvogel.model.model import new_pomdp
from stormvogel.model.branches import Branches
from stormvogel.model.action import Action
from stormvogel.model.observation import Observation
from stormvogel.model.distribution import Distribution


def test_make_observations_deterministic():
    model = new_pomdp(create_initial_state=False)

    # Create initial state with an action leading to a state with multiple observations
    s0 = model.new_state(observation=Observation("init"))

    obs_dist = Distribution([(0.3, Observation("obs1")), (0.7, Observation("obs2"))])
    s1 = model.new_state(observation=obs_dist)

    action1 = Action("action1")

    model.initial_states = [s0]
    model.choices[s0].choices[action1] = Branches([(1.0, s1)])

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
    s0_transitions = list(model.choices[s0].choices[action1])
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
    obs_a = Observation(alias="a")
    obs_b = Observation(alias="b")
    assert obs_a.observation_id != obs_b.observation_id


def test_observation_id_explicit_preserved():
    """An explicitly supplied observation_id must not be overwritten."""
    from uuid import uuid4

    explicit = uuid4()
    obs = Observation(alias="x", observation_id=explicit)
    assert obs.observation_id == explicit
