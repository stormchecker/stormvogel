"""Tests for stormvogel.model.validation."""

import stormvogel.model as m
from stormvogel.model.validation import Severity, validate


def _simple_valid_mdp() -> m.Model:
    """Two-state MDP: init --a--> s1 (prob 1), s1 self-loops."""
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    s1 = mdp.new_state(["target"])
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s1)]})
    mdp.set_choices(s1, {a: [(1.0, s1)]})
    return mdp


# --- shared checks ---


def test_valid_mdp_has_no_issues():
    result = _simple_valid_mdp().validate()
    assert result.is_valid
    assert result.issues == []


def test_no_init_state_is_error():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state([])  # no "init" label
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s0)]})
    result = validate(mdp)
    assert not result.is_valid
    errors = [i.message for i in result.errors]
    assert any("init" in msg for msg in errors)


def test_multiple_init_states_is_error():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    s1 = mdp.new_state(["init"])
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s1)]})
    mdp.set_choices(s1, {a: [(1.0, s0)]})
    result = validate(mdp)
    assert not result.is_valid
    assert any("init" in i.message for i in result.errors)


def test_distribution_not_summing_to_one_is_error():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    s1 = mdp.new_state(["other"])
    a = mdp.new_action("go")
    # Bypass the normal set_choices to plant a bad distribution
    from stormvogel.model.choices import Choices
    from stormvogel.model.distribution import Distribution

    mdp.transitions[s0] = Choices({a: Distribution({s1: 0.5})})
    mdp.transitions[s1] = Choices({a: Distribution({s1: 1.0})})
    result = validate(mdp)
    assert not result.is_valid
    assert any("sum" in i.message or "1" in i.message for i in result.errors)


def test_transition_to_foreign_state_is_error():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    a = mdp.new_action("go")

    # Build a foreign state that is NOT in mdp.states
    foreign = m.new_mdp(create_initial_state=False).new_state(["other"])

    from stormvogel.model.choices import Choices
    from stormvogel.model.distribution import Distribution

    mdp.transitions[s0] = Choices({a: Distribution({foreign: 1.0})})
    result = validate(mdp)
    assert not result.is_valid
    assert any(i.context is foreign for i in result.errors)


# --- MDP-specific checks ---


def test_deadlock_state_is_warning():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    s1 = mdp.new_state(["dead"])  # no choices set -> deadlock
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s1)]})
    result = validate(mdp)
    assert result.is_valid  # warnings don't make it invalid
    assert any(
        i.severity == Severity.WARNING and i.context is s1 for i in result.warnings
    )


def test_unreachable_state_is_warning():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(["init"])
    s1 = mdp.new_state(["unreachable"])
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s0)]})
    mdp.set_choices(s1, {a: [(1.0, s0)]})
    result = validate(mdp)
    assert result.is_valid
    assert any(
        i.severity == Severity.WARNING and i.context is s1 for i in result.warnings
    )


def test_no_warnings_for_fully_reachable_mdp():
    result = _simple_valid_mdp().validate()
    assert result.warnings == []


# --- DTMC: only shared checks, no MDP-specific ones ---


def test_valid_dtmc_has_no_issues():
    dtmc = m.new_dtmc(create_initial_state=False)
    s0 = dtmc.new_state(["init"])
    s1 = dtmc.new_state(["done"])
    dtmc.set_choices(s0, [(1.0, s1)])
    dtmc.set_choices(s1, [(1.0, s1)])
    result = validate(dtmc)
    assert result.is_valid
    assert result.issues == []


# --- ValidationResult helpers ---


def test_validation_result_str_valid():
    result = _simple_valid_mdp().validate()
    assert "valid" in str(result)


def test_validation_result_str_invalid():
    mdp = m.new_mdp(create_initial_state=False)
    s0 = mdp.new_state([])
    a = mdp.new_action("go")
    mdp.set_choices(s0, {a: [(1.0, s0)]})
    result = validate(mdp)
    assert "invalid" in str(result)
    assert "ERROR" in str(result)
