"""Tests for stormvogel.stormpy_utils.mec."""

import pytest

import stormvogel.examples as examples
from stormvogel.stormpy_utils.mec import detect_mecs, eliminate_mecs


@pytest.fixture
def ec_mdp():
    return examples.create_end_components_mdp()


def test_detect_mecs_count(ec_mdp):
    pytest.importorskip("stormpy")
    mecs = detect_mecs(ec_mdp)
    assert len(mecs) == 1


def test_detect_mecs_states(ec_mdp):
    pytest.importorskip("stormpy")
    mecs = detect_mecs(ec_mdp)
    (mec,) = mecs
    labels_in_mec = {frozenset(s.labels) for s in mec}
    assert frozenset({"mec1"}) in labels_in_mec
    assert frozenset({"mec2"}) in labels_in_mec


def test_detect_mecs_init_not_in_mec(ec_mdp):
    pytest.importorskip("stormpy")
    mecs = detect_mecs(ec_mdp)
    all_mec_states = {s for mec in mecs for s in mec}
    init_state = next(iter(ec_mdp.get_states_with_label("init")))
    assert init_state not in all_mec_states


def test_eliminate_mecs_state_count(ec_mdp):
    pytest.importorskip("stormpy")
    sv_new, _ = eliminate_mecs(ec_mdp)
    assert len(sv_new.states) == 2


def test_eliminate_mecs_mec_states_merge(ec_mdp):
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(ec_mdp)
    mec1 = next(iter(ec_mdp.get_states_with_label("mec1")))
    mec2 = next(iter(ec_mdp.get_states_with_label("mec2")))
    assert state_map[mec1] is state_map[mec2]


def test_eliminate_mecs_init_preserved(ec_mdp):
    pytest.importorskip("stormpy")
    sv_new, state_map = eliminate_mecs(ec_mdp)
    init_old = next(iter(ec_mdp.get_states_with_label("init")))
    init_new = next(iter(sv_new.get_states_with_label("init")))
    assert state_map[init_old] is init_new


def test_eliminate_mecs_merged_labels(ec_mdp):
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(ec_mdp)
    mec1 = next(iter(ec_mdp.get_states_with_label("mec1")))
    merged = state_map[mec1]
    merged_labels = set(merged.labels)
    assert "mec1" in merged_labels
    assert "mec2" in merged_labels


@pytest.fixture
def mixed_mec_mdp():
    return examples.create_mixed_mec_mdp()


def test_eliminate_mecs_no_rep_selfloop_state_count(mixed_mec_mdp):
    pytest.importorskip("stormpy")
    new_mdp, _ = eliminate_mecs(mixed_mec_mdp, remove_representative_selfloops=True)
    assert len(new_mdp.states) == 5


def test_eliminate_mecs_no_rep_selfloop_rep_has_one_action(mixed_mec_mdp):
    """Representative state (merged s3+s4) has only the escape action, no self-loop."""
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(mixed_mec_mdp, remove_representative_selfloops=True)
    old_s3 = next(s for s in mixed_mec_mdp.states if s.friendly_name == "s3")
    rep = state_map[old_s3]
    assert rep.nr_choices == 1


def test_eliminate_mecs_sink_keeps_selfloop(mixed_mec_mdp):
    """Trivial MEC sink (s5) is NOT touched by remove_representative_selfloops."""
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(mixed_mec_mdp, remove_representative_selfloops=True)
    old_s5 = next(s for s in mixed_mec_mdp.states if s.friendly_name == "s5")
    sink_new = state_map[old_s5]
    assert sink_new.nr_choices == 1  # still has its self-loop


def test_eliminate_mecs_absorbing_rep_has_one_action(mixed_mec_mdp):
    """make_representatives_absorbing leaves only the self-loop on the representative."""
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(mixed_mec_mdp, make_representatives_absorbing=True)
    old_s3 = next(s for s in mixed_mec_mdp.states if s.friendly_name == "s3")
    rep = state_map[old_s3]
    assert rep.nr_choices == 1
    # The single action must be a pure self-loop.
    _, branch = next(iter(rep.choices))
    assert all(succ is rep for _, succ in branch)


def test_eliminate_mecs_absorbing_sink_unchanged(mixed_mec_mdp):
    """make_representatives_absorbing does not touch trivial MEC (sink s5)."""
    pytest.importorskip("stormpy")
    _, state_map = eliminate_mecs(mixed_mec_mdp, make_representatives_absorbing=True)
    old_s5 = next(s for s in mixed_mec_mdp.states if s.friendly_name == "s5")
    assert state_map[old_s5].nr_choices == 1


def test_eliminate_mecs_mutually_exclusive(mixed_mec_mdp):
    pytest.importorskip("stormpy")
    with pytest.raises(ValueError):
        eliminate_mecs(
            mixed_mec_mdp,
            remove_representative_selfloops=True,
            make_representatives_absorbing=True,
        )
