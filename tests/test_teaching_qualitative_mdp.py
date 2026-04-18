"""Tests for stormvogel.teaching.qualitative_mdp."""

import stormvogel.model as sv_model
from stormvogel.examples.parker import create_parker_mdp
from stormvogel.teaching.qualitative_mdp import (
    FixpointIterator,
    psi_smaxas,
    psi_sminas,
    psi_spos,
    psi_sposmin,
    smaxas,
    sminas,
    spos,
    sposmin,
    visualise_iterations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _friendly_names(state_set):
    return {s.friendly_name for s in state_set}


def _run_to_fixpoint(it: FixpointIterator) -> list[frozenset]:
    """Return [initial, step1, step2, ...] up to and including the first repeat."""
    snaps = [it.current]
    while not it.has_converged():
        snaps.append(it.step())
    return snaps


def _parker_setup():
    mdp = create_parker_mdp()
    s2 = mdp.get_states_with_label("s2").pop()
    return mdp, s2


def _two_action_mdp():
    """Small MDP:

    s0 --a--> s_target
       --b--> s0          (self-loop)
    s_target --a--> s_target
    """
    mdp = sv_model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    st = mdp.new_state(labels=["target"])
    s0.set_friendly_name("s0")
    st.set_friendly_name("st")
    a = mdp.new_action("a")
    b = mdp.new_action("b")
    mdp.set_choices(s0, {a: [(1, st)], b: [(1, s0)]})
    mdp.set_choices(st, {a: [(1, st)]})
    return mdp, s0, st


def _isolated_mdp():
    """Small MDP:

    s0 --a--> s_target
    s_target --a--> s_target
    s_iso --a--> s_iso      (no path to s_target)
    """
    mdp = sv_model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    st = mdp.new_state(labels=["target"])
    si = mdp.new_state(labels=["iso"])
    s0.set_friendly_name("s0")
    st.set_friendly_name("st")
    si.set_friendly_name("si")
    a = mdp.new_action("a")
    mdp.set_choices(s0, {a: [(1, st)]})
    mdp.set_choices(st, {a: [(1, st)]})
    mdp.set_choices(si, {a: [(1, si)]})
    return mdp, s0, st, si


# ---------------------------------------------------------------------------
# Spos (possible max reachability)
# ---------------------------------------------------------------------------


def test_spos_parker_fixpoint():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(spos(mdp, [s2]))
    assert _friendly_names(snaps[-1]) == {"s0", "s1", "s2", "s3"}


def test_spos_target_always_in_initial():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    assert s2 in it.current


def test_spos_isolated_state_excluded():
    mdp, s0, st, si = _isolated_mdp()
    snaps = _run_to_fixpoint(spos(mdp, [st]))
    assert "si" not in _friendly_names(snaps[-1])
    assert "s0" in _friendly_names(snaps[-1])
    assert "st" in _friendly_names(snaps[-1])


def test_spos_lfp_grows():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(spos(mdp, [s2]))
    for prev, curr in zip(snaps, snaps[1:]):
        assert prev <= curr, "LFP must be monotone non-decreasing"


def test_spos_converges():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    while not it.has_converged():
        it.step()
    assert it.has_converged()


# ---------------------------------------------------------------------------
# Sposmin (possible min reachability)
# ---------------------------------------------------------------------------


def test_sposmin_parker_excludes_s3():
    """s3 has action b (self-loop), so the minimiser can avoid the target."""
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(sposmin(mdp, [s2]))
    fp = _friendly_names(snaps[-1])
    assert fp == {"s0", "s1", "s2"}
    assert "s3" not in fp


def test_sposmin_target_always_in_initial():
    mdp, s2 = _parker_setup()
    it = sposmin(mdp, [s2])
    assert s2 in it.current


def test_sposmin_self_loop_action_excluded():
    """s0 in _two_action_mdp has action b (self-loop), so it is NOT in Sposmin."""
    mdp, s0, st = _two_action_mdp()
    snaps = _run_to_fixpoint(sposmin(mdp, [st]))
    fp = _friendly_names(snaps[-1])
    # st (target) is always in Sposmin; s0 has a self-loop action so it is excluded
    assert "st" in fp
    assert "s0" not in fp


def test_sposmin_lfp_grows():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(sposmin(mdp, [s2]))
    for prev, curr in zip(snaps, snaps[1:]):
        assert prev <= curr, "LFP must be monotone non-decreasing"


def test_sposmin_intermediate_steps_parker():
    """s1 joins before s0 because s0's action b leads to s1 which needs to be in X first."""
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(sposmin(mdp, [s2]))
    # step 0: just {s2}
    assert _friendly_names(snaps[0]) == {"s2"}
    # step 1: s1 added (action a has s2 as successor)
    assert "s1" in _friendly_names(snaps[1])
    assert "s0" not in _friendly_names(snaps[1])
    # step 2: s0 added (action b → s1 now in X)
    assert "s0" in _friendly_names(snaps[2])


def test_sposmin_subset_of_spos():
    """Every state reachable by all policies is also reachable by some policy."""
    mdp, s2 = _parker_setup()
    fp_sposmin = _run_to_fixpoint(sposmin(mdp, [s2]))[-1]
    fp_spos = _run_to_fixpoint(spos(mdp, [s2]))[-1]
    assert fp_sposmin <= fp_spos


# ---------------------------------------------------------------------------
# Smaxas (almost-sure max reachability)
# ---------------------------------------------------------------------------


def test_smaxas_parker_fixpoint():
    """Every Parker state has an action achieving P=1, so Smaxas = S."""
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(smaxas(mdp, [s2]))
    assert _friendly_names(snaps[-1]) == {"s0", "s1", "s2", "s3"}


def test_smaxas_target_in_result():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(smaxas(mdp, [s2]))
    assert s2 in snaps[-1]


def test_smaxas_gfp_shrinks_or_stays():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(smaxas(mdp, [s2]))
    for prev, curr in zip(snaps, snaps[1:]):
        assert curr <= prev, "GFP must be monotone non-increasing"


# ---------------------------------------------------------------------------
# FixpointIterator
# ---------------------------------------------------------------------------


def test_has_converged_false_before_any_step():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    assert not it.has_converged()


def test_has_converged_true_after_stable_step():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    while not it.has_converged():
        it.step()
    # An extra step on a converged iterator should keep has_converged True
    it.step()
    assert it.has_converged()


def test_operator_directly_callable():
    """psi_* functions can be used directly, not just via convenience constructors."""
    mdp, s2 = _parker_setup()
    target = frozenset([s2])
    result = psi_spos(target, mdp, target)
    assert s2 in result

    result2 = psi_sposmin(target, mdp, target)
    assert s2 in result2

    result3 = psi_smaxas(frozenset(mdp.states), mdp, target)
    assert s2 in result3

    result4 = psi_sminas(frozenset(mdp.states), mdp, target)
    assert s2 in result4


# ---------------------------------------------------------------------------
# Sminas (almost-sure min reachability)
# ---------------------------------------------------------------------------


def test_sminas_parker_only_target():
    """Min reachability < 1 for s0, s1, s3 in Parker; only s2 is in Sminas."""
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(sminas(mdp, [s2]))
    assert _friendly_names(snaps[-1]) == {"s2"}


def test_sminas_initial_set_is_sposmin():
    """sminas() initialises from Sposmin, not S."""
    mdp, s2 = _parker_setup()
    it_sminas = sminas(mdp, [s2])
    it_sposmin = sposmin(mdp, [s2])
    while not it_sposmin.has_converged():
        it_sposmin.step()
    assert it_sminas.current == it_sposmin.current


def test_sminas_gfp_shrinks_or_stays():
    mdp, s2 = _parker_setup()
    snaps = _run_to_fixpoint(sminas(mdp, [s2]))
    for prev, curr in zip(snaps, snaps[1:]):
        assert curr <= prev, "GFP must be monotone non-increasing"


def test_sminas_subset_of_smaxas():
    """Every state reachable with P=1 by all policies is also reachable by some policy."""
    mdp, s2 = _parker_setup()
    fp_sminas = _run_to_fixpoint(sminas(mdp, [s2]))[-1]
    fp_smaxas = _run_to_fixpoint(smaxas(mdp, [s2]))[-1]
    assert fp_sminas <= fp_smaxas


def test_sminas_chain_mdp():
    """s0 -> s1 -> s_target (all deterministic); every policy gives P=1 from s0 and s1."""
    mdp = sv_model.new_mdp(create_initial_state=False)
    s0 = mdp.new_state(labels=["init"])
    s1 = mdp.new_state(labels=["mid"])
    st = mdp.new_state(labels=["target"])
    s0.set_friendly_name("s0")
    s1.set_friendly_name("s1")
    st.set_friendly_name("st")
    a = mdp.new_action("a")
    mdp.set_choices(s0, {a: [(1, s1)]})
    mdp.set_choices(s1, {a: [(1, st)]})
    mdp.set_choices(st, {a: [(1, st)]})

    snaps = _run_to_fixpoint(sminas(mdp, [st]))
    assert _friendly_names(snaps[-1]) == {"s0", "s1", "st"}


def test_sminas_self_loop_excluded():
    """In _two_action_mdp, s0 has action b (self-loop) so it is not in Sminas."""
    mdp, s0, st = _two_action_mdp()
    snaps = _run_to_fixpoint(sminas(mdp, [st]))
    fp = _friendly_names(snaps[-1])
    assert "st" in fp
    assert "s0" not in fp


# ---------------------------------------------------------------------------
# visualise_iterations
# ---------------------------------------------------------------------------


def test_visualise_iterations_shape():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    snaps = _run_to_fixpoint(it)
    df = visualise_iterations(snaps, mdp, highlight=False)
    assert df.shape == (len(mdp.states), len(snaps))


def test_visualise_iterations_target_true_in_all_columns():
    mdp, s2 = _parker_setup()
    it = spos(mdp, [s2])
    snaps = _run_to_fixpoint(it)
    df = visualise_iterations(snaps, mdp, highlight=False)
    assert df.loc[s2.friendly_name].all()


def test_visualise_iterations_s3_false_in_sposmin():
    """s3 is never in Sposmin, so its row should be all False."""
    mdp, s2 = _parker_setup()
    it = sposmin(mdp, [s2])
    snaps = _run_to_fixpoint(it)
    df = visualise_iterations(snaps, mdp, highlight=False)
    assert not df.loc["s3"].any()
