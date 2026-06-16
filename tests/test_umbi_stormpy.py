"""Tests comparing stormpy→stormvogel vs stormpy→umb→stormvogel paths."""

import tempfile
import os

import pytest

stormpy = pytest.importorskip("stormpy")

import stormpy.examples.files as F  # noqa: E402

import stormvogel.umbi as svu  # noqa: E402
from stormvogel.stormpy_utils.mapping import stormpy_to_stormvogel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(
    prism_path,
    *,
    state_valuations=False,
    choice_labels=False,
    observation_valuations=False,
):
    """Build a stormpy sparse model from a PRISM file."""
    prog = stormpy.parse_prism_program(prism_path)
    opts = stormpy.BuilderOptions()
    opts.set_build_state_valuations(state_valuations)
    opts.set_build_choice_labels(choice_labels)
    if observation_valuations:
        opts.set_build_observation_valuations(True)
    return stormpy.build_sparse_model_with_options(prog, opts)


def _via_umb(sp):
    """Export a stormpy model to a .umb file and read it back via UMBI."""
    with tempfile.NamedTemporaryFile(suffix=".umb", delete=False) as f:
        path = f.name
    try:
        stormpy.export_to_umb(sp, path)
        return svu.read_from_umb(path, ignore_unsupported_rewards=True)
    finally:
        os.unlink(path)


def _assert_same_structure(m_sp, m_umb):
    assert m_sp.model_type == m_umb.model_type
    assert m_sp.nr_states == m_umb.nr_states
    assert m_sp.nr_choices == m_umb.nr_choices


# ---------------------------------------------------------------------------
# DTMC
# ---------------------------------------------------------------------------


def test_dtmc_die_stormpy_vs_umb():
    sp = _build(F.prism_dtmc_die)
    m_sp = stormpy_to_stormvogel(sp)
    m_umb = _via_umb(sp)
    _assert_same_structure(m_sp, m_umb)


def test_dtmc_die_labels():
    sp = _build(F.prism_dtmc_die)
    m_umb = _via_umb(sp)
    # stormpy's UMB exporter writes only a subset of AP labels; those must be
    # a subset of the labels present in the stormpy-imported model
    m_sp = stormpy_to_stormvogel(sp)
    assert m_umb.state_labels.keys() <= m_sp.state_labels.keys()


# ---------------------------------------------------------------------------
# MDP
# ---------------------------------------------------------------------------


def test_mdp_coin_stormpy_vs_umb():
    sp = _build(F.prism_mdp_coin_2_2, choice_labels=True)
    m_sp = stormpy_to_stormvogel(sp)
    m_umb = _via_umb(sp)
    _assert_same_structure(m_sp, m_umb)


# ---------------------------------------------------------------------------
# CTMC
# ---------------------------------------------------------------------------


def test_ctmc_stormpy_vs_umb():
    sp = stormpy.build_model_from_drn(F.drn_ctmc_dft)
    m_sp = stormpy_to_stormvogel(sp)
    m_umb = _via_umb(sp)
    _assert_same_structure(m_sp, m_umb)


def test_ctmc_model_type():
    sp = stormpy.build_model_from_drn(F.drn_ctmc_dft)
    from stormvogel.model.model import ModelType

    m_umb = _via_umb(sp)
    assert m_umb.model_type == ModelType.CTMC


# ---------------------------------------------------------------------------
# POMDP
# ---------------------------------------------------------------------------


def test_pomdp_maze_stormpy_vs_umb():
    sp = _build(F.prism_pomdp_maze, choice_labels=True)
    m_sp = stormpy_to_stormvogel(sp)
    m_umb = _via_umb(sp)
    _assert_same_structure(m_sp, m_umb)


def test_pomdp_maze_observation_count():
    sp = _build(F.prism_pomdp_maze, choice_labels=True)
    m_sp = stormpy_to_stormvogel(sp)
    m_umb = _via_umb(sp)
    obs_sp = len(set(m_sp.state_observations.values()))
    obs_umb = len(set(m_umb.state_observations.values()))
    assert obs_sp == obs_umb
