from fractions import Fraction

import pytest

import stormvogel.examples as examples
import stormvogel.model
import stormvogel.teaching.bellman as bellman


@pytest.mark.parametrize(
    "input",
    [
        (examples.create_monty_hall_mdp(), "target"),
        (examples.create_lion_mdp(), "full"),
    ],
)
def test_teaching_bellman(input):
    pytest.importorskip("stormpy")
    bellman.maxreachprob(*input)


def _simple_mdp():
    """Two-state MDP: init --(go)--> full (self-loop)."""
    mdp = stormvogel.model.new_mdp()
    full = mdp.new_state("full")
    act = mdp.new_action("go")
    mdp.set_choices(mdp.initial_state, {act: [(1.0, full)]})
    mdp.add_self_loops()
    return mdp


def test_zero_value():
    mdp = _simple_mdp()
    v = bellman.zero_value(mdp)
    assert all(val == 0 for val in v.values())
    assert set(v.keys()) == set(mdp.sorted_states)


def test_one_value():
    mdp = _simple_mdp()
    v = bellman.one_value(mdp)
    assert all(val == 1 for val in v.values())


def test_value_to_latex():
    mdp = _simple_mdp()
    values: dict[stormvogel.model.State, stormvogel.model.Value] = {
        s: float(i) for i, s in enumerate(mdp.sorted_states)
    }
    latex_lines = bellman.value_to_latex(values)
    assert len(latex_lines) == len(mdp.states)
    for line in latex_lines:
        assert "=" in line


# ---------------------------------------------------------------------------
# _robust_action_value — no stormpy needed
# ---------------------------------------------------------------------------


def _interval_branch(*entries):
    """Build a branch as [(Interval(lo, hi), state), ...] using mock state objects."""

    class _S:
        def __init__(self, name):
            self.name = name

        def __hash__(self):
            return hash(self.name)

        def __eq__(self, other):
            return self.name == other.name

    return [(stormvogel.model.Interval(lo, hi), _S(name)) for lo, hi, name in entries]


def test_robust_action_value_nature_min_pushes_to_low():
    """Nature minimises: mass goes to the lowest-value state."""
    # Two states: A with value 1, B with value 0
    # Interval: A=[0.2, 0.8], B=[0.2, 0.8], lower sum = 0.4 → 0.6 remaining
    branch = _interval_branch((0.2, 0.8, "A"), (0.2, 0.8, "B"))
    s_a, s_b = [s for _, s in branch]
    values = {s_a: 1.0, s_b: 0.0}
    # nature minimises: pushes to B → A gets lower (0.2), B gets upper (0.8)
    result = bellman._robust_action_value(branch, values, nature_maximizes=False)
    assert result == pytest.approx(0.2 * 1.0 + 0.8 * 0.0)


def test_robust_action_value_nature_max_pushes_to_high():
    """Nature maximises: mass goes to the highest-value state."""
    branch = _interval_branch((0.2, 0.8, "A"), (0.2, 0.8, "B"))
    s_a, s_b = [s for _, s in branch]
    values = {s_a: 1.0, s_b: 0.0}
    # nature maximises: pushes to A → A gets upper (0.8), B gets lower (0.2)
    result = bellman._robust_action_value(branch, values, nature_maximizes=True)
    assert result == pytest.approx(0.8 * 1.0 + 0.2 * 0.0)


def test_robust_action_value_exact_fractions():
    """Corner-point algorithm is exact with Fraction values."""
    branch = _interval_branch(
        (Fraction(1, 5), Fraction(4, 5), "A"),
        (Fraction(1, 5), Fraction(4, 5), "B"),
    )
    s_a, s_b = [s for _, s in branch]
    values = {s_a: Fraction(1), s_b: Fraction(0)}
    result = bellman._robust_action_value(branch, values, nature_maximizes=False)
    assert result == Fraction(1, 5)


def test_robust_action_value_tight_intervals():
    """With lo == hi there is no freedom; result is the point-probability expected value."""
    branch = _interval_branch((0.3, 0.3, "A"), (0.7, 0.7, "B"))
    s_a, s_b = [s for _, s in branch]
    values = {s_a: 0.5, s_b: 0.2}
    for nat in (True, False):
        result = bellman._robust_action_value(branch, values, nature_maximizes=nat)
        assert result == pytest.approx(0.3 * 0.5 + 0.7 * 0.2)


# ---------------------------------------------------------------------------
# VI with reward operators — no stormpy needed
# ---------------------------------------------------------------------------


def _reward_chain():
    """Linear chain MDP: init --(go)--> mid --(go)--> done (self-loop).

    State rewards: init=2, mid=1, done=0.
    """
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    init = mdp.new_state(labels=["init"], friendly_name="init")
    mid = mdp.new_state(labels=[], friendly_name="mid")
    done = mdp.new_state(labels=["done"], friendly_name="done")
    go = mdp.action("go")
    init.set_choices({go: [(1.0, mid)]})
    mid.set_choices({go: [(1.0, done)]})
    done.set_choices({go: [(1.0, done)]})
    rw = mdp.new_reward_model("cost")
    rw.set_state_reward(init, 2.0)
    rw.set_state_reward(mid, 1.0)
    rw.set_state_reward(done, 0.0)
    return mdp


def test_reachreward_converges_to_correct_value():
    """Expected reward until done: init should converge to 3 (2 + 1 + 0)."""
    mdp = _reward_chain()
    op = bellman.make_operator_max_reachreward(mdp, "cost", "done")
    vi = bellman.VI(op, bellman.zero_value(mdp))
    for _ in range(5):
        vi.step()
    v = vi.current_values
    init = next(s for s in mdp.states if "init" in s.labels)
    mid = next(s for s in mdp.states if s.friendly_name == "mid")
    assert v[init] == pytest.approx(3.0)
    assert v[mid] == pytest.approx(1.0)


def test_min_reachreward_equals_max_on_dtmc_like_mdp():
    """With a single action per state, min and max reward operators agree."""
    mdp = _reward_chain()
    op_max = bellman.make_operator_max_reachreward(mdp, "cost", "done")
    op_min = bellman.make_operator_min_reachreward(mdp, "cost", "done")
    vi_max = bellman.VI(op_max, bellman.zero_value(mdp))
    vi_min = bellman.VI(op_min, bellman.zero_value(mdp))
    for _ in range(5):
        vi_max.step()
        vi_min.step()
    for s in mdp.states:
        assert vi_max.current_values[s] == pytest.approx(vi_min.current_values[s])


def test_discounted_reward_converges():
    """Discounted reward: init converges to 2 + 0.5*1 + 0.25*0 = 2.5."""
    mdp = _reward_chain()
    op = bellman.make_operator_max_discounted_reward(mdp, "cost", discount=0.5)
    vi = bellman.VI(op, bellman.zero_value(mdp))
    for _ in range(30):
        vi.step()
    init = next(s for s in mdp.states if "init" in s.labels)
    assert vi.current_values[init] == pytest.approx(2.5, abs=1e-6)


def test_discounted_reward_lower_discount_gives_lower_value():
    """Higher discount (closer to 1) gives higher total discounted value."""
    mdp = _reward_chain()
    op_lo = bellman.make_operator_max_discounted_reward(mdp, "cost", discount=0.1)
    op_hi = bellman.make_operator_max_discounted_reward(mdp, "cost", discount=0.9)
    vi_lo = bellman.VI(op_lo, bellman.zero_value(mdp))
    vi_hi = bellman.VI(op_hi, bellman.zero_value(mdp))
    for _ in range(50):
        vi_lo.step()
        vi_hi.step()
    init = next(s for s in mdp.states if "init" in s.labels)
    assert float(vi_lo.current_values[init]) < float(vi_hi.current_values[init])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# IVI — no stormpy needed
# ---------------------------------------------------------------------------


def _choice_mdp_plain():
    """Two-action MDP: init can go to full (prob 1) or stay (prob 1, self-loop)."""
    mdp = stormvogel.model.new_mdp(create_initial_state=False)
    init = mdp.new_state(labels=["init"], friendly_name="init")
    full = mdp.new_state(labels=["full"], friendly_name="full")
    a = mdp.action("go")
    b = mdp.action("stay")
    init.set_choices({a: [(1.0, full)], b: [(1.0, init)]})
    full.set_choices({a: [(1.0, full)]})
    return mdp


def test_ivi_current_values_returns_dict_of_tuples():
    """IVI.current_values maps each state to a (lower, upper) pair."""
    pytest.importorskip("stormpy")
    mdp = _choice_mdp_plain()
    op = bellman.make_operator_maxreachprob(mdp, "full")
    lower_vi = bellman.VI(op, bellman.zero_value(mdp))
    upper_vi = bellman.VI(op, bellman.one_value(mdp))
    ivi = bellman.IVI(lower_vi, upper_vi)
    cv = ivi.current_values
    assert isinstance(cv, dict)
    for s in mdp.sorted_states:
        assert s in cv
        lo, hi = cv[s]
        assert lo <= hi


def test_ivi_step_returns_dict_of_tuples():
    pytest.importorskip("stormpy")
    mdp = _choice_mdp_plain()
    op = bellman.make_operator_maxreachprob(mdp, "full")
    ivi = bellman.IVI(
        bellman.VI(op, bellman.zero_value(mdp)),
        bellman.VI(op, bellman.one_value(mdp)),
    )
    result = ivi.step()
    assert isinstance(result, dict)
    for s in mdp.sorted_states:
        lo, hi = result[s]
        assert lo <= hi


def test_ivi_bounds_converge():
    """After enough steps the lower and upper bounds should agree."""
    pytest.importorskip("stormpy")
    mdp = _choice_mdp_plain()
    op = bellman.make_operator_maxreachprob(mdp, "full")
    ivi = bellman.IVI(
        bellman.VI(op, bellman.zero_value(mdp)),
        bellman.VI(op, bellman.one_value(mdp)),
    )
    for _ in range(30):
        ivi.step()
    for _, (lo, hi) in ivi.current_values.items():
        assert hi - lo == pytest.approx(0.0, abs=1e-6)
