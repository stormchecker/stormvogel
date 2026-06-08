"""Smoke test: verifies stormvogel works with only its mandatory dependencies.

Run with: python smoke.py
Exits with code 0 on success, 1 on any failure.
"""

import sys

import stormvogel.examples.cheese_maze as cheese_maze
import stormvogel.examples.monty_hall as monty_hall
import stormvogel.examples.nuclear_fusion_ctmc as nfc
import stormvogel.model as sv
import stormvogel.simulator as sim

failures = []


def check(description, fn):
    try:
        fn()
        print(f"  ok  {description}")
    except Exception as e:
        print(f"FAIL  {description}: {e}")
        failures.append(description)


check("import stormvogel", lambda: __import__("stormvogel"))

# --- model construction ------------------------------------------------------


def _build_dtmc():
    m = sv.new_dtmc()
    init = m.initial_state
    s1 = m.new_state("s1")
    s2 = m.new_state("s2")
    init.set_choices([(0.5, s1), (0.5, s2)])
    s1.set_choices([(1.0, s1)])
    s2.set_choices([(1.0, s2)])
    return m


def _build_mdp():
    m = sv.new_mdp()
    init = m.initial_state
    s1 = m.new_state("s1")
    s2 = m.new_state("s2")
    init.set_choices({m.action("go"): [(1.0, s1)], m.action("stay"): [(1.0, s2)]})
    s1.set_choices([(1.0, s1)])
    s2.set_choices([(1.0, s2)])
    return m


check("build DTMC", _build_dtmc)
check("build MDP", _build_mdp)
check("build Monty Hall MDP", monty_hall.create_monty_hall_mdp)
check("build nuclear fusion CTMC", nfc.create_nuclear_fusion_ctmc)
check("build cheese maze POMDP", cheese_maze.create_cheese_maze)

# --- model properties --------------------------------------------------------


def _dtmc_properties():
    m = _build_dtmc()
    assert m.nr_states == 3
    assert m.nr_choices == 3


def _reward_model():
    m = _build_dtmc()
    init = m.initial_state
    rm = m.new_reward_model("steps")
    rm.set_state_reward(init, 1.0)
    assert rm.get_state_reward(init) == 1.0


check("DTMC state/choice count", _dtmc_properties)
check("reward model", _reward_model)

# --- simulator ---------------------------------------------------------------


def _simulate_dtmc():
    m = _build_dtmc()
    path = sim.simulate_path(m, steps=5, seed=42)
    assert len(path.path) > 0


def _simulate_mdp():
    m = monty_hall.create_monty_hall_mdp()
    path = sim.simulate_path(m, steps=10, seed=0)
    assert len(path.path) > 0


check("simulate DTMC", _simulate_dtmc)
check("simulate MDP", _simulate_mdp)

# --- result ------------------------------------------------------------------

if failures:
    print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
else:
    print("\nAll checks passed.")
