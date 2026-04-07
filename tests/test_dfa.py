import stormvogel
import stormvogel.dfa as dfa
from stormvogel.examples.minitown import create_minitown_mdp

def test_aut1():
    aut = dfa.SymbolicDFA(
        states={"s0", "s1"},
        initial_state="s0",
        accepting_states={"s1"},
    )

    aut.add_transition("s0", lambda s: "p" in s, "s1", label="p")
    aut.add_transition("s0", lambda s: "p" not in s, "s0", label="¬p")
    aut.add_transition("s1", lambda s: True, "s1", label="true")

    dfa.visualize_dfa(aut)

def test_libsup():
    aut = dfa.SymbolicDFA(
        states={"q0", "q1", "q2", "q3"},
        initial_state="q0",
        accepting_states={"q3"},
    )

    aut.add_transition("q0", lambda s: "L" in s and "S" not in s, "q1", label="L & -S")
    aut.add_transition("q0", lambda s: "S" in s and "L" not in s, "q2", label="S & -L")
    aut.add_transition("q0", lambda s: "S" in s and "L" in s, "q3", label="S & L")
    aut.add_transition("q0", lambda s: "S" not in s and "L" not in s, "q0", label="-S & -L")
    aut.add_transition("q1", lambda s: "L" in s, "q3", label="L")
    aut.add_transition("q1", lambda s: "L" not in s, "q1", label="-L")
    aut.add_transition("q2", lambda s: "S" in s, "q3", label="S")
    aut.add_transition("q2", lambda s: "S" not in s, "q2", label="-S")
    aut.add_transition("q3", lambda s: True, "q3", label="true")

    dfa.plot_symbolic_dfa_pydot(aut, output_file="dfa2.svg")
    assert aut.step("q0", []) == "q0"
    assert aut.step("q0", ["L"]) == "q1"

    mdp = create_minitown_mdp()
    dfa.product(mdp, aut)
    stormvogel.show(dfa.product(mdp, aut))

