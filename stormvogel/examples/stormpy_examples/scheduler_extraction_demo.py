"""Demonstrate that Storm cannot extract a scheduler for R=? [C] on MDPs with
end components, but CAN do so for R=? [F "done"] on the same model.

Run with:  python stormvogel/examples/stormpy_examples/scheduler_extraction_demo.py
"""

import stormpy
import stormvogel.model
import stormvogel.stormpy_utils.mapping as mapping
from stormvogel.teaching.multiobjective import weighted_multi_target_reachability


def _make_mdp():
    mdp = stormvogel.model.new_mdp()
    s0 = mdp.initial_state
    s0.set_friendly_name("center")
    left = mdp.new_action("left")
    right = mdp.new_action("right")
    sL = mdp.new_state(friendly_name="sL")
    sR = mdp.new_state(friendly_name="sR")
    t1 = mdp.new_state("t1", friendly_name="t1")
    t2 = mdp.new_state("t2", friendly_name="t2")
    sink = mdp.new_state(friendly_name="-")
    sinkc = mdp.new_state(friendly_name="-c")
    s0.set_choices({left: [(0.7, sL), (0.3, s0)], right: [(0.8, sR), (0.2, s0)]})
    sL.set_choices({left: [(0.9, t1), (0.1, sink)], right: [(0.65, s0), (0.35, sink)]})
    sR.set_choices({right: [(0.8, t2), (0.2, sinkc)], left: [(0.3, s0), (0.7, sR)]})
    mdp.add_self_loops()
    return mdp


def main():
    mdp = _make_mdp()
    transformed = weighted_multi_target_reachability(mdp, ["t1", "t2"], [1.0, 1.0])
    transformed.add_self_loops()

    # Label absorbing states (only EmptyAction available) as "done"
    for s in transformed.states:
        if all(a.label is None for a in s.available_actions()):
            s.add_label("__done__")

    stormpy_model = mapping.stormvogel_to_stormpy(transformed)
    print(f"Stormpy states: {stormpy_model.nr_states}")

    for formula_str in [
        'R{"weighted_reach"}max=? [C]',
        'R{"weighted_reach"}max=? [F "__done__"]',
    ]:
        prop = stormpy.parse_properties(formula_str)[0]
        result = stormpy.model_checking(stormpy_model, prop, extract_scheduler=True)
        print(f"\nFormula: {formula_str}")
        print(f"  result[0] (init) = {result.at(0):.6f}")
        print(f"  has_scheduler    = {result.has_scheduler}")


if __name__ == "__main__":
    main()
