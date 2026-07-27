"""Most-probable-observation-sequence monitoring on PRISM POMDP benchmarks.

Loads a PRISM POMDP, uniformises its actions away to get a hidden Markov model,
and asks :mod:`stormvogel.teaching.belief_search` for the most probable
observation sequence that drives the belief onto a target label.

Viewing the benchmark as an HMM is what makes the answer an honest probability.
On the POMDP the search maximises over actions, so it reports
``max_a P(o_1..o_k | a_1..a_k)`` — a probability *conditioned* on a choice of
actions.  Once the actions are uniformised away there is nothing left to
choose, and the number is the plain marginal ``P(o_1..o_k)``, which is the
quantity a monitor cares about.

The :data:`PREMISE_MODELS` table records the PREMISE benchmark configuration.
The models themselves are not shipped with stormvogel; point *directory* at a
checkout to run them::

    python -m stormvogel.examples.premise_monitoring path/to/premise

Requires stormpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stormvogel.model.model import Model
    from stormvogel.teaching.belief_search import BeliefSearchResult


@dataclass(frozen=True)
class MonitoringModel:
    """One row of a monitoring benchmark configuration.

    :param filename: PRISM file, relative to the benchmark directory.
    :param constants: Constant assignment passed to the PRISM preprocessor.
    :param good_label: Label whose probability the monitor tracks.
    :param step_bound: Step bound of the original bounded-reachability spec,
        kept for comparison: the belief search is unbounded, so a trace longer
        than this reaches confidence later than the spec allows.
    """

    filename: str
    constants: str
    good_label: str
    step_bound: int


#: The PREMISE configuration from ``compare_conditional_methods.yml``.
PREMISE_MODELS: tuple[MonitoringModel, ...] = (
    MonitoringModel("refuel.nm", "N=3,ENERGY=3", "empty", 3),
    MonitoringModel("refuelB.nm", "N=3,ENERGY=3", "empty", 3),
    MonitoringModel("evade-monitoring.nm", "N=4,RADIUS=2", "crash", 4),
    MonitoringModel("hidden-incentive.nm", "N=3", "crash", 4),
    MonitoringModel("airportA-3.nm", "DMAX=3,PMAX=3", "crash", 4),
    MonitoringModel("airportB-7.nm", "DMAX=3,PMAX=3", "crash", 4),
)


def load_monitoring_hmm(path: str, constants: str = "") -> "Model":
    """Load a PRISM POMDP and return it as an HMM with its actions uniformised.

    Uses :func:`~stormvogel.stormpy_utils.mapping.stormpy_pomdp_to_hmm`, which
    tolerates PRISM models whose states have several equally-labelled choices —
    common when the nondeterminism models an unnamed adversary, and rejected by
    the canonicalising POMDP import.

    :param path: Path to the PRISM file.
    :param constants: Constant assignment, e.g. ``"N=3,ENERGY=3"``.
    :returns: The model as an HMM, with one observation per observation class.
    :raises ImportError: If stormpy is not installed.
    """
    try:
        import stormpy
    except ImportError as e:  # pragma: no cover - exercised only without stormpy
        raise ImportError(
            "load_monitoring_hmm requires stormpy (pip install stormpy)."
        ) from e

    from stormvogel.stormpy_utils.mapping import stormpy_pomdp_to_hmm

    program = stormpy.parse_prism_program(path)
    if constants:
        program = stormpy.preprocess_symbolic_input(program, [], constants)[
            0
        ].as_prism_program()
    options = stormpy.BuilderOptions()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_observation_valuations()
    return stormpy_pomdp_to_hmm(
        stormpy.build_sparse_model_with_options(program, options)
    )


def monitoring_trace(
    hmm: "Model",
    good_label: str,
    threshold: "Fraction | float" = Fraction(9, 10),
    max_expansions: int = 5000,
) -> "BeliefSearchResult":
    """Find the most probable observation sequence that convinces the monitor.

    Searches from the initial state's point belief for the most probable
    observation sequence after which at least *threshold* of the belief mass
    sits on *good_label*.

    Pick *threshold* strictly below 1 unless the model can actually reach
    certainty.  Where a noisy sensor never fully resolves the state, no belief
    ever reaches probability 1, the belief space is infinite, and the search
    runs until *max_expansions* instead of terminating.

    :param hmm: The model, e.g. from :func:`load_monitoring_hmm`.
    :param good_label: Label whose probability the monitor tracks.
    :param threshold: Confidence the monitor must reach.
    :param max_expansions: Budget, since the belief space may be infinite.
    :returns: The :class:`~stormvogel.teaching.belief_search.BeliefSearchResult`.
    """
    from stormvogel.teaching.belief import Belief
    from stormvogel.teaching.belief_search import dijkstra_belief_search

    return dijkstra_belief_search(
        hmm,
        Belief({hmm.initial_state: Fraction(1)}),
        good_label,
        threshold,
        max_expansions=max_expansions,
    )


def monitoring_report(
    hmm: "Model",
    good_label: str,
    threshold: "Fraction | float" = Fraction(9, 10),
    step_bound: int | None = None,
    max_expansions: int = 5000,
) -> str:
    """Return a human-readable report of :func:`monitoring_trace`.

    :param hmm: The model, e.g. from :func:`load_monitoring_hmm`.
    :param good_label: Label whose probability the monitor tracks.
    :param threshold: Confidence the monitor must reach.
    :param step_bound: Optional step bound to compare the trace length against.
    :param max_expansions: Budget passed to the search.
    :returns: A multi-line report.
    """
    result = monitoring_trace(hmm, good_label, threshold, max_expansions)

    effort = (
        f"  explored {result.explored} of {result.discovered} beliefs in "
        f"{result.elapsed * 1000:.0f} ms "
        f"({result.beliefs_per_second:,.0f} beliefs/s) "
        f"+ {result.setup_elapsed * 1000:.0f} ms index"
    )

    if not result.found:
        reason = "hit the expansion limit" if result.truncated else "no such belief"
        return f"P({good_label}) >= {threshold}: not reachable ({reason})\n{effort}"

    beyond = ""
    if step_bound is not None and len(result.steps) > step_bound:
        beyond = f"  [beyond the F<={step_bound} bound]"
    lines = [
        f"P({good_label}) >= {threshold}: "
        f"p = {float(result.probability):.5g} over {len(result.steps)} steps{beyond}",
        f"  reaches P({good_label}) = {float(result.final_target_probability):.4f}",
        effort,
    ]
    for i, step in enumerate(result.steps, start=1):
        alias = step.observation.alias if step.observation is not None else "-"
        lines.append(f"  {i}. P(o) = {float(step.probability):.4f}   {alias}")
    return "\n".join(lines)


def main(directory: str, threshold: "Fraction | float" = Fraction(9, 10)) -> None:
    """Run :func:`monitoring_report` over :data:`PREMISE_MODELS`.

    :param directory: Directory holding the PREMISE ``.nm`` files.
    :param threshold: Confidence the monitor must reach.
    """
    for model in PREMISE_MODELS:
        hmm = load_monitoring_hmm(f"{directory}/{model.filename}", model.constants)
        print("=" * 78)
        print(
            f"{model.filename}  [{model.constants}]  "
            f"{hmm.nr_states} states, "
            f"{len(hmm.get_states_with_label(model.good_label))} "
            f"'{model.good_label}' states"
        )
        print(
            monitoring_report(
                hmm, model.good_label, threshold, step_bound=model.step_bound
            )
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: python -m stormvogel.examples.premise_monitoring <premise-dir>"
        )
    main(sys.argv[1])
