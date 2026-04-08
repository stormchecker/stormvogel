from typing import TypeVar, Callable, Iterable, Dict, List, Tuple, Optional, Generic

import stormvogel.bird
import stormvogel.model
import pydot
from IPython.display import SVG, display

State = TypeVar("State")
Symbol = frozenset[str] | list[str]
Predicate = Callable[[Symbol], bool]


class SymbolicDFA(Generic[State]):
    def __init__(
        self,
        states: Iterable[State],
        initial_state: State,
        accepting_states: Iterable[State],
    ):
        self.states = set(states)
        self.initial_state = initial_state
        self.accepting_states = set(accepting_states)

        if initial_state not in self.states:
            raise ValueError("Initial state must be in states")
        if not self.accepting_states <= self.states:
            raise ValueError("Accepting states must be subset of states")

        # (predicate, target, label)
        self._transitions: Dict[State, List[Tuple[Predicate, State, str | None]]] = {
            s: [] for s in self.states
        }

    def add_transition(
        self,
        source: State,
        predicate: Predicate,
        target: State,
        label: Optional[str] = None,
    ):
        if source not in self.states or target not in self.states:
            raise ValueError("Transition uses unknown state")

        if label is None:
            label = getattr(predicate, "__name__", "?")

        self._transitions[source].append((predicate, target, label))

    def step(self, state: State, symbol: Symbol) -> State:
        matches = [
            target for pred, target, _ in self._transitions[state] if pred(symbol)
        ]

        if len(matches) == 0:
            raise ValueError(f"No transition from {state} on {symbol}")
        if len(matches) > 1:
            raise ValueError(f"Nondeterminism at {state} on {list(symbol)}: {matches}")

        return matches[0]

    def run(self, word: Iterable[Symbol]) -> State:
        state = self.initial_state
        for sym in word:
            state = self.step(state, sym)
        return state

    def accepts(self, word: Iterable[Symbol]) -> bool:
        return self.run(word) in self.accepting_states

    def __repr__(self) -> str:
        lines = [
            f"States: {self.states}",
            f"Initial: {self.initial_state}",
            f"Accepting: {self.accepting_states}",
            "Transitions:",
        ]
        for s, trans in self._transitions.items():
            for _, t, label in trans:
                lines.append(f"  {s} -[{label}]-> {t}")
        return "\n".join(lines)


class ProductState(stormvogel.bird.State):
    def __init__(self, mdp, dfa, mdp_state, dfa_state):
        super().__init__(_mdp=mdp, _dfa=dfa, _mdp_state=mdp_state, _dfa_state=dfa_state)

    @property
    def mdp_state(self) -> stormvogel.model.State:
        return self._mdp_state

    @property
    def dfa_state(self) -> State:
        return self._dfa_state

    @property
    def dfa(self) -> SymbolicDFA:
        return self._dfa

    @property
    def mdp(self) -> stormvogel.model.Model:
        return self._mdp


def product(mdp: stormvogel.model.Model, dfa: SymbolicDFA):
    _init = ProductState(
        mdp,
        dfa,
        mdp.initial_state,
        dfa.step(dfa.initial_state, list(mdp.initial_state.labels)),
    )

    def _friendly_name(sq: ProductState) -> str:
        assert sq.mdp_state.friendly_name is not None
        return "(" + sq.mdp_state.friendly_name + "," + sq.dfa_state + ")"

    def _delta(sq: ProductState, a: str):
        result = []
        next_distr = sq.mdp_state.get_outgoing_transitions(mdp.action(a))
        if next_distr is None:
            raise RuntimeError("Dont know what to do!")
        for prob, next_state in next_distr:
            next_q = dfa.step(sq.dfa_state, list(next_state.labels))
            result.append((prob, ProductState(mdp, dfa, next_state, next_q)))
        return result

    def _labels(state: ProductState) -> list[str]:
        labels = (
            [label for label in state.mdp_state.labels if label != "init"] + ["init"]
            if state == _init
            else [] + ["accept"]
            if state.dfa_state in dfa.accepting_states
            else []
        )
        return labels

    def _available_actions(state: ProductState) -> list[str]:
        return [
            a.label for a in state.mdp_state.available_actions() if a.label is not None
        ]

    return stormvogel.bird.build_bird(
        delta=_delta,
        init=_init,
        labels=_labels,
        friendly_names=_friendly_name,
        available_actions=_available_actions,
    )


def plot_symbolic_dfa_pydot(dfa, output_file=None, rankdir="LR"):
    """
    Plot a SymbolicDFA using Graphviz positions, but render labels with LaTeX in SVG.
    """
    # Create the directed graph
    graph = pydot.Dot(graph_type="digraph", rankdir=rankdir)
    graph.set_size('"3,3!"')

    for state in dfa.states:
        if state in dfa.accepting_states:
            node_shape = "doublecircle"
            fillcolor = "#b2f2bb"  # light green
        else:
            node_shape = "circle"
            fillcolor = "#a6cee3"  # light blue

        graph.add_node(
            pydot.Node(
                str(state),
                shape=node_shape,
                style="filled",
                fillcolor=fillcolor,
                fontcolor="black",
                fontsize="12",
                label=state,
            )
        )

    # Initial state arrow
    graph.add_node(pydot.Node("start", shape="point"))
    graph.add_edge(pydot.Edge("start", str(dfa.initial_state)))

    # Add transitions
    for src, trans_list in dfa._transitions.items():
        for _, tgt, label in trans_list:
            graph.add_edge(
                pydot.Edge(
                    str(src),
                    str(tgt),
                    label=label,
                    fontcolor="black",
                    color="black",
                    fontsize="10",
                )
            )
    if output_file:
        if output_file.lower().endswith(".svg"):
            graph.write_svg(output_file)
        elif output_file.lower().endswith(".pdf"):
            graph.write_pdf(output_file)
        else:
            raise ValueError("output_file must end with .svg or .pdf")
    else:
        # Display inline as SVG (for Jupyter notebooks)
        svg_data = graph.create_svg()
        display(SVG(svg_data))
