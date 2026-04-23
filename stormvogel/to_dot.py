import pydot
from IPython.display import SVG, display

import stormvogel.model
from stormvogel.model import EmptyAction


def plot_model_pydot(
    model: stormvogel.model.Model,
    output_file: str | None = None,
    positions: dict | None = None,
    rankdir: str = "LR",
    width: float = 0.8,
    policy: dict | None = None,
) -> None:
    """Render a stormvogel model to SVG/PDF using pydot.

    State nodes are circles; action nodes are small filled squares (unlabeled).
    The edge from a state to an action node carries the action label; edges
    from an action node to successors carry the probability. DTMCs use direct
    state-to-state edges instead. The initial state gets an incoming arrow.

    :param model: The model to render.
    :param output_file: Path ending in .svg or .pdf. None returns an HTML object
        containing the SVG for display in Jupyter / MyST notebooks.
    :param positions: Optional dict mapping State or (State, Action) to (x, y).
        Nodes with a position entry are pinned; others are auto-placed by neato.
    :param rankdir: Layout direction ("LR", "TB", etc.); ignored when positions
        are given (neato is used instead of dot).
    :param width: Diameter of state circles in inches (all states uniform).
    :param policy: Optional dict[State, Action]. The chosen action node and the
        edge leading to it are highlighted in orange; non-chosen action edges
        are drawn in a lighter gray.
    """
    use_positions = bool(positions)
    prog = "neato" if use_positions else "dot"

    if use_positions:
        graph = pydot.Dot(graph_type="digraph")
    else:
        graph = pydot.Dot(graph_type="digraph", rankdir=rankdir)

    def _node_pos(key) -> str | None:
        if not positions:
            return None
        p = positions.get(key)
        if p is None:
            return None
        return f"{p[0]},{p[1]}!"

    def _state_label(st) -> str:
        labels = frozenset(st.labels)
        extra = sorted(labels - {"init"})
        name = st.friendly_name or str(st.state_id)[:8]
        return f"{name}\\n[{', '.join(extra)}]" if extra else name

    node_id = {s: str(i) for i, s in enumerate(model.states)}

    # Initial state arrow
    try:
        init = model.initial_state
        graph.add_node(pydot.Node("__start__", shape="point", width="0", label=""))
        graph.add_edge(pydot.Edge("__start__", node_id[init]))
    except RuntimeError:
        pass

    # State nodes
    for s in model.states:
        attrs = dict(
            shape="circle",
            style="filled",
            fillcolor="#a6cee3",
            fontcolor="black",
            fontsize="12",
            label=_state_label(s),
            width=str(width),
            height=str(width),
            fixedsize="true",
        )
        pos = _node_pos(s)
        if pos:
            attrs["pos"] = pos
        graph.add_node(pydot.Node(node_id[s], **attrs))

    # Transitions (with or without action nodes)
    for state, choice in model.transitions.items():
        for action, branch in choice:
            if action != EmptyAction:
                chosen = policy is not None and policy.get(state) == action
                act_id = f"act_{node_id[state]}_{action.label}"
                act_attrs = dict(
                    shape="box",
                    style="filled",
                    fillcolor="#e8820c" if chosen else "#aaaaaa",
                    label="",
                    width="0.2",
                    height="0.2",
                    fixedsize="true",
                )
                pos = _node_pos((state, action))
                if pos:
                    act_attrs["pos"] = pos
                graph.add_node(pydot.Node(act_id, **act_attrs))
                graph.add_edge(
                    pydot.Edge(
                        node_id[state],
                        act_id,
                        label=action.label or "",
                        fontsize="10",
                        color="#e8820c" if chosen else "#aaaaaa",
                        fontcolor="#e8820c" if chosen else "#aaaaaa",
                        penwidth="2" if chosen else "1",
                    )
                )
                src_id = act_id
            else:
                src_id = node_id[state]

            for prob, target in branch:
                graph.add_edge(
                    pydot.Edge(src_id, node_id[target], label=str(prob), fontsize="10")
                )

    graph.set_prog(prog)
    if output_file:
        if output_file.lower().endswith(".svg"):
            graph.write_svg(output_file)
        elif output_file.lower().endswith(".pdf"):
            graph.write_pdf(output_file)
        else:
            raise ValueError("output_file must end with .svg or .pdf")
    else:
        svg_data = graph.create_svg()
        display(SVG(svg_data))
