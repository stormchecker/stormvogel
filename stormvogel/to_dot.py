import colorsys
import math
import re
from uuid import uuid4

import pydot
from IPython.display import display

import stormvogel.model
from stormvogel.model import EmptyAction

_NAMED_COLORS = {"black": "#000000", "white": "#ffffff"}


def _invert_lightness(hex_color: str) -> str:
    """Return the lightness-inverted counterpart of *hex_color* for dark mode."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(hue, 1.0 - lum, sat)
    return "#{:02x}{:02x}{:02x}".format(
        round(r2 * 255), round(g2 * 255), round(b2 * 255)
    )


def _dark_mode_parts(svg_bytes: bytes) -> tuple[str, str, str] | None:
    """Collect colors and build dark-mode CSS for an SVG.

    Returns ``(svg_id, style_block, svg_with_id)`` or ``None`` if no colors
    were found.  The ``<style>`` block is not yet inserted into the SVG so
    callers can decide whether to inject it inside or outside the ``<svg>``.
    """
    svg = svg_bytes.decode("utf-8")

    found: set[str] = set()
    for m in re.finditer(r'(?:fill|stroke)="([^"]+)"', svg):
        val = m.group(1).lower()
        if val in ("none", "transparent"):
            continue
        canon = _NAMED_COLORS.get(val) or (
            val if re.fullmatch(r"#[0-9a-f]{6}", val) else None
        )
        if canon:
            found.add(canon)

    if not found:
        return None

    svg_id = f"sv-{uuid4().hex[:8]}"
    svg_with_id = re.sub(r"<svg\b", f'<svg id="{svg_id}"', svg, count=1)

    dark_rules: list[str] = []
    media_rules: list[str] = []
    for color in sorted(found):
        dark = _invert_lightness(color)
        aliases = [color] + [n for n, h in _NAMED_COLORS.items() if h == color]
        for alias in aliases:
            for attr in ("fill", "stroke"):
                scoped = f'#{svg_id} [{attr}="{alias}"]'
                dark_rules.append(f"html.dark {scoped} {{ {attr}: {dark}; }}")
                media_rules.append(
                    f"  :root:not(.light) {scoped} {{ {attr}: {dark}; }}"
                )

    # SVG <text> elements carry no fill attribute; add an explicit rule so that
    # label text inverts alongside the surrounding graphics.
    dark_rules.append(f"html.dark #{svg_id} text {{ fill: #ffffff; }}")
    media_rules.append(f"  :root:not(.light) #{svg_id} text {{ fill: #ffffff; }}")

    style = (
        "<style>\n"
        + "\n".join(dark_rules)
        + "\n@media (prefers-color-scheme: dark) {\n"
        + "\n".join(media_rules)
        + "\n}\n</style>"
    )
    return svg_id, style, svg_with_id


def _dark_mode_svg(svg_bytes: bytes) -> bytes:
    """Return SVG bytes with the dark-mode ``<style>`` injected inside the ``<svg>``.

    Used for ``.svg`` file output, where the style must travel with the file.
    """
    parts = _dark_mode_parts(svg_bytes)
    if parts is None:
        return svg_bytes
    _svg_id, style, svg_with_id = parts
    result = re.sub(r"(<svg\b[^>]*>)", r"\1\n" + style, svg_with_id, count=1)
    return result.encode("utf-8")


def _dark_mode_html(svg_bytes: bytes) -> str:
    """Return an HTML string with the dark-mode ``<style>`` placed *outside* the SVG.

    MyST/Sphinx strips ``<style>`` elements from SVG cell outputs but passes
    HTML outputs through unchanged, so placing the style adjacent to the SVG
    keeps it intact.
    """
    parts = _dark_mode_parts(svg_bytes)
    if parts is None:
        return svg_bytes.decode("utf-8")
    _svg_id, style, svg_with_id = parts
    return style + "\n" + svg_with_id


def _auto_action_positions(
    model: stormvogel.model.Model,
    positions: dict,
    self_loop_radius: float = 0.7,
) -> dict:
    """Compute positions for action nodes whose state source has a known position.

    For regular (non-self-loop) actions the action node is placed at the midpoint
    between the source state and the probability-weighted average of target positions
    (targets without positions are ignored in the average).

    For self-loop actions the action node is placed at ``self_loop_radius`` inches
    from the state, cycling through compass angles (below, lower-left, lower-right,
    …) so that multiple self-loops on the same state do not overlap.
    """
    # Angles used for successive self-loop action nodes on the same state (degrees,
    # measured from positive-x axis in Graphviz/neato coordinate space).
    _self_loop_angles = [270, 225, 315, 180, 0, 90, 135, 45]

    auto: dict = {}
    loop_count: dict = {}  # state → number of self-loop action nodes already placed

    for state, choice in model.transitions.items():
        if state not in positions:
            continue
        src = positions[state]

        for action, branch in choice:
            if action == EmptyAction:
                continue

            targets = [t for _, t in branch]
            is_self_loop = all(t is state for t in targets)

            if is_self_loop:
                idx = loop_count.get(state, 0)
                loop_count[state] = idx + 1
                angle = math.radians(_self_loop_angles[idx % len(_self_loop_angles)])
                auto[(state, action)] = (
                    src[0] + self_loop_radius * math.cos(angle),
                    src[1] + self_loop_radius * math.sin(angle),
                )
            else:
                known = [(prob, positions[t]) for prob, t in branch if t in positions]
                if not known:
                    continue
                total = sum(p for p, _ in known)
                avg_x = sum(p * pos[0] for p, pos in known) / total
                avg_y = sum(p * pos[1] for p, pos in known) / total
                auto[(state, action)] = (
                    (src[0] + avg_x) / 2,
                    (src[1] + avg_y) / 2,
                )

    return auto


def suggest_positions(
    model: stormvogel.model.Model,
    rankdir: str = "LR",
    width: float = 0.8,
) -> dict:
    """Run the graphviz layout engine on *model* and return the computed state positions.

    The returned dict maps each :class:`~stormvogel.model.State` to an ``(x, y)``
    tuple (inches) that can be passed directly as the ``positions`` argument of
    :func:`plot_model_pydot`.  Useful when two models share the same state space
    and you want both rendered with an identical layout.

    :param model: The model to lay out.
    :param rankdir: Layout direction ("LR", "TB", etc.).
    :param width: Diameter of state circles in inches (affects spacing).
    """
    graph = pydot.Dot(graph_type="digraph", rankdir=rankdir)
    node_id = {s: str(i) for i, s in enumerate(model.states)}

    for s in model.states:
        graph.add_node(
            pydot.Node(
                node_id[s],
                shape="circle",
                width=str(width),
                height=str(width),
                fixedsize="true",
            )
        )

    for state, choice in model.transitions.items():
        for action, branch in choice:
            if action != EmptyAction:
                act_id = f"act_{node_id[state]}_{action.label}"
                graph.add_node(
                    pydot.Node(
                        act_id, shape="box", width="0.1", height="0.1", fixedsize="true"
                    )
                )
                graph.add_edge(pydot.Edge(node_id[state], act_id))
                src_id = act_id
            else:
                src_id = node_id[state]
            for _, target in branch:
                graph.add_edge(pydot.Edge(src_id, node_id[target]))

    graph.set_prog("dot")
    plain = graph.create("plain").decode("utf-8")

    id_to_state = {v: k for k, v in node_id.items()}
    positions = {}
    for line in plain.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "node" and parts[1] in id_to_state:
            positions[id_to_state[parts[1]]] = (float(parts[2]), float(parts[3]))
    return positions


def _observation_color_map(
    model: stormvogel.model.Model,
    saturation: float = 0.40,
    lightness: float = 0.82,
) -> dict:
    """Return a mapping from each Observation to a muted pastel hex color."""
    if not model.supports_observations():
        return {}
    obs_list = list(model.observation_aliases)
    n = len(obs_list)
    result = {}
    for i, obs in enumerate(obs_list):
        hue = i / n if n > 1 else 0.0
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        result[obs] = "#{:02x}{:02x}{:02x}".format(
            round(r * 255), round(g * 255), round(b * 255)
        )
    return result


def plot_model_pydot(
    model: stormvogel.model.Model,
    output_file: str | None = None,
    positions: dict | None = None,
    rankdir: str = "LR",
    width: float = 0.8,
    policy: dict | None = None,
    state_colors: dict[str, str] | None = None,
    default_fill: str = "#a6cee3",
    show_state_rewards: bool = True,
    show_transition_rewards: bool = True,
    auto_action_positions: bool = True,
    highlight_state: "stormvogel.model.State | None" = None,
    self_loop_position: "str | dict[stormvogel.model.State, str] | None" = None,
    color_by_observation: bool = False,
) -> None:
    """Render a stormvogel model to SVG/PDF using pydot.

    State nodes are circles; action nodes are small filled squares (unlabeled).
    The edge from a state to an action node carries the action label; edges
    from an action node to successors carry the probability. DTMCs use direct
    state-to-state edges instead. The initial state gets an incoming arrow.

    :param model: The model to render.
    :param output_file: Path ending in .svg or .pdf. None returns an HTML object
        containing the SVG for display in Jupyter / MyST notebooks.
    :param positions: Optional dict mapping :class:`~stormvogel.model.State` to
        ``(x, y)``. States without an entry are auto-placed by neato. Action
        nodes are never required in this dict: when the source state has a
        known position the action node is placed automatically (midpoint toward
        target states for regular actions; offset below the state for
        self-loops). Explicit ``(State, Action)`` entries override the
        automatic placement.
    :param rankdir: Layout direction ("LR", "TB", etc.); ignored when positions
        are given (neato is used instead of dot).
    :param width: Diameter of state circles in inches (all states uniform).
    :param policy: Optional dict[State, Action]. The chosen action node and the
        edge leading to it are highlighted in orange; non-chosen action edges
        are drawn in a lighter gray.
    :param state_colors: Optional mapping from state label to fill color. The
        first matching label (in dict-insertion order) determines the fill.
        States with no matching label use ``default_fill``.
    :param default_fill: Fill color applied to states that have no entry in
        ``state_colors``.
    :param show_state_rewards: When ``True``, append state reward values to
        each state node label. Has no effect if the model has no state rewards.
    :param show_transition_rewards: When ``True``, append transition reward
        values to the edge label between an action node and its successor.
        Has no effect if the model has no transition rewards.
    :param auto_action_positions: When ``True`` (default), action node
        positions are computed automatically from state positions. Set to
        ``False`` to let the layout engine place all action nodes freely.
    :param highlight_state: Optional single state to highlight with a thick
        orange border. The fill color is still determined by ``state_colors``
        / ``default_fill``.
    :param self_loop_position: Compass direction(s) controlling where self-loop
        edges are drawn. A single string (e.g. ``"s"``, ``"ne"``) applies to
        all self-loops; a ``dict[State, str]`` gives per-state control.
        Both ``headport`` and ``tailport`` are set to the chosen direction.
        ``None`` (default) lets Graphviz decide.
    :param color_by_observation: When ``True``, states are filled with
        automatically chosen muted pastel colors, one distinct color per
        observation. ``state_colors`` (label-based) still takes priority.
        Has no effect on models that do not support observations.
    """
    use_positions = bool(positions)
    prog = "neato" if use_positions else "dot"

    # Pre-compute action node positions from state positions.
    auto_act_pos = (
        _auto_action_positions(model, positions)
        if (positions and auto_action_positions)
        else {}
    )

    if use_positions:
        graph = pydot.Dot(graph_type="digraph", overlap="false")
    else:
        graph = pydot.Dot(graph_type="digraph", rankdir=rankdir)

    def _node_pos(key) -> str | None:
        # Explicit entry takes priority; fall back to auto-computed position.
        for lookup in (positions, auto_act_pos):
            if lookup is None:
                continue
            p = lookup.get(key)
            if p is not None:
                return f"{p[0]},{p[1]}!"
        return None

    multi_rw = len(model.rewards) > 1

    def _state_label(st) -> str:
        labels = frozenset(st.labels)
        extra = sorted(labels - {"init"})
        name = st.friendly_name or str(st.state_id)[:8]
        label = f"{name}\\n[{', '.join(extra)}]" if extra else name
        if show_state_rewards:
            for rw in model.rewards:
                val = rw.rewards.get(st)
                if val is not None:
                    prefix = f"{rw.name}=" if multi_rw else "r="
                    label += f"\\n{prefix}{val}"
        return label

    def _transition_reward_label(state, action, target) -> str:
        if not show_transition_rewards:
            return ""
        parts = []
        for rw in model.rewards:
            val = rw.transition_rewards.get((state, action, target))
            if val:
                prefix = f"{rw.name}=" if multi_rw else "r="
                parts.append(f"{prefix}{val}")
        return ", ".join(parts)

    def _self_loop_port(state) -> str | None:
        if self_loop_position is None:
            return None
        if isinstance(self_loop_position, str):
            return self_loop_position
        return self_loop_position.get(state)

    node_id = {s: str(i) for i, s in enumerate(model.states)}
    obs_color_map = _observation_color_map(model) if color_by_observation else {}

    # Initial state arrow
    try:
        init = model.initial_state
        graph.add_node(pydot.Node("__start__", shape="point", width="0", label=""))
        graph.add_edge(pydot.Edge("__start__", node_id[init], len=str(width)))
    except RuntimeError:
        pass

    # State nodes
    for s in model.states:
        fill = default_fill
        if obs_color_map and isinstance(s.observation, stormvogel.model.Observation):
            obs_color = obs_color_map.get(s.observation)
            if obs_color:
                fill = obs_color
        if state_colors:
            for label, color in state_colors.items():
                if label in s.labels:
                    fill = color
                    break
        highlighted = s is highlight_state
        attrs = dict(
            shape="circle",
            style="filled",
            fillcolor=fill,
            color="#e8820c" if highlighted else "black",
            penwidth="3" if highlighted else "1",
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
                unchosen = policy is not None and not chosen
                act_id = f"act_{node_id[state]}_{action.label}"
                act_attrs = dict(
                    shape="box",
                    style="filled",
                    fillcolor="#e8820c"
                    if chosen
                    else ("#aaaaaa" if unchosen else "black"),
                    label="",
                    width="0.1",
                    height="0.1",
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
                        color="#e8820c"
                        if chosen
                        else ("#aaaaaa" if unchosen else "black"),
                        fontcolor="#e8820c"
                        if chosen
                        else ("#aaaaaa" if unchosen else "black"),
                        penwidth="2" if chosen else "1",
                    )
                )
                src_id = act_id
            else:
                src_id = node_id[state]

            for prob, target in branch:
                rw_lbl = (
                    _transition_reward_label(state, action, target)
                    if action != EmptyAction
                    else ""
                )
                edge_lbl = f"{prob}, {rw_lbl}" if rw_lbl else str(prob)
                edge_attrs: dict = dict(
                    label=edge_lbl,
                    fontsize="10",
                    fontcolor="black",
                )
                if target is state:
                    port = _self_loop_port(state)
                    if port is not None:
                        edge_attrs["headport"] = port
                        edge_attrs["tailport"] = port
                graph.add_edge(pydot.Edge(src_id, node_id[target], **edge_attrs))

    graph.set_prog(prog)
    if output_file:
        if output_file.lower().endswith(".svg"):
            with open(output_file, "wb") as _f:
                _f.write(_dark_mode_svg(graph.create_svg()))
        elif output_file.lower().endswith(".pdf"):
            graph.write_pdf(output_file)
        else:
            raise ValueError("output_file must end with .svg or .pdf")
    else:

        class _Display:
            def __init__(self, g):
                self._g = g

            def _repr_html_(self):
                return _dark_mode_html(self._g.create_svg())

            def _repr_png_(self):
                return self._g.create_png()

        display(_Display(graph))
