"""Generate TikZ/LaTeX source code for stormvogel models."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stormvogel.model

from stormvogel.model import EmptyAction

_LETTER_NUM_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------


def _prob_to_latex(p) -> str:
    """Convert a probability to a LaTeX math string (fraction or decimal).

    Handles SymPy symbolic expressions (parametric models) via SymPy's LaTeX
    printer when the ``sympy`` package is available.
    """
    try:
        import sympy  # optional; only needed for parametric models

        if isinstance(p, sympy.Basic):
            return sympy.latex(p)
    except ImportError:
        pass
    try:
        f = Fraction(p).limit_denominator(100)  # type: ignore[arg-type]
        if abs(float(f) - float(p)) > 1e-9:  # type: ignore[arg-type]
            return str(round(float(p), 6))  # type: ignore[arg-type]
        if f.denominator == 1:
            return str(f.numerator)
        return rf"\frac{{{f.numerator}}}{{{f.denominator}}}"
    except (TypeError, ValueError):
        return str(p)


def _name_to_math_label(name: str) -> str:
    """Convert a plain string name to a LaTeX math label.

    ``"s0"`` → ``$s_0$``, ``"state12"`` → ``$\\text{state}_{12}$``,
    ``"sink"`` → ``$\\text{sink}$``, ``"q"`` → ``$q$``.
    """
    m = _LETTER_NUM_RE.match(name)
    if m:
        prefix, num = m.group(1), m.group(2)
        if len(prefix) == 1:
            return f"${prefix}_{{{num}}}$"
        return rf"$\text{{{prefix}}}_{{{num}}}$"
    if len(name) == 1 and name.isalpha():
        return f"${name}$"
    return rf"$\text{{{name}}}$"


def _safe_name(s: str) -> str:
    """Strip characters that are invalid in TikZ node names."""
    return re.sub(r"[^a-zA-Z0-9]", "", s)


def _initial_where(
    init_pos: tuple,
    all_pos: "list[tuple]",
    allowed: "tuple[str, ...]" = ("above", "below", "left"),
) -> str:
    """Return the least-crowded TikZ initial-arrow placement direction.

    Picks from *allowed* (default: above/below/left, i.e. north/south/west)
    by choosing the direction most opposite to the average displacement from
    the initial state to all other states.
    """
    others = [p for p in all_pos if p is not init_pos]
    if not others:
        return allowed[0]
    dx = sum(p[0] - init_pos[0] for p in others) / len(others)
    dy = sum(p[1] - init_pos[1] for p in others) / len(others)
    # Direction vectors for each candidate (TikZ coords: x right, y up)
    vecs = {"above": (0, 1), "below": (0, -1), "left": (-1, 0), "right": (1, 0)}
    return max(allowed, key=lambda d: vecs[d][0] * (-dx) + vecs[d][1] * (-dy))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def model_to_tikz(
    model: "stormvogel.model.Model",
    output_file: str | None = None,
    positions: dict | None = None,
    coord_scale: float = 2.5,
    rankdir: str = "LR",
    self_loop_bend: int = 30,
    target_labels: "list[str] | None" = None,
    center: bool = True,
    layout_size: "tuple[float, float] | None" = None,
    ranksep: "float | None" = None,
    nodesep: "float | None" = None,
    badge_labels: "list[str] | None" = None,
) -> str:
    """Generate a TikZ ``tikzpicture`` for *model* and return it as a string.

    Required in the LaTeX preamble::

        \\usepackage{tikz}
        \\usetikzlibrary{automata,arrows.meta,calc}
        \\tikzset{state/.append style={minimum size=1.2cm, circle, draw}}

    The generated code uses absolute ``at (x, y)`` coordinates derived from
    the Graphviz layout, so no relative-positioning library is needed.  Label
    positions default to ``midway, above``; adjust per-edge for
    publication-quality output.

    :param model: The model to render.
    :param output_file: If given, write the TikZ snippet to this path.
    :param positions: Optional ``dict`` mapping :class:`~stormvogel.model.State`
        to ``(x, y)`` in inches (same format as ``suggest_positions()``).
        Auto-computed via Graphviz when omitted.
    :param coord_scale: Multiply Graphviz inch coordinates by this factor to
        get centimetres in TikZ (default 2.5).
    :param rankdir: Graphviz layout direction used when *positions* are
        omitted (default ``"LR"``).
    :param self_loop_bend: Bend angle in degrees applied to edges from a
        self-loop action node back to its source state (default 30).
    :param target_labels: List of state label strings (as used in
        :meth:`~stormvogel.model.State.has_label`) that mark accepting/target
        states.  Any state carrying one of these labels is rendered with the
        TikZ ``accepting`` style (double circle).
    :param center: If ``True`` (default), wrap the ``tikzpicture`` in a
        ``\\begin{center}...\\end{center}`` block.  Set to ``False`` when
        embedding the figure inside a minipage or other environment.
    :param badge_labels: Model label strings that should be rendered as small
        annotation badges outside the state circle.  For each state carrying
        one of these labels a tiny labelled node is placed at successive
        compass positions (``north east``, ``north west``, …).
    :param layout_size: Optional ``(width_cm, height_cm)`` pair.  Passed to
        Graphviz as a bounding-box constraint (converted from cm to inches via
        *coord_scale*) so the layout fits within the given area.  Ignored when
        *positions* are provided explicitly.
    """
    from stormvogel.to_dot import _auto_action_positions, suggest_positions

    if positions is None:
        dot_size = (
            f"{layout_size[0] / coord_scale:.3f},{layout_size[1] / coord_scale:.3f}"
            if layout_size is not None
            else None
        )
        positions = suggest_positions(
            model,
            rankdir=rankdir,
            size=dot_size,
            ranksep=ranksep,
            nodesep=nodesep,
        )

    act_positions = _auto_action_positions(model, positions)

    # Build state → index map; used as fallback node name/label when
    # friendly_name is not set (state_id is a UUID, not human-readable).
    state_index = {s: i for i, s in enumerate(model.states)}

    def _state_node_name(s: "stormvogel.model.State") -> str:
        if s.friendly_name is not None:
            name = _safe_name(s.friendly_name)
            if name and not name[0].isdigit():
                return name
        return f"s{state_index[s]}"

    def _state_math_label(s: "stormvogel.model.State") -> str:
        if s.friendly_name is not None:
            return _name_to_math_label(s.friendly_name)
        return f"$s_{{{state_index[s]}}}$"

    def _action_node_name(
        state: "stormvogel.model.State",
        action: "stormvogel.model.Action",
    ) -> str:
        sname = _state_node_name(state)
        aname = _safe_name(action.label) if action.label else "act"
        return f"{sname}{aname}"

    # Collect all (state, action, branch) triples once.
    transitions = [
        (state, action, branch)
        for state, choice in model.transitions.items()
        for action, branch in choice
    ]

    # Self-loop action: all targets of this action node are the source state.
    self_loop_action_pairs: set[tuple] = {
        (state, action)
        for state, action, branch in transitions
        if action != EmptyAction and all(t is state for _, t in branch)
    }

    lines: list[str] = []
    ind = "    "

    # --- Preamble comment -------------------------------------------------
    preamble = [
        "% Requires in LaTeX preamble:",
        "%   \\usepackage{tikz}",
        "%   \\usetikzlibrary{automata,arrows.meta,calc}",
        "%   \\tikzset{state/.append style={minimum size=1.2cm, circle, draw}}",
        "",
    ]
    if center:
        preamble += ["\\begin{center}"]
    preamble += ["\\begin{tikzpicture}", ""]
    lines += preamble

    def _is_target(s: "stormvogel.model.State") -> bool:
        return target_labels is not None and any(
            s.has_label(lbl) for lbl in target_labels
        )

    # Resolve initial state once (may raise RuntimeError if absent).
    try:
        init_state = model.initial_state
    except RuntimeError:
        init_state = None

    # Pre-compute initial-arrow direction from the layout.
    if init_state is not None and positions.get(init_state) is not None:
        all_pos = [p for s, p in positions.items() if p is not None]
        _init_where = _initial_where(positions[init_state], all_pos)
    else:
        _init_where = "above"

    # --- State nodes ------------------------------------------------------
    lines.append(f"{ind}% States")
    for s in model.states:
        pos = positions.get(s)
        if pos is None:
            continue
        x = pos[0] * coord_scale
        y = pos[1] * coord_scale
        nname = _state_node_name(s)
        label = _state_math_label(s)
        style_parts = ["state"]
        if _is_target(s):
            style_parts.append("accepting")
        if s is init_state:
            style_parts += ["initial", "initial text=", f"initial where={_init_where}"]
        style = ", ".join(style_parts)
        lines.append(
            f"{ind}\\node[{style}] ({nname}) at ({x:.3f}cm, {y:.3f}cm) {{{label}}};"
        )
    lines.append("")

    # --- Badge annotations ------------------------------------------------
    _badge_anchors = ["north east", "north west", "south east", "south west"]
    if badge_labels:
        lines.append(f"{ind}% Label badges")
        for s in model.states:
            if positions.get(s) is None:
                continue
            nname = _state_node_name(s)
            hits = [lbl for lbl in badge_labels if s.has_label(lbl)]
            for i, lbl in enumerate(hits):
                anchor = _badge_anchors[i % len(_badge_anchors)]
                math_lbl = f"${lbl}$" if len(lbl) == 1 else rf"$\mathit{{{lbl}}}$"
                lines.append(
                    f"{ind}\\node[draw, inner sep=2pt, font=\\scriptsize, fill=white]"
                    f" at ({nname}.{anchor}) {{{math_lbl}}};"
                )
        lines.append("")

    # --- Action nodes (MDP only) ------------------------------------------
    has_actions = any(a != EmptyAction for _, a, _ in transitions)
    if has_actions:
        lines.append(f"{ind}% Action nodes")
        for state, action, _ in transitions:
            if action == EmptyAction:
                continue
            apos = act_positions.get((state, action))
            spos = positions.get(state)
            if apos is None or spos is None:
                continue
            dx = (apos[0] - spos[0]) * coord_scale
            dy = (apos[1] - spos[1]) * coord_scale
            sname = _state_node_name(state)
            aname = _action_node_name(state, action)
            lines.append(
                f"{ind}\\node[circle, inner sep=2pt, fill=black]"
                f" ({aname}) at ($({sname})+({dx:.3f}cm,{dy:.3f}cm)$) {{}};"
            )
        lines.append("")

    # --- Bend detection ---------------------------------------------------
    # Collect all (from_name, to_name) pairs to detect reverse/parallel edges.
    all_pairs: list[tuple[str, str]] = []
    for state, action, branch in transitions:
        sname = _state_node_name(state)
        if action != EmptyAction:
            if (state, action) not in act_positions:
                continue
            aname = _action_node_name(state, action)
            all_pairs.append((sname, aname))
            for _, target in branch:
                tname = _state_node_name(target)
                if tname != sname:
                    all_pairs.append((aname, tname))
        else:
            for _, target in branch:
                tname = _state_node_name(target)
                if tname != sname:
                    all_pairs.append((sname, tname))

    pair_count = Counter(all_pairs)

    def _bend_opt(from_: str, to_: str, occ: int) -> str:
        reverse_exists = pair_count.get((to_, from_), 0) > 0
        duplicate = pair_count.get((from_, to_), 1) > 1
        if reverse_exists or duplicate:
            angle = 20
            return f"bend left={angle}" if occ == 0 else f"bend right={angle}"
        return ""

    occ_tracker: dict[tuple[str, str], int] = defaultdict(int)

    # Position lookup by TikZ node name — used for direction-aware label placement.
    _name_pos: dict[str, tuple] = {}
    for _s, _p in positions.items():
        _name_pos[_state_node_name(_s)] = _p
    for (_st, _act), _ap in act_positions.items():
        _name_pos[_action_node_name(_st, _act)] = _ap

    def _label_anchor(from_: str, to_: str) -> str:
        """Pick midway label anchor based on straight-line edge direction."""
        fp = _name_pos.get(from_)
        tp = _name_pos.get(to_)
        if fp is None or tp is None:
            return "above"
        angle = math.degrees(math.atan2(tp[1] - fp[1], tp[0] - fp[0]))
        if -45 < angle <= 45:  # rightward
            return "above"
        elif 45 < angle <= 135:  # upward
            return "right"
        elif -135 < angle <= -45:  # downward
            return "left"
        else:  # leftward / back edge
            return "below"

    _anchor_flip = {
        "above": "below",
        "below": "above",
        "left": "right",
        "right": "left",
    }

    def _draw(
        opts: list[str],
        from_: str,
        to_: str,
        label: str = "",
        flip_anchor: bool = False,
    ) -> str:
        opts_str = ", ".join(opts)
        if label:
            anchor = _label_anchor(from_, to_)
            if flip_anchor:
                anchor = _anchor_flip[anchor]
            return f"{ind}\\draw ({from_}) edge[{opts_str}] node[midway, {anchor}] {{{label}}} ({to_});"
        return f"{ind}\\draw ({from_}) edge[{opts_str}] ({to_});"

    # --- State-to-action edges (MDP) -------------------------------------
    if has_actions:
        lines.append(f"{ind}% State-to-action edges")
        for state, action, _ in transitions:
            if action == EmptyAction:
                continue
            if (state, action) not in act_positions:
                continue
            sname = _state_node_name(state)
            aname = _action_node_name(state, action)
            pair = (sname, aname)
            occ = occ_tracker[pair]
            occ_tracker[pair] += 1
            if (state, action) in self_loop_action_pairs:
                bend = f"bend right={self_loop_bend}"
            else:
                bend = _bend_opt(sname, aname, occ)
            opts = ["-"] + ([bend] if bend else [])
            lbl = f"${action.label}$" if action.label else ""
            lines.append(_draw(opts, sname, aname, lbl))
        lines.append("")

    # --- Transition edges -------------------------------------------------
    lines.append(f"{ind}% Transition edges")
    for state, action, branch in transitions:
        sname = _state_node_name(state)
        is_self_loop_action = action != EmptyAction and all(
            t is state for _, t in branch
        )

        if action != EmptyAction:
            if (state, action) not in act_positions:
                continue
            src_name = _action_node_name(state, action)
        else:
            src_name = sname

        for prob, target in branch:
            tname = _state_node_name(target)
            prob_lbl = f"${_prob_to_latex(prob)}$"

            # DTMC self-loop: use TikZ loop style
            if action == EmptyAction and target is state:
                lines.append(
                    f"{ind}\\draw ({sname}) edge[->, loop above]"
                    f" node[above] {{{prob_lbl}}} ({sname});"
                )
                continue

            pair = (src_name, tname)
            occ = occ_tracker[pair]
            occ_tracker[pair] += 1

            if is_self_loop_action and target is state:
                bend = f"bend right={self_loop_bend}"
            else:
                bend = _bend_opt(src_name, tname, occ)

            opts = ["->"] + ([bend] if bend else [])
            # Return edge of a self-loop action: mirror the forward-edge anchor
            # so both labels appear on the same side of the loop.
            flip = is_self_loop_action and target is state
            lines.append(_draw(opts, src_name, tname, prob_lbl, flip_anchor=flip))

    lines += ["", "\\end{tikzpicture}"]
    if center:
        lines += ["\\end{center}"]

    result = "\n".join(lines)

    if output_file:
        with open(output_file, "w") as fh:
            fh.write(result)

    return result
