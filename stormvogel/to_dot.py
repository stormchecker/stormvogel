from stormvogel.model import EmptyAction


def state_id(state):
    return state.state_id


def format_attrs(attrs: dict):
    if not attrs:
        return ""
    return " [" + ", ".join(f'{k}="{v}"' for k, v in attrs.items()) + "]"


def model_to_dot(
    model,
    state_properties=None,
    action_properties=None,
    transition_properties=None,
):
    lines = ["digraph G {"]

    # --- States ---
    for state in model:
        props = state_properties(state) if state_properties else {}
        attrs = {"shape": "circle", **props}
        lines.append(f'"{state_id(state)}"{format_attrs(attrs)};')

    # --- Actions + Transitions ---
    for state, choice in model.transitions.items():
        for action, branch in choice:
            if action != EmptyAction:
                action_node = f"{state.state_id}_{action.label}"
                action_props = (
                    action_properties(state, action) if action_properties else {}
                )
                attrs = {"shape": "box", **action_props}

                lines.append(f'"{action_node}"{format_attrs(attrs)};')
                lines.append(f'"{state_id(state)}" -> "{action_node}";')

                src = action_node
            else:
                src = state

            for probability, target in branch:
                trans_props = (
                    transition_properties(state, action, target)
                    if transition_properties
                    else {}
                )

                attrs = {"label": probability, **trans_props}

                lines.append(
                    f'"{str(src)}" -> "{state_id(target)}"{format_attrs(attrs)};'
                )

    lines.append("}")
    return "\n".join(lines)
