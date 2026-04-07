import stormvogel
import stormvogel.to_dot as to_dot
from stormvogel.examples.minitown import create_minitown_mdp

def test_todot_minitown():
    mdp = create_minitown_mdp()
    dot = to_dot.model_to_dot(
        mdp,
        state_properties=lambda s: {
            "label": s.friendly_name,
            "xlabel": ",".join(s.labels),
            "style": "filled",
            "fillcolor": "lightblue",
        },
        action_properties=lambda s, a: {
            "label": str(a.label),
            "fillcolor": "lightgray",
            "style": "filled",
        },
    )

    with open("mdp.dot", "w") as f:
        f.write(dot)
