import stormvogel.bird as bird
from stormvogel.model import ModelType


def create_parker_mdp():
    """Return the Parker MDP, a small MDP used in the lecture notes by David Parker.

    States s0–s3; s0 and s3 have two actions (a, b), the rest only action a.
    s2 is an absorbing target state.
    """

    def _available_actions(s):
        if s in [0, 3]:
            return ["a", "b"]
        return ["a"]

    def _delta(s, act):
        match s:
            case 0:
                match act:
                    case "a":
                        return [(0.25, 0), (0.5, 2), (0.25, 3)]
                    case "b":
                        return [(1, 1)]
            case 1:
                return [(0.1, 0), (0.5, 1), (0.4, 2)]
            case 2:
                return [(1, s)]
            case 3:
                match act:
                    case "a":
                        return [(1, 2)]
                    case "b":
                        return [(1, 3)]

    def _labels(s):
        return "s" + str(s)

    def _friendly_name(s):
        return "s" + str(s)

    return bird.build_bird(
        _delta,
        available_actions=_available_actions,
        init=0,
        labels=_labels,
        modeltype=ModelType.MDP,
        friendly_names=_friendly_name,
    )
