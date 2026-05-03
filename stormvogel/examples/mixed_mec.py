import stormvogel.bird as bird
from stormvogel.model import ModelType


def create_mixed_mec_mdp():
    """MDP illustrating trivial MECs, non-trivial MECs, and non-MEC cycling.

    States
    ------
    s0 : entry state; routes to each region
    s1 : cycle with stochastic exit — NOT a MEC (notmec1)
    s2 : cycle with stochastic exit — NOT a MEC; labeled "target" (notmec2)
    s3 : non-trivial MEC via "loop"; "escape" exits to s1  (mec1)
    s4 : non-trivial MEC via "loop"; "self" self-loops      (mec2)
    s5 : absorbing self-loop — trivial MEC                  (sink)

    Extra actions on s3/s4 highlight that being in an EC is about the
    existence of a scheduler that stays forever, not that every scheduler does.
    """
    (S0, S1, S2, S3, S4, S5) = range(6)

    def _available_actions(s):
        if s == S3:
            return ["loop", "escape"]
        if s == S4:
            return ["loop", "self"]
        return ["loop"]

    def _delta(s, act):
        match s:
            case 0:  # s0
                return [(1 / 3, S1), (1 / 3, S3), (1 / 3, S5)]
            case 1:  # s1
                return [(0.7, S2), (0.3, S5)]
            case 2:  # s2 (target)
                return [(0.7, S1), (0.3, S5)]
            case 3:  # s3
                if act == "escape":
                    return [(1.0, S1)]
                return [(1.0, S4)]  # loop
            case 4:  # s4
                if act == "self":
                    return [(1.0, S4)]
                return [(1.0, S3)]  # loop
            case 5:  # s5 (sink)
                return [(1.0, S5)]

    def _labels(s):
        if s == S2:
            return ["target"]
        return []

    def _friendly_name(s):
        return f"s{s}"

    return bird.build_bird(
        _delta,
        available_actions=_available_actions,
        init=S0,
        labels=_labels,
        modeltype=ModelType.MDP,
        friendly_names=_friendly_name,
    )
