from stormvogel import bird
from stormvogel.model import ModelType


def create_debugging_mdp():
    def available_actions(s: str):
        if s == ("init", "A"):
            return [["mec"], ["probs"]]
        return [[]]

    def delta(s: str, a: bird.Action):
        if "mec" in a:
            return [(1, ("X", "mec"))]
        elif s == ("X", "mec"):
            return [(1, ("Y", "mec"))]
        elif s == ("Y", "mec"):
            return [(1, ("X", "mec"))]
        elif "probs" in a:
            return [(1, ("A", "E"))]
        elif s == ("A", "E"):
            return [(0.5, ("C", "D")), (0.5, ("B", "D"))]
        return [(1, s)]

    def labels(s):
        if isinstance(s, tuple):
            return list(s)
        return [s]

    return bird.build_bird(
        delta=delta,
        init=("init", "A"),
        available_actions=available_actions,
        labels=labels,
        modeltype=ModelType.MDP,
    )
