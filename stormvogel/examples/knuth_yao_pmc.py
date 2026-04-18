import sympy as sp

from stormvogel import model, bird


def create_knuth_yao_pmc():
    """Build the Knuth--Yao dice as a parametric DTMC.

    The parameter ``x`` is the success probability of the underlying coin.
    Probabilities are expressed as ordinary sympy expressions, so ``1 - x`` is
    literally ``1 - x`` — no special constructors needed.
    """
    x = sp.Symbol("x")

    initial_state = bird.State(s=0)

    def delta(s: bird.State):
        match s.s:
            case 0:
                return [(x, bird.State(s=1)), (1 - x, bird.State(s=2))]
            case 1:
                return [(x, bird.State(s=3)), (1 - x, bird.State(s=4))]
            case 2:
                return [(x, bird.State(s=5)), (1 - x, bird.State(s=6))]
            case 3:
                return [(x, bird.State(s=1)), (1 - x, bird.State(s=7, d=1))]
            case 4:
                return [
                    (x, bird.State(s=7, d=2)),
                    (1 - x, bird.State(s=7, d=3)),
                ]
            case 5:
                return [
                    (x, bird.State(s=7, d=4)),
                    (1 - x, bird.State(s=7, d=5)),
                ]
            case 6:
                return [(x, bird.State(s=2)), (1 - x, bird.State(s=7, d=6))]
            case 7:
                return [(1, s)]

    def labels(s: bird.State) -> str | None:
        if s.s == 7:
            return f"rolled{str(s.d)}"

    return bird.build_bird(
        delta=delta,
        labels=labels,
        init=initial_state,
        modeltype=model.ModelType.DTMC,
    )


if __name__ == "__main__":
    print(create_knuth_yao_pmc())
