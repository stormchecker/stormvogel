from stormvogel import parametric
from stormvogel import model, bird


def create_knuth_yao_pmc():
    # we first make polynomials 'x' and '1-x'
    x = parametric.Polynomial(["x"])
    x.add_term((1,), 1)

    invx = parametric.Polynomial(["x"])
    invx.add_term((1,), -1)
    invx.add_term((0,), 1)

    # we build the knuth yao dice using the bird model builder
    initial_state = bird.State(s=0)

    def delta(s: bird.State):
        match s.s:
            case 0:
                return [(x, bird.State(s=1)), (invx, bird.State(s=2))]
            case 1:
                return [(x, bird.State(s=3)), (invx, bird.State(s=4))]
            case 2:
                return [(x, bird.State(s=5)), (invx, bird.State(s=6))]
            case 3:
                return [(x, bird.State(s=1)), (invx, bird.State(s=7, d=1))]
            case 4:
                return [
                    (x, bird.State(s=7, d=2)),
                    (invx, bird.State(s=7, d=3)),
                ]
            case 5:
                return [
                    (x, bird.State(s=7, d=4)),
                    (invx, bird.State(s=7, d=5)),
                ]
            case 6:
                return [(x, bird.State(s=2)), (invx, bird.State(s=7, d=6))]
            case 7:
                return [(1, s)]

    # we add labels to the final states
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
