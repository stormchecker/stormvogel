from stormvogel import model, bird


def create_knuth_yao_imc():
    # We create our interval values
    interval = model.Interval(2 / 7, 6 / 7)
    inv_interval = model.Interval(1 / 7, 5 / 7)

    # we build the knuth yao dice using the bird model builder
    def delta(s: bird.BirdState):
        match s.s:
            case 0:
                return [
                    (interval, bird.BirdState(s=1)),
                    (inv_interval, bird.BirdState(s=2)),
                ]
            case 1:
                return [
                    (interval, bird.BirdState(s=3)),
                    (inv_interval, bird.BirdState(s=4)),
                ]
            case 2:
                return [
                    (interval, bird.BirdState(s=5)),
                    (inv_interval, bird.BirdState(s=6)),
                ]
            case 3:
                return [
                    (interval, bird.BirdState(s=1)),
                    (inv_interval, bird.BirdState(s=7, d=1)),
                ]
            case 4:
                return [
                    (interval, bird.BirdState(s=7, d=2)),
                    (inv_interval, bird.BirdState(s=7, d=3)),
                ]
            case 5:
                return [
                    (interval, bird.BirdState(s=7, d=4)),
                    (inv_interval, bird.BirdState(s=7, d=5)),
                ]
            case 6:
                return [
                    (interval, bird.BirdState(s=2)),
                    (inv_interval, bird.BirdState(s=7, d=6)),
                ]
            case 7:
                return [(1, s)]

    def labels(s: bird.BirdState):
        if s.s == 7:
            return f"rolled{str(s.d)}"

    knuth_yao_imc = bird.build_bird(
        delta=delta,
        init=bird.BirdState(s=0),
        labels=labels,
        modeltype=model.ModelType.DTMC,
    )

    return knuth_yao_imc


if __name__ == "__main__":
    print(create_knuth_yao_imc())
