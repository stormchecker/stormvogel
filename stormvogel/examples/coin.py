from stormvogel import *

def create_coin_mdps():
    init = ("flip",)

    def available_actions(s):
        if "heads" in s or "tails" in s:
            return ["guess_heads", "guess_tails"]
        return [""]

    def delta(s, a):
        if s == init:
            return [(0.5, ("heads",)), (0.5, ("tails",))]
        elif a.startswith("guess"):
            if "heads" in s and a == "guess_heads" or "tails" in s and a == "guess_tails":
                return [(1, ("correct", "done"))]
            else:
                return [(1, ("wrong", "done"))]
        else:
            return [(1, s)]


    labels = lambda s: list(s)


    def rewards(s, a):
        if "correct" in s:
            return {"R": 100}
        return {"R": 0}


    coin_mdp = bird.build_bird(
        delta=delta,
        init=init,
        available_actions=available_actions,
        labels=labels,
        modeltype=ModelType.MDP,
        rewards=rewards,
    )

    def observations(s):
        return 0


    coin_pomdp = bird.build_bird(
        delta=delta,
        init=init,
        available_actions=available_actions,
        labels=labels,
        modeltype=ModelType.POMDP,
        rewards=rewards,
        observations=observations,
    )

    def observations_stochastic(s):
        if "heads" in s:
            return [(0.8, 0), (0.2, 1)]
        elif "tails" in s:
            return [(0.2, 0), (0.8, 1)]
        else:
            return [(1.0, 2)]

    def observation_valuations(o):
        if o == 0:
            return {"heads": True, "tails": False, "done": False}
        elif o == 1:
            return {"heads": False, "tails": True, "done": False}
        else:
            return {"done": True, "heads": False, "tails": False}

    coin_pomdp_stochastic = bird.build_bird(
        delta=delta,
        init=init,
        available_actions=available_actions,
        labels=labels,
        modeltype=ModelType.POMDP,
        rewards=rewards,
        observations=observations_stochastic,
        observation_valuations=observation_valuations,
    )

    return coin_mdp, coin_pomdp, coin_pomdp_stochastic

if __name__ == "__main__":
    print(create_coin_mdps())