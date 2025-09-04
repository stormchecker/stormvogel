from stormvogel.model import Model, ModelType, Branch, Choice


def create_lion_mdp():
    lion = Model(model_type=ModelType.MDP)
    init = lion.get_initial_state()
    full = lion.new_state("full")
    satisfied = init
    hungry = lion.new_state("hungry :(")
    starving = lion.new_state("starving :((")
    dead = lion.new_state("dead")

    hunt = lion.new_action("hunt >:D")
    rawr = lion.new_action("rawr")

    init.set_choice([(1, satisfied)])

    full.set_choice(
        Choice(
            {
                hunt: Branch(
                    [
                        (0.5, satisfied),
                        (0.5, full),
                    ]
                ),
                rawr: Branch([(0.9, full), (0.1, satisfied)]),
            }
        )
    )

    satisfied.set_choice(
        Choice(
            {
                hunt: Branch([(0.5, satisfied), (0.3, full), (0.2, hungry)]),
                rawr: Branch([(0.9, satisfied), (0.1, hungry)]),
            }
        )
    )

    hungry.set_choice(
        Choice(
            {
                hunt: Branch(
                    [(0.2, full), (0.5, satisfied), (0.1, hungry), (0.2, starving)]
                ),
                rawr: Branch([(0.9, hungry), (0.1, starving)]),
            }
        )
    )

    starving.set_choice(
        Choice(
            {
                hunt: Branch(
                    [(0.1, full), (0.5, satisfied), (0.2, hungry), (0.2, dead)]
                ),
                rawr: Branch([(0.9, starving), (0.1, dead)]),
            }
        )
    )
    lion.add_self_loops()

    reward_model = lion.new_reward_model("R")
    reward_model.set_state_action_reward(full, rawr, 100)
    reward_model.set_unset_rewards(0)
    return lion
