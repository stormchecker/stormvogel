import stormvogel.model


def create_study_mdp():
    mdp = stormvogel.model.new_mdp()

    init = mdp.get_initial_state()
    study = mdp.action("study")
    not_study = mdp.action("don't study")

    pass_test = mdp.new_state("pass test")
    fail_test = mdp.new_state("fail test")
    end = mdp.new_state("end")

    init.set_choice(
        stormvogel.model.Choice(
            {
                study: stormvogel.model.Branch(
                    [(9 / 10, pass_test), (1 / 10, fail_test)]
                ),
                not_study: stormvogel.model.Branch(
                    [(4 / 10, pass_test), (6 / 10, fail_test)]
                ),
            }
        )
    )

    pass_test.set_choice([(1, end)])
    fail_test.set_choice([(1, end)])

    reward_model = mdp.new_reward_model("R")
    reward_model.set_state_action_reward(pass_test, stormvogel.model.EmptyAction, 100)
    reward_model.set_state_action_reward(fail_test, stormvogel.model.EmptyAction, 0)
    reward_model.set_state_action_reward(init, not_study, 15)
    reward_model.set_state_action_reward(init, study, 0)
    reward_model.set_unset_rewards(0)

    return mdp
