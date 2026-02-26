import stormvogel.model


def create_study_mdp():
    mdp = stormvogel.model.new_mdp()

    init = mdp.initial_state
    study = mdp.action("study")
    not_study = mdp.action("don't study")

    did_study = mdp.new_state("did study")
    did_not_study = mdp.new_state("did not study")

    pass_test = mdp.new_state("pass test")
    fail_test = mdp.new_state("fail test")
    end = mdp.new_state("end")

    init.set_choices(
        {
            study: [(1, did_study)],
            not_study: [(1, did_not_study)],
        }
    )

    did_study.set_choices([(9 / 10, pass_test), (1 / 10, fail_test)])
    did_not_study.set_choices([(4 / 10, pass_test), (6 / 10, fail_test)])

    pass_test.set_choices([(1, end)])
    fail_test.set_choices([(1, end)])

    mdp.add_self_loops()

    reward_model = mdp.new_reward_model("R")
    reward_model.set_state_reward(pass_test, 100)
    reward_model.set_state_reward(fail_test, 0)
    reward_model.set_state_reward(did_study, 15)
    reward_model.set_state_reward(did_not_study, 0)
    reward_model.set_unset_rewards(0)

    return mdp
