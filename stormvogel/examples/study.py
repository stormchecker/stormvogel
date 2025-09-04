import stormvogel.model
from stormvogel.model import EmptyAction


def create_study_mdp():
    mdp = stormvogel.model.new_mdp()

    init = mdp.get_initial_state()
    study = mdp.action("study")
    not_study = mdp.action("don't study")

    studied = mdp.new_state("studied")
    not_studied = mdp.new_state("didn't study")
    pass_test = mdp.new_state("pass test")
    fail_test = mdp.new_state("fail test")
    end = mdp.new_state("end")

    init.set_choice([(study, studied), (not_study, not_studied)])

    studied.set_choice([(9 / 10, pass_test), (1 / 10, fail_test)])

    not_studied.set_choice([(4 / 10, pass_test), (6 / 10, fail_test)])
    mdp.add_self_loops()

    pass_test.set_choice([(1, end)])
    fail_test.set_choice([(1, end)])

    reward_model = mdp.new_reward_model("R")
    reward_model.set_state_action_reward(pass_test, EmptyAction, 100)
    reward_model.set_state_action_reward(fail_test, EmptyAction, 0)
    reward_model.set_state_action_reward(not_studied, EmptyAction, 15)
    reward_model.set_unset_rewards(0)
    return mdp
