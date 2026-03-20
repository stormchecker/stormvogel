import pytest

import stormvogel.examples as examples
import stormvogel.teaching.bellman as bellman


@pytest.mark.parametrize(
    "input",
    [
        (examples.create_monty_hall_mdp(), "target"),
        (examples.create_lion_mdp(), "full"),
        # examples.create_car_mdp(),
        # examples.create_die_dtmc(),
        # examples.create_nuclear_fusion_ctmc(),
        # examples.create_study_mdp(),
    ],
)
def test_teaching_bellman(input):
    bellman.bellman_equations_maxreachprob(*input)
