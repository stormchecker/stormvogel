import pytest

import stormvogel.examples as examples
from stormvogel.graph import ModelGraph
from stormvogel.model import EmptyAction


@pytest.mark.parametrize(
    "model",
    [
        examples.create_monty_hall_mdp(),
        examples.create_monty_hall_pomdp(),
        examples.create_lion_mdp(),
        examples.create_car_mdp(),
        examples.create_die_dtmc(),
        examples.create_nuclear_fusion_ctmc(),
        examples.create_study_mdp(),
    ],
)
def test_graph_creation(model):
    G = ModelGraph.from_model(model)
    for state in model:
        assert state in G.nodes, f"Missing state {state} in ModelGraph"
        for action in state.available_actions():
            if action == EmptyAction:
                continue
            assert (
                (state, action) in G.nodes
            ), f"Mapping state: {state}, Action: {action} missing in ModelGraph"
