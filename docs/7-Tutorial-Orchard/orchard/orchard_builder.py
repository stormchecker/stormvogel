import stormpy
import stormvogel

from .orchard_game_stormvogel import (
    Orchard,
    available_actions,
    delta,
    labels,
    rewards,
    Fruit,
    GameState,
)


# Build the simple orchard model
def build_simple():
    # Initialize the simple game model
    init_small = Orchard([Fruit.APPLE, Fruit.CHERRY], num_fruits=2, raven_distance=2)

    # Build the orchard model
    orchard_simple = stormvogel.bird.build_bird(
        modeltype=stormvogel.ModelType.MDP,
        init=init_small,
        available_actions=available_actions,
        delta=delta,
        labels=labels,
        rewards=rewards,
    )
    return orchard_simple


# Build the full orchard model
def build_full():
    init_game = Orchard(
        [Fruit.APPLE, Fruit.CHERRY, Fruit.PEAR, Fruit.PLUM],
        num_fruits=4,
        raven_distance=5,
    )

    # For the full model, we only set the relevant labels for the winning conditions
    # and do not expose the internal state information
    def labels_full(state):
        labels = []
        if state.game_state() == GameState.PLAYERS_WON:
            labels.append("PlayersWon")
        elif state.game_state() == GameState.RAVEN_WON:
            labels.append("RavenWon")
        return labels

    orchard = stormvogel.bird.build_bird(
        modeltype=stormvogel.ModelType.MDP,
        init=init_game,
        available_actions=available_actions,
        delta=delta,
        labels=labels_full,
        rewards=rewards,
        max_size=100000,
    )

    # Convert to stormpy model
    orchard_storm = stormvogel.mapping.stormvogel_to_stormpy(orchard)

    return orchard, orchard_storm


def build_prism():
    """
    Builds the model from the prism program language model.
    :return: A tuple of the explicit MDP and the programmatic description.
    """
    prism_program = stormpy.parse_prism_program("orchard/orchard_stormvogel.pm")
    constants = "NUM_FRUIT=4,DISTANCE_RAVEN=5"
    prism_program = stormpy.preprocess_symbolic_input(prism_program, [], constants)[
        0
    ].as_prism_program()
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_all_labels()
    options.set_build_choice_labels()
    options.set_build_with_choice_origins()
    orchard_prism = stormpy.build_sparse_model_with_options(prism_program, options)
    return orchard_prism, prism_program
