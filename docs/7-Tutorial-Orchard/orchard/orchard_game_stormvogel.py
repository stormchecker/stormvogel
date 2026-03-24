# Stormvogel model of the orchard game

# Imports
import stormvogel

from enum import Enum
from copy import deepcopy


# General data structures
class Fruit(Enum):
    APPLE = "🍏"
    PEAR = "🍐"
    CHERRY = "🍒"
    PLUM = "🍇"  # Use grape as there is no Emoji for plum

    def __str__(self):
        return self.value


class DiceOutcome(Enum):
    FRUIT = "🍉"
    BASKET = "🧺"
    RAVEN = "🐦‍⬛"

    def __str__(self):
        return self.value


class GameState(Enum):
    NOT_ENDED = 0
    PLAYERS_WON = 1
    RAVEN_WON = 2


# Main class for the orchard game
class Orchard(stormvogel.bird.State):
    def __init__(self, fruit_types, num_fruits, raven_distance):
        self.trees = {fruit: num_fruits for fruit in fruit_types}
        self.raven = raven_distance
        self.dice = None

    def game_state(self):
        if all(self.is_tree_empty(tree) for tree in self.trees.keys()):
            assert self.raven > 0
            return GameState.PLAYERS_WON
        elif self.raven == 0:
            return GameState.RAVEN_WON
        else:
            return GameState.NOT_ENDED

    def is_tree_empty(self, tree):
        return self.trees[tree] == 0

    def pick_fruit(self, fruit):
        if self.trees[fruit] > 0:
            self.trees[fruit] -= 1

    def next_round(self):
        self.dice = None

    def move_raven(self):
        self.raven -= 1

    def __hash__(self):
        trees = [hash((f, n)) for f, n in self.trees.items()]
        return hash((tuple(trees), self.raven, self.dice))

    def label(self):
        if self.dice is None:
            # Output game state
            return str(self)
        else:
            if self.dice[0] == DiceOutcome.FRUIT:
                return "🎲" + str(self.dice[1])
            else:
                return "🎲" + str(self.dice[0])

    def __str__(self):
        s = ", ".join(["{}{}".format(n, f) for f, n in self.trees.items()])
        s += ", {}←{}".format(self.raven, DiceOutcome.RAVEN)
        return s


# Define available actions
def available_actions(state):
    if state.game_state() != GameState.NOT_ENDED:
        return ["gameEnded"]
    if state.dice is None:
        return ["nextRound"]
    if state.dice[0] == DiceOutcome.FRUIT:
        return ["pick{}".format(state.dice[1].name)]
    if state.dice[0] == DiceOutcome.BASKET:
        available_fruits = []
        # Choice over available fruits
        for fruit in state.trees.keys():
            if not state.is_tree_empty(fruit):
                available_fruits.append(fruit)
        return ["choose{}".format(fruit.name) for fruit in available_fruits]
    if state.dice[0] == DiceOutcome.RAVEN:
        return ["moveRaven"]
    assert False


# The transition function
def delta(state, action):
    if state.game_state() != GameState.NOT_ENDED:
        # Game has ended -> self loop
        return [(1, state)]

    if state.dice is None:
        # Player throws dice and considers outcomes
        outcomes = []
        # Probability of fair dice throw over
        # each fruit type + 1 basket + 1 raven
        fair_dice_prob = 1 / (len(state.trees.keys()) + 2)

        # 1. Dice shows fruit
        for fruit in state.trees.keys():
            next_state = deepcopy(state)
            next_state.dice = DiceOutcome.FRUIT, fruit
            outcomes.append((fair_dice_prob, next_state))

        # 2. Dice shows basket
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.BASKET, None
        outcomes.append((fair_dice_prob, next_state))

        # 3. Dice shows raven
        next_state = deepcopy(state)
        next_state.dice = DiceOutcome.RAVEN, None
        outcomes.append((fair_dice_prob, next_state))
        return outcomes

    elif state.dice[0] == DiceOutcome.FRUIT:
        # Player picks specified fruit
        fruit = state.dice[1]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(1, next_state)]

    elif state.dice[0] == DiceOutcome.BASKET:
        assert action.startswith("choose")
        # Player chooses fruit specified by action
        fruit = Fruit[action.removeprefix("choose")]
        next_state = deepcopy(state)
        next_state.pick_fruit(fruit)
        next_state.next_round()
        return [(1, next_state)]

    elif state.dice[0] == DiceOutcome.RAVEN:
        next_state = deepcopy(state)
        next_state.move_raven()
        next_state.next_round()
        return [(1, next_state)]

    assert False


# Add labels for game state
def labels(state):
    labels = [state.label()]
    if state.game_state() == GameState.PLAYERS_WON:
        labels.append("PlayersWon")
    elif state.game_state() == GameState.RAVEN_WON:
        labels.append("RavenWon")
    return labels


# Reward function
def rewards(state, action):
    if state.game_state() == GameState.NOT_ENDED:
        if state.dice is None:
            return {"rounds": 1}
    return {"rounds": 0}
