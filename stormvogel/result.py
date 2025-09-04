import stormvogel.model
import random


class Scheduler:
    """
    Scheduler object specifies what action to take in each state

    Args:
        model: mdp model associated with the scheduler
        taken_actions: for each state the action we choose in that state
    """

    model: stormvogel.model.Model
    # taken actions are hashed by the state id
    taken_actions: dict[int, stormvogel.model.Action]

    # TODO functionality to convert a lambda scheduler to this object

    def __init__(
        self,
        model: stormvogel.model.Model,
        taken_actions: dict[int, stormvogel.model.Action],
    ):
        self.model = model
        self.taken_actions = taken_actions

    def get_choice_of_state(
        self, state: stormvogel.model.State | int
    ) -> stormvogel.model.Action:
        """returns the choice in the scheduler for the given state if present in the model"""
        if isinstance(state, int):
            state = self.model.get_state_by_id(state)
        if state in self.model.states.values():
            return self.taken_actions[state.id]
        else:
            raise RuntimeError("This state is not a part of the model")

    def generate_induced_dtmc(self) -> stormvogel.model.Model | None:
        """This function resolves the nondeterminacy of the mdp and returns the scheduler induced dtmc"""
        if self.model.get_type() == stormvogel.model.ModelType.MDP:
            induced_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)

            # we initialize the reward models
            for reward_model in self.model.rewards:
                induced_dtmc.new_reward_model(reward_model.name)

            # we add all the states and choices according to the choices
            for _, state in self.model:
                induced_dtmc.new_state(labels=state.labels, valuations=state.valuations)
                action = self.get_choice_of_state(state)
                choices = state.get_outgoing_choice(action)
                assert choices is not None
                induced_dtmc.add_choice(s=state, choices=choices)

                # we also add the rewards
                for reward_model in self.model.rewards:
                    induced_reward_model = induced_dtmc.get_rewards(reward_model.name)
                    reward = reward_model.get_state_action_reward(state, action)
                    assert reward is not None
                    induced_reward_model.set_state_reward(state, reward)

            return induced_dtmc

    def __str__(self) -> str:
        if self.model.name is not None:
            add = "Scheduler for model: " + self.model.name + "\n"
        else:
            add = ""
        return add + "taken actions: " + str(self.taken_actions)

    def __eq__(self, other) -> bool:
        if isinstance(other, Scheduler):
            return self.taken_actions == other.taken_actions
        return False


def random_scheduler(model: stormvogel.model.Model) -> Scheduler:
    """Create a random scheduler for the provided model."""
    choices = {i: random.choice(s.available_actions()) for (i, s) in model}
    return Scheduler(model, taken_actions=choices)


class Result:
    """Result object represents the model checking results for a given model

    Args:
        model: stormvogel representation of the model associated with the results
        values: for each state the model checking result
        scheduler: in case the model is an mdp we can optionally store a scheduler
    """

    model: stormvogel.model.Model
    # values are hashed by the state id:
    values: dict[int, stormvogel.model.Value]
    scheduler: Scheduler | None

    def __init__(
        self,
        model: stormvogel.model.Model,
        values: dict[int, stormvogel.model.Value],
        scheduler: Scheduler | None = None,
    ):
        self.model = model
        self.values = values

        if isinstance(scheduler, Scheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

    def get_result_of_state(
        self, state: stormvogel.model.State | int
    ) -> stormvogel.model.Value | None:
        """returns the model checking result for a given state"""
        if isinstance(state, int) and state in self.model.states.keys():
            return self.values[state]
        if (
            isinstance(state, stormvogel.model.State)
            and state in self.model.states.values()
        ):
            return self.values[state.id]
        else:
            raise RuntimeError("This state is not a part of the model")

    def __str__(self) -> str:
        add = ""
        if self.model.name is not None:
            add = "model: " + str(self.model.name) + "\n"
        return (
            add
            + "values: \n "
            + str(self.values)
            + "\n"
            + "scheduler: \n "
            + str(self.scheduler)
        )

    def maximum_result(self) -> stormvogel.model.Value:
        """Return the maximum result."""
        values = list(self.values.values())
        max_val = values[0]
        for v in values:
            if isinstance(v, stormvogel.model.Interval) or isinstance(
                v, stormvogel.parametric.Parametric
            ):
                raise RuntimeError(
                    "maximum result function does not work for interval/parametric models"
                )
            assert isinstance(v, stormvogel.model.Number)
            if v > max_val:
                max_val = v
        return max_val

    def __eq__(self, other) -> bool:
        if isinstance(other, Result):
            return self.values == other.values and self.scheduler == other.scheduler
        return False

    def __iter__(self):
        return iter(self.values.items())
