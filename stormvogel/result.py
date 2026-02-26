import stormvogel.model
import random


class Scheduler:
    """
    Scheduler object specifies what action to take in each state.
    All schedulers are nondeterminstic and memoryless.

    Args:
        model: model associated with the scheduler (has to support actions).
        taken_actions: for each state the action we choose in that state.
    """

    model: stormvogel.model.Model
    # taken actions are hashed by the state id
    taken_actions: dict[stormvogel.model.State, stormvogel.model.Action]

    def __init__(
        self,
        model: stormvogel.model.Model,
        taken_actions: dict[stormvogel.model.State, stormvogel.model.Action],
    ):
        self.model = model
        self.taken_actions = taken_actions

    def get_action_at_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Action:
        """returns the action in the scheduler for the given state if present in the model"""
        if state in self.model.states:
            return self.taken_actions[state]
        else:
            raise RuntimeError("This state is not a part of the model")

    def generate_induced_dtmc(self) -> stormvogel.model.Model | None:
        """This function resolves the nondeterminacy of the mdp and returns the scheduler induced dtmc"""
        if self.model.model_type == stormvogel.model.ModelType.MDP:
            induced_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)

            # we initialize the reward models
            for reward_model in self.model.rewards:
                induced_dtmc.new_reward_model(reward_model.name)

            # build a mapping from old states to new states
            state_map: dict[stormvogel.model.State, stormvogel.model.State] = {}
            for state in self.model:
                new_state = induced_dtmc.new_state(
                    labels=list(state.labels), valuations=state.valuations
                )
                state_map[state] = new_state

            # add transitions with remapped state references
            for state in self.model:
                new_state = state_map[state]
                action = self.get_action_at_state(state)
                transitions = state.get_outgoing_transitions(action)
                assert transitions is not None
                # remap branch targets from MDP states to induced DTMC states
                remapped = [(prob, state_map[target]) for prob, target in transitions]
                induced_dtmc.set_choices(new_state, remapped)

                # we also add the rewards
                for reward_model in self.model.rewards:
                    induced_reward_model = induced_dtmc.get_rewards(reward_model.name)
                    reward = reward_model.get_state_reward(state)
                    if reward is not None:
                        induced_reward_model.set_state_reward(new_state, reward)

            return induced_dtmc

    def __str__(self) -> str:
        return "taken actions: " + str(self.taken_actions)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Scheduler):
            return False

        return self.taken_actions == other.taken_actions


def random_scheduler(model: stormvogel.model.Model) -> Scheduler:
    """Create a random scheduler for the provided model."""
    choices = {
        state: random.choice(state.available_actions())
        for state in model
        if state.available_actions()
    }
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
    values: dict[stormvogel.model.State, stormvogel.model.Value]
    scheduler: Scheduler | None

    def __init__(
        self,
        model: stormvogel.model.Model,
        values: dict[stormvogel.model.State, stormvogel.model.Value],
        scheduler: Scheduler | None = None,
    ):
        self.model = model
        self.values = values

        if isinstance(scheduler, Scheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Value | None:
        """returns the model checking result for a given state"""
        if isinstance(state, stormvogel.model.State) and state in self.values:
            return self.values[state]
        else:
            raise RuntimeError("This state is not a part of the model")

    def __str__(self) -> str:
        return (
            "values: \n "
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
        if not isinstance(other, Result):
            return False

        if len(self.values) != len(other.values):
            return False

        for index, s1 in enumerate(self.model.states):
            if s1 not in self.values:
                continue
            v1 = self.values[s1]
            s2 = other.model.states[index]
            if s2 not in other.values:
                return False
            v2 = other.values[s2]
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                if v1 != v2:
                    return False
            elif abs(float(v1) - float(v2)) > 1e-6:
                return False

        if (self.scheduler is None) != (other.scheduler is None):
            return False

        if self.scheduler is not None and other.scheduler is not None:
            if self.scheduler.taken_actions != other.scheduler.taken_actions:
                for index, s1 in enumerate(self.model.states):
                    if s1 in self.scheduler.taken_actions:
                        a1 = self.scheduler.taken_actions[s1]
                        s2 = other.model.states[index]
                        if (
                            s2 not in other.scheduler.taken_actions
                            or other.scheduler.taken_actions[s2] != a1
                        ):
                            return False
        return True

    def __iter__(self):
        return iter(self.values.items())
