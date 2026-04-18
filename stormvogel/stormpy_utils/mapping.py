import json
from typing import Union, cast

from stormvogel import parametric
from stormvogel.model.distribution import Distribution
from stormvogel.model.model import (
    Model,
    ModelType,
    new_mdp,
    new_ctmc,
    new_pomdp,
    new_ma,
)
from stormvogel.model.state import State
from stormvogel.model.choices import Choices, choices_from_shorthand
from stormvogel.model.action import EmptyAction
from stormvogel.model.value import Value, Interval
from stormvogel.model.variable import Variable

try:
    import stormpy
except ImportError:
    stormpy = None


def stormvogel_to_stormpy(model):
    """Convert a stormvogel model to a stormpy sparse model.

    :param model: The stormvogel model to convert.
    :returns: The equivalent stormpy sparse model.
    """
    from stormvogel.stormpy_utils.stormvogel_to_stormpy import (
        stormvogel_to_stormpy as _stormvogel_to_stormpy,
    )

    return _stormvogel_to_stormpy(model)


def value_to_stormvogel(value, sparsemodel) -> Value:
    """Convert a stormpy transition value to a stormvogel value.

    :param value: The stormpy transition value.
    :param sparsemodel: The stormpy sparse model providing context.
    :returns: The converted stormvogel value.
    """

    assert stormpy is not None

    if sparsemodel.has_parameters:
        # For parametric models, stormpy values are (factorized) rational
        # functions. We delegate the conversion to the active parametric
        # backend, which knows how to build a native expression from a pycarl
        # value. The backend returns a plain float when the value is a
        # constant (denominator 1, numerator constant), which keeps the
        # non-parametric arithmetic path in the caller unchanged.
        regular_form = value.rational_function()
        backend = parametric.get_default()
        return backend.from_pycarl(regular_form)
    else:
        # we check if our value is an interval
        if isinstance(value, stormpy.pycarl.Interval):
            # if lower and upper are the same, we return a singular value
            lower = float(value.lower())
            upper = float(value.upper())
            if lower == upper:
                return lower

            return Interval(lower, upper)

        # if our function is just a rational number we return a float:
        return float(value)


def stormpy_to_stormvogel(
    sparsemodel: Union[
        "stormpy.storage.SparseDtmc",
        "stormpy.storage.SparseMdp",
        "stormpy.storage.SparseCtmc",
        "stormpy.storage.SparsePomdp",
        "stormpy.storage.SparseMA",
    ],
) -> Model:
    """Convert a stormpy sparse model to a stormvogel model.

    :param sparsemodel: The stormpy sparse model to convert.
    :returns: The equivalent stormvogel model.
    :raises NotImplementedError: If the model type is not supported.
    """
    assert stormpy is not None

    def add_states(
        model: Model,
        sparsemodel: (
            stormpy.storage.SparseDtmc
            | stormpy.storage.SparseMdp
            | stormpy.storage.SparseCtmc
            | stormpy.storage.SparsePomdp
            | stormpy.storage.SparseMA
        ),
    ):
        """Add states from the sparse model to the stormvogel model.

        :param model: The stormvogel model to add states to.
        :param sparsemodel: The stormpy sparse model containing the states.
        """
        # First add all known labels to the model
        for label in sparsemodel.labeling.get_labels():
            model.add_label(label)

        for state in sparsemodel.states:
            if state.id == 0:
                if model.supports_observations():
                    if len(model.states) == 0:
                        model.new_state(
                            labels=list(state.labels),
                            observation=model.observation(str(state.id)),
                        )
                    else:
                        for label in state.labels:
                            model.states[0].add_label(label)
                else:
                    if len(model.states) == 0:
                        model.new_state(labels=list(state.labels))
                    else:
                        for label in state.labels:
                            model.states[0].add_label(label)
            if state.id > 0:
                if model.supports_observations():
                    model.new_state(
                        labels=list(state.labels),
                        observation=model.observation(str(state.id)),
                    )
                else:
                    model.new_state(labels=list(state.labels))

    def new_reward_model(
        model: Model,
        sparsemodel: (
            stormpy.storage.SparseDtmc
            | stormpy.storage.SparseMdp
            | stormpy.storage.SparseCtmc
            | stormpy.storage.SparsePomdp
            | stormpy.storage.SparseMA
        ),
    ):
        """Add reward models from the sparse model to the stormvogel model.

        :param model: The stormvogel model to add rewards to.
        :param sparsemodel: The stormpy sparse model containing the reward models.
        """
        for reward_model_name in sparsemodel.reward_models:
            rewards = sparsemodel.get_reward_model(reward_model_name)
            rewardmodel = model.new_reward_model(reward_model_name)
            if rewards.has_state_rewards:
                rewardmodel.set_from_rewards_vector(rewards.state_rewards)
            else:
                rewardmodel.set_from_rewards_vector(
                    rewards.state_action_rewards, state_action=True
                )

    def add_valuations(
        model: Model,
        sparsemodel: (
            stormpy.storage.SparseDtmc
            | stormpy.storage.SparseMdp
            | stormpy.storage.SparseCtmc
            | stormpy.storage.SparsePomdp
            | stormpy.storage.SparseMA
        ),
    ):
        """Add state valuations from the sparse model to the stormvogel model.

        :param model: The stormvogel model to add valuations to.
        :param sparsemodel: The stormpy sparse model containing the valuations.
        """
        if sparsemodel.has_state_valuations():
            valuations = sparsemodel.state_valuations

            for state_id, state in enumerate(model.states):
                v = json.loads(str(valuations.get_json(state_id)))
                if v is not None:
                    state.valuations = {
                        Variable(str(key)): value for key, value in v.items()
                    }

    def map_dtmc(sparsedtmc: stormpy.storage.SparseDtmc) -> Model:
        """Convert a stormpy DTMC to a stormvogel model.

        :param sparsedtmc: The stormpy sparse DTMC to convert.
        :returns: The equivalent stormvogel model.
        """

        # we create the model
        model = Model(ModelType.DTMC, create_initial_state=False)

        # pre-declare parameters so set_choices validation passes
        if sparsedtmc.has_parameters:
            for var in sparsedtmc.collect_all_parameters():
                model.declare_parameter(var.name)

        # we add the states
        add_states(model, sparsedtmc)

        # we add the transitions
        matrix = sparsedtmc.transition_matrix
        for state in sparsedtmc.states:
            row = matrix.get_row(state.id)
            choiceshorthand = [
                (
                    value_to_stormvogel(x.value(), sparsedtmc),
                    model.states[x.column],
                )
                for x in row
            ]
            choices = choices_from_shorthand(choiceshorthand)
            model.set_choices(model.states[state.id], choices)

        # we add the valuations
        add_valuations(model, sparsedtmc)

        # we add self loops to all states with no outgoing transition
        model.add_self_loops()

        # we add the reward models to the states
        new_reward_model(model, sparsedtmc)

        return model

    def map_mdp(sparsemdp: stormpy.storage.SparseMdp) -> Model:
        """Convert a stormpy MDP to a stormvogel model.

        :param sparsemdp: The stormpy sparse MDP to convert.
        :returns: The equivalent stormvogel model.
        """

        # we create the model
        model = new_mdp(create_initial_state=False)

        # pre-declare parameters so set_choices validation passes
        if sparsemdp.has_parameters:
            for var in sparsemdp.collect_all_parameters():
                model.declare_parameter(var.name)

        # we add the states
        add_states(model, sparsemdp)

        # we add the transitions
        matrix = sparsemdp.transition_matrix
        for index, state in enumerate(sparsemdp.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            choice = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                if sparsemdp.has_choice_labeling():
                    labels = sparsemdp.choice_labeling.get_labels_of_choice(i)
                    if labels:
                        action = model.action(",".join(sorted(labels)))
                    else:
                        action = EmptyAction
                else:
                    action = model.action(str(i))
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsemdp),
                        model.states[x.column],
                    )
                    for x in row
                ]
                choice[action] = Distribution(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.states[state.id], Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsemdp)

        # we add the valuations
        add_valuations(model, sparsemdp)

        return model

    def map_ctmc(sparsectmc: stormpy.storage.SparseCtmc) -> Model:
        """Convert a stormpy CTMC to a stormvogel model.

        :param sparsectmc: The stormpy sparse CTMC to convert.
        :returns: The equivalent stormvogel model.
        """

        # we create the model
        model = new_ctmc(create_initial_state=False)

        # we add the states
        add_states(model, sparsectmc)

        # we add the transitions
        matrix = sparsectmc.transition_matrix
        for state in sparsectmc.states:
            row = matrix.get_row(state.id)
            # In stormpy CTMCs, the transition values are already the individual rates (exit_rate * probability)
            # No need to multiply by exit_rate again
            choiceshorthand: list[tuple[Value, State]] = []
            for x in row:
                value = value_to_stormvogel(x.value(), sparsectmc)
                if value == 0:
                    continue
                choiceshorthand.append((value, model.states[x.column]))
            if choiceshorthand:
                choices = choices_from_shorthand(choiceshorthand)
                model.set_choices(model.states[state.id], choices)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the states
        new_reward_model(model, sparsectmc)

        # we add the valuations
        add_valuations(model, sparsectmc)

        return model

    def map_pomdp(sparsepomdp: stormpy.storage.SparsePomdp) -> Model:
        """Convert a stormpy POMDP to a stormvogel model.

        :param sparsepomdp: The stormpy sparse POMDP to convert.
        :returns: The equivalent stormvogel model.
        """
        model = new_pomdp(create_initial_state=False)
        if len(sparsepomdp.states) > 0:
            max_obs = max(
                sparsepomdp.get_observation(i) for i in range(len(sparsepomdp.states))
            )
            for i in range(max_obs + 1):
                model.observation(str(i))

        add_states(model, sparsepomdp)

        # we add the transitions
        matrix = sparsepomdp.transition_matrix
        for index, state in enumerate(sparsepomdp.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            choice = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                if sparsepomdp.has_choice_labeling():
                    labels = sparsepomdp.choice_labeling.get_labels_of_choice(i)
                    if labels:
                        action = model.action(",".join(sorted(labels)))
                    else:
                        action = EmptyAction
                else:
                    action = model.action(str(i))
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsepomdp),
                        model.states[x.column],
                    )
                    for x in row
                ]
                choice[action] = Distribution(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.states[state.id], Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsepomdp)

        # we add the valuations
        add_valuations(model, sparsepomdp)

        # map the observations to the states
        for index, state in enumerate(model.states):
            state.observation = model.observation(
                str(sparsepomdp.get_observation(index))
            )

        return model

    def map_ma(sparsema: stormpy.storage.SparseMA) -> Model:
        """Convert a stormpy MA to a stormvogel model.

        :param sparsema: The stormpy sparse MA to convert.
        :returns: The equivalent stormvogel model.
        """

        # we create the model
        model = new_ma(create_initial_state=False)

        # we add the states
        add_states(model, sparsema)

        # we add the transitions
        matrix = sparsema.transition_matrix
        for index, state in enumerate(sparsema.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            choice = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                if sparsema.has_choice_labeling():
                    labels = sparsema.choice_labeling.get_labels_of_choice(i)
                    if labels:
                        action = model.action(",".join(sorted(labels)))
                    else:
                        action = EmptyAction
                else:
                    action = model.action(str(i))
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsema),
                        model.states[x.column],
                    )
                    for x in row
                ]
                choice[action] = Distribution(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.states[state.id], Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsema)

        # we add the valuations
        add_valuations(model, sparsema)

        # we set the markovian states
        for state_id in list(sparsema.markovian_states):
            model.add_markovian_state(model.states[state_id])

        return model

    # we check the type to handle the sparse model correctly
    if sparsemodel.model_type.name == "DTMC":
        return map_dtmc(sparsemodel)
    elif sparsemodel.model_type.name == "MDP":
        return map_mdp(sparsemodel)
    elif sparsemodel.model_type.name == "CTMC":
        return map_ctmc(sparsemodel)
    elif sparsemodel.model_type.name == "POMDP":
        return map_pomdp(sparsemodel)
    elif sparsemodel.model_type.name == "MA":
        return map_ma(sparsemodel)
    else:
        raise NotImplementedError(
            "Converting this type of model to stormvogel is not yet supported"
        )


def from_prism(prism_code="stormpy.storage.storage.PrismProgram"):
    """Create a stormvogel model from a PRISM program.

    :param prism_code: The PRISM program to build from.
    :returns: The converted stormvogel model.
    """

    assert stormpy is not None
    return stormpy_to_stormvogel(stormpy.build_model(prism_code))
