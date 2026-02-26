import re
import json
from typing import Union, cast

from stormvogel import parametric
from stormvogel.model.model import (
    Model,
    ModelType,
    new_mdp,
    new_ctmc,
    new_pomdp,
    new_ma,
)
from stormvogel.model.state import State
from stormvogel.model.branches import Branches
from stormvogel.model.choices import Choices, choices_from_shorthand
from stormvogel.model.value import Value, Interval
from stormvogel.model.observation import Observation

try:
    import stormpy
except ImportError:
    stormpy = None


def stormvogel_to_stormpy(model):
    from stormvogel.stormpy_utils.stormvogel_to_stormpy import (
        stormvogel_to_stormpy as _stormvogel_to_stormpy,
    )

    return _stormvogel_to_stormpy(model)


def value_to_stormvogel(value, sparsemodel) -> Value:
    """Converts a stormpy transition value to a stormvogel one"""

    assert stormpy is not None

    def is_float(val) -> bool:
        """we check if we can convert a value to a float"""
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False

    def convert_polynomial(
        polynomial: stormpy.pycarl.cln.Polynomial,
    ) -> parametric.Polynomial:
        """helper function for converting pycarl polynomials to stormvogel polynomials"""

        # we create the list of variables
        # TODO make this more concise
        var = polynomial.gather_variables()
        var = re.sub(r"{|}", "", str(var))
        var = re.sub(r"<Variable (\w+).*?>", r"\1", var)
        variables_list = [v.strip() for v in var.split(",")]

        # we convert the polynomial to a more suitable list format
        parts = re.split(r"\+(?![^(]*\))", str(polynomial))
        stripped_parts = [part.replace("(", "").replace(")", "") for part in parts]
        term_list = []
        for p in stripped_parts:
            factors = re.split(r"[*^](?![^(]*\))", str(p))
            term_list.append(factors)

        # we initialize the polynomial
        stormvogel_polynomial = parametric.Polynomial(variables_list)

        # and then we add the terms to it
        for term in term_list:
            # we iterate through all terms, variables and exponents
            index_tuple = [0 for i in range(len(variables_list))]
            for i in range(len(term)):
                for j, var in enumerate(variables_list):
                    # we check if there is an exponent or not
                    if term[i] == var:
                        if i < len(term) - 1 and is_float(term[i + 1]):
                            index_tuple[j] = int(term[i + 1])
                        else:
                            index_tuple[j] = 1

            # we check if there is a coefficient at the beginning
            if is_float(term[0]):
                stormvogel_polynomial.add_term(tuple(index_tuple), float(term[0]))
            else:
                stormvogel_polynomial.add_term(tuple(index_tuple), float(1))

        return stormvogel_polynomial

    if sparsemodel.has_parameters:
        # if the model has parameters, all values are rational functions
        regular_form = value.rational_function()
        numerator = regular_form.numerator
        denominator = regular_form.denominator
        converted_numerator = convert_polynomial(numerator)

        # we only return a polynomial if the denominator is 1
        if denominator.is_constant():
            if float(denominator.constant_part()) == 1:
                if numerator.is_constant():
                    return float(numerator.constant_part())
                else:
                    return converted_numerator
        converted_denominator = convert_polynomial(denominator)
        stormvogel_rational_function = parametric.RationalFunction(
            converted_numerator, converted_denominator
        )
        return stormvogel_rational_function
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
) -> Model | None:
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
        """
        helper function to add the states from the sparsemodel to the model
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
                            observation=Observation(str(state.id)),
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
                        observation=Observation(str(state.id)),
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
        """
        adds the rewards from the sparsemodel to either the states or the state action pairs of the model
        """
        for reward_model_name in sparsemodel.reward_models:
            rewards = sparsemodel.get_reward_model(reward_model_name)
            rewardmodel = model.new_reward_model(reward_model_name)
            get_reward_vector = (
                rewards.state_action_rewards
                if rewards.has_state_action_rewards
                else rewards.state_rewards
            )

            rewardmodel.set_from_rewards_vector(get_reward_vector)

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
        """
        adds the valuations from the sparsemodel to the states of the model
        """
        if sparsemodel.has_state_valuations():
            valuations = sparsemodel.state_valuations

            for state_id, state in enumerate(model.states):
                v = json.loads(str(valuations.get_json(state_id)))
                if v is not None:
                    state.valuations = v

    def map_dtmc(sparsedtmc: stormpy.storage.SparseDtmc) -> Model:
        """
        Takes a dtmc stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = Model(ModelType.DTMC, create_initial_state=False)

        # we add the states
        add_states(model, sparsedtmc)

        # we add the transitions
        matrix = sparsedtmc.transition_matrix
        for state in sparsedtmc.states:
            row = matrix.get_row(state.id)
            choiceshorthand = [
                (
                    value_to_stormvogel(x.value(), sparsedtmc),
                    model.get_state_by_stormpy_id(x.column),
                )
                for x in row
            ]
            choices = choices_from_shorthand(choiceshorthand)
            model.set_choices(model.get_state_by_stormpy_id(state.id), choices)

        # we add the valuations
        add_valuations(model, sparsedtmc)

        # we add self loops to all states with no outgoing transition
        model.add_self_loops()

        # we add the reward models to the states
        new_reward_model(model, sparsedtmc)

        return model

    def map_mdp(sparsemdp: stormpy.storage.SparseMdp) -> Model:
        """
        Takes a mdp stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = new_mdp(create_initial_state=False)

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
                    actionlabel = ",".join(sorted(labels)) if labels else None
                else:
                    actionlabel = str(i)

                action = model.action(actionlabel)
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsemdp),
                        model.get_state_by_stormpy_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.get_state_by_stormpy_id(state.id), Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsemdp)

        # we add the valuations
        add_valuations(model, sparsemdp)

        return model

    def map_ctmc(sparsectmc: stormpy.storage.SparseCtmc) -> Model:
        """
        Takes a ctmc stormpy representation as input and outputs a simple stormvogel representation
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
            choiceshorthand = [
                (
                    value_to_stormvogel(x.value(), sparsectmc),
                    model.get_state_by_stormpy_id(x.column),
                )
                for x in row
            ]
            choices = choices_from_shorthand(choiceshorthand)
            model.set_choices(model.get_state_by_stormpy_id(state.id), choices)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the states
        new_reward_model(model, sparsectmc)

        # we add the valuations
        add_valuations(model, sparsectmc)

        return model

    def map_pomdp(sparsepomdp: stormpy.storage.SparsePomdp) -> Model:
        # Takes a pomdp stormpy representation as input and outputs a simple stormvogel representation
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
                    actionlabel = ",".join(sorted(labels)) if labels else None
                else:
                    actionlabel = str(i)

                action = model.action(actionlabel)
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsepomdp),
                        model.get_state_by_stormpy_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.get_state_by_stormpy_id(state.id), Choices(choice))

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
                    actionlabel = ",".join(sorted(labels)) if labels else None
                else:
                    actionlabel = str(i)

                action = model.action(actionlabel)
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsepomdp),
                        model.get_state_by_stormpy_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.get_state_by_stormpy_id(state.id), Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsepomdp)

        # we add the valuations
        add_valuations(model, sparsepomdp)

        # we add the observations:
        max_obs = (
            max(sparsepomdp.get_observation(i) for i in range(len(sparsepomdp.states)))
            if len(sparsepomdp.states) > 0
            else -1
        )
        for i in range(max_obs + 1):
            model.observation(str(i))
        for index, state in enumerate(model.states):
            state.observation = model.observation(
                str(sparsepomdp.get_observation(index))
            )

        return model

    def map_ma(sparsema: stormpy.storage.SparseMA) -> Model:
        # Takes a ma stormpy representation as input and outputs a simple stormvogel representation

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
                    actionlabel = ",".join(sorted(labels)) if labels else None
                else:
                    actionlabel = str(i)

                action = model.action(actionlabel)
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsema),
                        model.get_state_by_stormpy_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
            model.set_choices(model.get_state_by_stormpy_id(state.id), Choices(choice))

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsema)

        # we add the valuations
        add_valuations(model, sparsema)

        # we set the markovian states
        for state_id in list(sparsema.markovian_states):
            model.add_markovian_state(model.get_state_by_stormpy_id(state_id))

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
    # Create a model from prism.

    assert stormpy is not None
    return stormpy_to_stormvogel(stormpy.build_model(prism_code))
