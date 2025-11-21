import re
import json
from typing import Optional, Union, cast

from stormvogel import parametric
from stormvogel.model.action import EmptyAction
from stormvogel.model.model import Model, ModelType
from stormvogel.model.value import Number, Value, Interval

try:
    import stormpy
except ImportError:
    stormpy = None



def stormvogel_to_stormpy(
    model: Model,
) -> Optional[
    Union[
        "stormpy.storage.SparseDtmc",
        "stormpy.storage.SparseMdp",
        "stormpy.storage.SparseCtmc",
        "stormpy.storage.SparsePomdp",
    ]
]:
    assert stormpy is not None

    def build_matrix(
        model: Model,
        choice_labeling: stormpy.storage.ChoiceLabeling | None,
    ) -> stormpy.storage.SparseMatrix:
        """
        Takes a model and creates a stormpy sparsematrix that represents the same choices.
        We also create the choice_labeling by reference simultaneously
        """

        assert stormpy is not None

        # we precompute the following two values
        nondeterministic = model.supports_actions()
        is_parametric = model.is_parametric()
        is_interval = model.is_interval_model()

        # we distinguish between parametric, interval and regular models
        if is_parametric:
            builder = stormpy.ParametricSparseMatrixBuilder(
                rows=0,
                columns=0,
                entries=0,
                force_dimensions=False,
                has_custom_row_grouping=nondeterministic,
                row_groups=0,
            )
        elif is_interval:
            builder = stormpy.IntervalSparseMatrixBuilder(
                rows=0,
                columns=0,
                entries=0,
                force_dimensions=False,
                has_custom_row_grouping=nondeterministic,
                row_groups=0,
            )
        else:
            builder = stormpy.SparseMatrixBuilder(
                rows=0,
                columns=0,
                entries=0,
                force_dimensions=False,
                has_custom_row_grouping=nondeterministic,
                row_groups=0,
            )

        # we build the matrix
        row_index = 0
        for transition in sorted(model.choices.items()):
            if nondeterministic:
                builder.new_row_group(row_index)
            for action in transition[1]:
                action[1].sort_states()
                for tuple in action[1]:
                    val = value_to_stormpy(tuple[0], variables, model)
                    builder.add_next_value(
                        row=row_index,
                        column=model.stormpy_id[tuple[1].id],
                        value=val,
                    )

                # if there is an action then add the label to the choice
                if (
                    not action[0] == EmptyAction
                    and choice_labeling is not None
                ):
                    if action[0].label is not None:
                        choice_labeling.add_label_to_choice(action[0].label, row_index)
                row_index += 1

        return builder.build()

    def add_labels(model: Model) -> stormpy.storage.StateLabeling:
        """
        Takes a model and creates a state labelling object that determines which states get which labels in the stormpy representation
        """
        assert stormpy is not None

        # we first initialize all labels
        state_labeling = stormpy.storage.StateLabeling(len(model.states.keys()))
        for label in model.get_labels():
            state_labeling.add_label(label)

        # then we assign the labels to the correct states
        for state in model:
            for label in state.labels:
                state_labeling.add_label_to_state(label, model.stormpy_id[state])

        return state_labeling

    def new_reward_model(
        model: Model,
    ) -> dict[str, stormpy.SparseRewardModel]:
        """
        Takes a model and creates a dictionary of all the stormpy representations of reward models
        """
        assert stormpy is not None

        reward_models = {}
        for rewardmodel in model.rewards:
            reward_models[rewardmodel.name] = stormpy.SparseRewardModel(
                optional_state_action_reward_vector=rewardmodel.get_reward_vector()
            )

        return reward_models

    def add_valuations(model: Model) -> stormpy.storage.StateValuation:
        """
        Helps to add the valuations to the sparsemodel using a statevaluation object
        """
        assert stormpy is not None

        manager = stormpy.ExpressionManager()
        valuations = stormpy.storage.StateValuationsBuilder()

        # we create all the variable names
        created_vars = set()
        for state in model.states:
            for var in sorted(state.valuations.items()):
                name = str(var[0])
                if name not in created_vars:
                    storm_var = manager.create_integer_variable(name)
                    valuations.add_variable(storm_var)
                    created_vars.add(name)

        # we assign the values to the variables in the states
        for state in model.states:
            valuations.add_state(
                model.stormpy_id[state],
                integer_values=list(state.valuations.values()),
            )

        return valuations.build()

    def map_dtmc(model: Model) -> stormpy.storage.SparseDtmc:
        """
        Takes a simple representation of a dtmc as input and outputs a dtmc how it is represented in stormpy
        """
        assert stormpy is not None

        # we first build the SparseMatrix
        matrix = build_matrix(model, None)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = new_reward_model(model)

        # we add the valuations
        valuations = add_valuations(model)

        # then we build the dtmc
        if model.is_parametric():
            components = stormpy.SparseParametricModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            dtmc = stormpy.storage.SparseParametricDtmc(components)
        elif model.is_interval_model():
            components = stormpy.SparseIntervalModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            dtmc = stormpy.storage.SparseIntervalDtmc(components)
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            dtmc = stormpy.storage.SparseDtmc(components)

        return dtmc

    def map_mdp(model: Model) -> stormpy.storage.SparseMdp:
        """
        Takes a simple representation of an mdp as input and outputs an mdp how it is represented in stormpy
        """
        assert stormpy is not None

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model:
            for action in state.available_actions():
                count += 1
                if action != EmptyAction and action.label is not None:
                    labels.add(action.label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = new_reward_model(model)

        # we add the valuations
        valuations = add_valuations(model)

        # then we build the mdp
        if model.is_parametric():
            components = stormpy.SparseParametricModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.choice_labeling = choice_labeling
            mdp = stormpy.storage.SparseParametricMdp(components)
        elif model.is_interval_model():
            components = stormpy.SparseIntervalModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            components.choice_labeling = choice_labeling
            mdp = stormpy.storage.SparseIntervalMdp(components)
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            components.choice_labeling = choice_labeling
            mdp = stormpy.storage.SparseMdp(components)

        return mdp

    def map_ctmc(model: Model) -> stormpy.storage.SparseCtmc:
        """
        Takes a simple representation of a ctmc as input and outputs a ctmc how it is represented in stormpy
        """
        assert stormpy is not None

        # we first build the SparseMatrix (in stormvogel these are always the rate choices)
        matrix = build_matrix(model, None)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = new_reward_model(model)

        # we add the valuations
        valuations = add_valuations(model)

        # then we build the ctmc and we add the exit rates if necessary
        if model.is_parametric():
            components = stormpy.SparseParametricModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                rate_choices=True,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())

            ctmc = stormpy.storage.SparseParametricCtmc(components)
        elif model.is_interval_model():
            components = stormpy.SparseIntervalModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                rate_transitions=True,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())

            ctmc = stormpy.storage.SparseIntervalCtmc(components)
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                rate_transitions=True,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())

            ctmc = stormpy.storage.SparseCtmc(components)

        return ctmc

    def map_pomdp(model: Model) -> stormpy.storage.SparsePomdp:
        """
        Takes a simple representation of an pomdp as input and outputs an pomdp how it is represented in stormpy
        """
        assert stormpy is not None

        # Check if the model contains stochastic observation, since we need to do a transformation in that case
        for state in model.states:
            if isinstance(state.observation, list):
                raise NotImplementedError(
                    "Stormpy does not support stochastic observations in POMDPs. Please convert the stochastic observations to deterministic ones before converting the model."
                )

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model.states:
            for action in state.available_actions():
                count += 1
                if (
                    not action == EmptyAction
                    and action.label is not None
                ):
                    labels.add(action.label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = new_reward_model(model)

        # we add the valuations
        valuations = add_valuations(model)

        # then we build the pomdp
        if model.is_parametric():
            components = stormpy.SparseParametricModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            observations = []
            for state in model.states:
                if state.get_observation() is not None:
                    observations.append(state.get_observation().get_observation())
                else:
                    raise RuntimeError(
                        f"State {state} does not have an observation. Please assign an observation to each state."
                    )

            components.observability_classes = observations
            components.choice_labeling = choice_labeling
            pomdp = stormpy.storage.SparseParametricPomdp(components)
        elif model.is_interval_model():
            components = stormpy.SparseIntervalModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            observations = []
            for state in model.states:
                if state.get_observation() is not None:
                    observations.append(state.get_observation().get_observation())
                else:
                    raise RuntimeError(
                        f"State {state} does not have an observation. Please assign an observation to each state."
                    )

            components.observability_classes = observations
            components.choice_labeling = choice_labeling
            pomdp = stormpy.storage.SparseIntervalPomdp(components)
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
            )
            components.state_valuations = valuations
            observations = []
            for state in model.states:
                if state.get_observation() is None:
                    raise RuntimeError(
                        f"State {state} does not have an observation. Please assign an observation to each state."
                    )
                elif isinstance(state.get_observation(), list):
                    raise NotImplementedError(
                        "Stormpy does not support stochastic observations in POMDPs. Please convert the stochastic observations to deterministic ones before converting the model."
                    )
                else:
                    observations.append(state.get_observation().observation)

            components.observability_classes = observations
            components.choice_labeling = choice_labeling
            pomdp = stormpy.storage.SparsePomdp(components)

        return pomdp

    def map_ma(model: Model) -> stormpy.storage.SparseMA:
        """
        Takes a simple representation of an ma as input and outputs an ma how it is represented in stormpy
        """
        assert stormpy is not None

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model.states:
            for action in state.available_actions():
                count += 1
                if (
                    not action == EmptyAction
                    and action.label is not None
                ):
                    labels.add(action.label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = new_reward_model(model)

        # we add the valuations
        valuations = add_valuations(model)

        # we create the list of markovian state ids
        assert model.markovian_states is not None
        markovian_states_list = [state for state in model.markovian_states]
        if isinstance(markovian_states_list, list):
            markovian_states_bitvector = stormpy.storage.BitVector(
                max(markovian_states_list) + 1,
                markovian_states_list,
            )
        else:
            markovian_states_bitvector = stormpy.storage.BitVector(0)

        # then we build the ma
        if model.is_parametric():
            components = stormpy.SparseParametricModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                markovian_states=markovian_states_bitvector,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())
            else:
                components.exit_rates = []
            components.choice_labeling = choice_labeling
            ma = stormpy.storage.SparseParametricMA(components)
        elif model.is_interval_model():
            components = stormpy.SparseIntervalModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                markovian_states=markovian_states_bitvector,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())
            else:
                components.exit_rates = []
            components.choice_labeling = choice_labeling
            ma = stormpy.storage.SparseIntervalMA(components)
        else:
            components = stormpy.SparseModelComponents(
                transition_matrix=matrix,
                state_labeling=state_labeling,
                reward_models=reward_models,
                markovian_states=markovian_states_bitvector,
            )
            components.state_valuations = valuations
            if not model.exit_rates == {} and model.exit_rates is not None:
                components.exit_rates = list(model.exit_rates.values())
            else:
                components.exit_rates = []
            components.choice_labeling = choice_labeling
            ma = stormpy.storage.SparseMA(components)

        return ma

    # we throw the neccessary errors first
    if not model.all_states_outgoing_transition():
        raise RuntimeError(
            "This model has states with no outgoing transitions.\nUse the add_self_loops() function to add self loops to all states with no outgoing transition."
        )

    if model.unassigned_variables():
        raise RuntimeError("Each state should have a value for each variable")

    if not model.all_non_init_states_incoming_transition():
        raise RuntimeError(
            "There is more than one state in this model without incoming transitions."
        )

    if model.has_zero_transition() and not model.supports_rates():
        raise RuntimeError(
            "This model has transitions with probability=0. Stormpy assumes that these do not explicitly exist."
        )

    # we make a mapping between stormvogel and stormpy ids in case they are out of order.
    stormpy_id = {}
    for index, stormvogel_id in enumerate(model.states.keys()):
        stormpy_id[stormvogel_id] = index
    model.stormpy_id = stormpy_id

    # we store the pycarl parameters of a model
    stormpy.pycarl.clear_variable_pool()
    variables = []
    for p in model.get_parameters():
        var = stormpy.pycarl.Variable(p)
        variables.append(var)

    # we check the type to handle the model correctly
    if model.type == ModelType.DTMC:
        return map_dtmc(model)
    elif model.type == ModelType.MDP:
        return map_mdp(model)
    elif model.type == ModelType.CTMC:
        return map_ctmc(model)
    elif model.type == ModelType.POMDP:
        return map_pomdp(model)
    elif model.type == ModelType.MA:
        return map_ma(model)
    else:
        raise NotImplementedError(
            "Converting this type of model to stormpy is not yet supported"
        )


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
        model.new_state()
        for state in sparsemodel.states:
            if state == 0:
                for label in state.labels:
                    model.get_state_by_id(0).add_label(label)
            if state > 0:
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

            for state_id, state in model.states.items():
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
            row = matrix.get_row(state)
            ChoicesShorthand = [
                (
                    value_to_stormvogel(x.value(), sparsedtmc),
                    model.get_state_by_stormpy_id(x.column),
                )
                for x in row
            ]
            set_choiceschoice_from_shorthand(
                cast(
                    list[tuple[Value, State]],
                    ChoicesShorthand,
                ),
                model,
            )
            model.set_choicesmodel.get_state_by_id(state), choices

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
                        model.get_state_by_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
                model.set_choicesmodel.get_state_by_id(state), choice

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
            row = matrix.get_row(state)
            ChoicesShorthand = [
                (
                    value_to_stormvogel(x.value(), sparsectmc),
                    model.get_state_by_id(x.column),
                )
                for x in row
            ]
            set_choiceschoice_from_shorthand(
                cast(
                    list[tuple[Value, State]],
                    ChoicesShorthand,
                ),
                model,
            )
            model.set_choicesmodel.get_state_by_id(state), choices

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the states
        new_reward_model(model, sparsectmc)

        # we add the valuations
        add_valuations(model, sparsectmc)

        # we set the correct exit rates
        for state in model.states.items():
            model.set_rate(state[1], sparsectmc.exit_rates[state[1].id])

        return model

    def map_pomdp(sparsepomdp: stormpy.storage.SparsePomdp) -> Model:
        """
        Takes a pomdp stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = new_pomdp(create_initial_state=False)

        # we add the states
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
                        model.get_state_by_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
                model.set_choicesmodel.get_state_by_id(state), choice

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsepomdp)

        # we add the valuations
        add_valuations(model, sparsepomdp)

        # we add the observations:
        for state in model.states:
            state.set_observation(model.observation(sparsepomdp.get_observation(state)))

        return model

    def map_ma(sparsema: stormpy.storage.SparseMA) -> Model:
        """
        Takes a ma stormpy representation as input and outputs a simple stormvogel representation
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
                    actionlabel = ",".join(sorted(labels)) if labels else None
                else:
                    actionlabel = str(i)

                action = model.action(actionlabel)
                branch = [
                    (
                        value_to_stormvogel(x.value(), sparsema),
                        model.get_state_by_id(x.column),
                    )
                    for x in row
                ]
                choice[action] = Branches(
                    cast(
                        list[tuple[Value, State]],
                        branch,
                    )
                )
                model.set_choicesmodel.get_state_by_id(state), choice

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        new_reward_model(model, sparsema)

        # we add the valuations
        add_valuations(model, sparsema)

        # we set the correct exit rates
        for state in model.states.items():
            model.set_rate(state[1], sparsema.exit_rates[state[1].id])

        # we set the markovian states
        for state_id in list(sparsema.markovian_states):
            model.add_markovian_state(model.get_state_by_id(state_id))

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
    """Create a model from prism."""

    assert stormpy is not None
    return stormpy_to_stormvogel(stormpy.build_model(prism_code))
