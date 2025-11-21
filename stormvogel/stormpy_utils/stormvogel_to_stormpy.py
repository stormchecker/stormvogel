from typing import Optional, Union, cast

from stormvogel import parametric
from stormvogel.model.action import EmptyAction
from stormvogel.model.distribution import Distribution
from stormvogel.model.model import Model, ModelType
from stormvogel.model.value import Number, Value, Interval

try:
    import stormpy
except ImportError:
    stormpy = None

def convert_polynomial_to_stormpy(
    polynomial: parametric.Polynomial,
) -> "stormpy.pycarl.cln.FactorizedPolynomial":
    """helper function for converting polynomials to pycarl polyomials"""
    assert stormpy is not None

    terms = []
    for exponent, coefficient in polynomial.terms.items():
        if coefficient != 0:
            stormpy_term = stormpy.pycarl.cln.Term(
                stormpy.pycarl.cln.Rational(coefficient)
            )
            assert isinstance(exponent, tuple)
            for index, exp in enumerate(exponent):
                for i in range(exp):
                    stormpy_term *= variables[
                        [str(var) for var in variables].index(
                            polynomial.variables[index]
                        )
                    ]
            terms.append(stormpy_term)
    polynomial = stormpy.pycarl.cln.Polynomial(terms)
    factorized_polynomial = stormpy.pycarl.cln.FactorizedPolynomial(
        polynomial, stormpy.pycarl.cln.factorization_cache
    )
    return factorized_polynomial

def value_to_stormpy(
    value: Value,
    variables: list["stormpy.pycarl.Variable"],
    model: Model,
) -> "stormpy.pycarl.cln.FactorizedRationalFunction | stormpy.pycarl.Interval | Value":
    """converts a stormvogel transition value to a stormpy (pycarl) value"""

    assert stormpy is not None

    if model.is_parametric():
        # we have a special case for numbers as they are not just a specific case of a polynomial in stormvogel
        if isinstance(value, Number):
            rational = stormpy.pycarl.cln.Rational(float(value))
            polynomial = stormpy.pycarl.cln.Polynomial(rational)
            factorized_polynomial = stormpy.pycarl.cln.FactorizedPolynomial(
                polynomial, stormpy.pycarl.cln.factorization_cache
            )
            factorized_rational = stormpy.pycarl.cln.FactorizedRationalFunction(
                factorized_polynomial
            )
        elif isinstance(value, parametric.RationalFunction):
            factorized_numerator = convert_polynomial(value.numerator)
            factorized_denominator = convert_polynomial(value.denominator)

            # TODO gives segmentation fault
            factorized_rational = stormpy.pycarl.cln.FactorizedRationalFunction(
                factorized_numerator, factorized_denominator
            )
        else:
            assert isinstance(value, parametric.Polynomial)
            factorized_rational = stormpy.pycarl.cln.FactorizedRationalFunction(
                convert_polynomial(value)
            )

        return factorized_rational
    elif model.is_interval_model():
        # in the case of interval models, we convert intervals, and regular values are converted
        # to intervals where the lower and upper value are the same
        if isinstance(value, Interval):
            interval = stormpy.pycarl.Interval(value[0], value[1])
        else:
            interval = stormpy.pycarl.Interval(value, value)

        return interval
    else:
        return value

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

def build_choice_labeling(model: Model):
    """
    Takes a model and creates a stormpy choice labelling object.
    """
    labels: set[str] = set()
    for action in model.actions:
        if action.label is not None:
            labels.add(action.label)

    choice_count = sum(len(choices.choices) for choices in model.choices)

    # we add the labels to the choice labeling object
    choice_labeling = stormpy.storage.ChoiceLabeling(choice_count)
    for label in labels:
        choice_labeling.add_label(label)

    return choice_labeling

def build_state_labeling(model: Model) -> stormpy.storage.StateLabeling:
    """
    Takes a model and creates a state labelling object that determines which states get which labels in the stormpy representation
    """
    assert stormpy is not None

    # we first initialize all labels
    state_labeling = stormpy.storage.StateLabeling(len(model.states.keys()))
    for label in model.state_labels:
        state_labeling.add_label(label)

    # then we assign the labels to the correct states
    for state in model:
        for label in state.labels:
            state_labeling.add_label_to_state(label, model.stormpy_id[state])

    return state_labeling

def build_reward_models(
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

def build_state_valuations(model: Model) -> stormpy.storage.StateValuation:
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

def build_observations(model: Model) -> list[int]:
    """
    Builds the observation mapping for POMDPs
    """
    observations = []
    for state in model.states:
        if isinstance(state.observation, Distribution):
            raise NotImplementedError(
                "Strompy does not support probabilistic observations in POMDPs."
            )
        
        if model.observations is None or model.observation not in model.observations or state.observation is None:
            raise RuntimeError(
                f"State {state} has an observation {state.observation} that is not in the model's observations."
            )
        
        observations.append(model.observations.index(state.observation))

    return observations

def build_markovian_states_bitvector(model: Model) -> stormpy.BitVector:
    assert stormpy is not None

    if model.markovian_states is None:
        raise RuntimeError("Model does not have markovian states defined.")
    
    markovian_state_ids = [state.stormpy_id for state in model.markovian_states]
    if isinstance(markovian_state_ids, list):
        markovian_states_bitvector = stormpy.storage.BitVector(
            max(markovian_state_ids) + 1,
            markovian_state_ids,
        )
    else:
        markovian_states_bitvector = stormpy.storage.BitVector(0)

    return markovian_states_bitvector

def build_dtmc(model: Model, matrix, state_labeling, reward_models, state_valuations):
    assert stormpy is not None

    if model.is_parametric():
        components = stormpy.SparseParametricModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        dtmc = stormpy.storage.SparseParametricDtmc(components)
    elif model.is_interval_model():
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        dtmc = stormpy.storage.SparseIntervalDtmc(components)
    else:
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        dtmc = stormpy.storage.SparseDtmc(components)

    return dtmc

def build_mdp(model: Model, matrix, choice_labeling, state_labeling, reward_models, state_valuations):
    assert stormpy is not None

    if model.is_parametric():
        components = stormpy.SparseParametricModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseParametricMdp(components)
    elif model.is_interval_model():
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseIntervalMdp(components)
    else:
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseMdp(components)

    return mdp

def build_ctmc(model: Model, matrix, state_labeling, reward_models, state_valuations):
    assert stormpy is not None

    if model.is_parametric():
        components = stormpy.SparseParametricModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            rate_choices=True,
        )
        components.state_valuations = state_valuations
        ctmc = stormpy.storage.SparseParametricCtmc(components)
    elif model.is_interval_model():
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            rate_transitions=True,
        )
        components.state_valuations = state_valuations
        ctmc = stormpy.storage.SparseIntervalCtmc(components)
    else:
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            rate_transitions=True,
        )
        components.state_valuations = state_valuations
        ctmc = stormpy.storage.SparseCtmc(components)

    return ctmc

def build_pomdp(model: Model, matrix, choice_labeling, state_labeling, observations, reward_models, state_valuations):
    assert stormpy is not None

    if model.is_parametric():
        components = stormpy.SparseParametricModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.observability_classes = observations
        components.choice_labeling = choice_labeling
        pomdp = stormpy.storage.SparseParametricPomdp(components)
    elif model.is_interval_model():
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.observability_classes = observations
        components.choice_labeling = choice_labeling
        pomdp = stormpy.storage.SparseIntervalPomdp(components)
    else:
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.state_valuations = state_valuations
        components.observability_classes = observations
        components.choice_labeling = choice_labeling
        pomdp = stormpy.storage.SparsePomdp(components)

    return pomdp

def build_ma(model: Model, matrix, choice_labeling, state_labeling, markovian_states_bitvector, reward_models, state_valuations):
    assert stormpy is not None
    
    if model.is_parametric():
        components = stormpy.SparseParametricModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            markovian_states=markovian_states_bitvector,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        ma = stormpy.storage.SparseParametricMA(components)
    elif model.is_interval_model():
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            markovian_states=markovian_states_bitvector,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        ma = stormpy.storage.SparseIntervalMA(components)
    else:
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            markovian_states=markovian_states_bitvector,
        )
        components.state_valuations = state_valuations
        components.choice_labeling = choice_labeling
        ma = stormpy.storage.SparseMA(components)

    return ma

def stormvogel_to_stormpy(
    model: Model,
) -> stormpy.model.SparseModel:
    assert stormpy is not None

    # we throw the neccessary errors first
    if not model.all_states_outgoing_transition():
        raise RuntimeError(
            "This model has states with no outgoing transitions.\nUse the add_self_loops() function to add self loops to all states with no outgoing transition."
        )

    if model.unassigned_variables():
        raise RuntimeError("Each state should have a value for each variable")

    if model.has_zero_transition() and not model.supports_rates():
        raise RuntimeError(
            "This model has transitions with probability=0. Stormpy assumes that these do not explicitly exist."
        )

    # we store the pycarl parameters of a model
    stormpy.pycarl.clear_variable_pool()
    variables = []
    for p in model.get_parameters():
        var = stormpy.pycarl.Variable(p)
        variables.append(var)

    if model.supports_actions():
        choice_labeling = build_choice_labeling(model)
    else:
        choice_labeling = None

    if model.supports_observations():
        observations = build_observations(model)
    else:
        observations = None

    matrix = build_matrix(model, choice_labeling)
    state_labeling = build_state_labeling(model)
    reward_models = build_reward_models(model)
    state_valuations = build_state_valuations(model)

    # we check the type to handle the model correctly
    if model.type == ModelType.DTMC:
        return build_dtmc(model, matrix, state_labeling, reward_models, state_valuations)
    elif model.type == ModelType.MDP:
        return build_mdp(
            model,
            matrix,
            choice_labeling,
            state_labeling,
            reward_models,
            state_valuations,
        )
    elif model.type == ModelType.CTMC:
        return build_ctmc(model, matrix, state_labeling, reward_models, state_valuations)
    elif model.type == ModelType.POMDP:
        return build_pomdp(
            model,
            matrix,
            choice_labeling,
            state_labeling,
            observations,
            reward_models,
            state_valuations,
        )
    elif model.type == ModelType.MA:
        markovian_states_bitvector = build_markovian_states_bitvector(model)
        pass
    else:
        raise NotImplementedError(
            "Converting this type of model to stormpy is not yet supported"
        )
