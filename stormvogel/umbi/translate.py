import logging
import umbi
import umbi.ats
import stormvogel

# TODO support reward models
# TODO support observations
# TODO support Markov automata.

logger = logging.getLogger("stormvogel.translate")

def none_to_empty_string(x: str | None) -> str:
    return x if x is not None else ""


def emtpy_string_to_none(x: str) -> str | None:
    return x if x != "" else None


def to_nr_players(model_type: stormvogel.ModelType) -> int:
    if model_type in [stormvogel.ModelType.DTMC, stormvogel.ModelType.CTMC]:
        return 0
    else:
        return 1

def to_umbi_interval(interval: stormvogel.Interval) -> umbi.datatypes.Interval:
    # TODO update once stormvogel interval is updated.
    return umbi.datatypes.Interval(left=interval.bottom, right=interval.top)

def to_time_type(model_type: stormvogel.ModelType) -> umbi.ats.TimeType:
    if model_type in [stormvogel.ModelType.CTMC, stormvogel.ModelType.MA]:
        return umbi.ats.TimeType.STOCHASTIC
    else:
        return umbi.ats.TimeType.DISCRETE


def translate_to_umbi(model: stormvogel.Model) -> umbi.ats.ExplicitAts:
    """
    Create an Annotated Transition System in the UMBI library.

    Args:
        model: a stormvogel.Model

    Returns: the umbi.ats.ExplicitAts
    """
    ats = umbi.ats.ExplicitAts()
    ats.time = to_time_type(model.type)
    ats.num_players = to_nr_players(model.type)
    ats.num_initial_states = 1  # Note this is a constant for stormvogel currently.
    ats.num_states = model.nr_states
    ats.state_is_initial = [s.is_initial() for s in model.states.values()]
    # TODO do we already enforce that state ids are consecutive.
    if model.type == stormvogel.ModelType.MDP:
        assert model.actions is not None, "MDPs must have actions."
        actions_to_ids = {a: state_id for state_id, a in enumerate(model.actions)}
        ats.num_actions = len(model.actions)  # TODO change once stormvogel is updated
        ats.action_strings = [
            none_to_empty_string(a.label) for a in model.actions
        ]
    else:
        actions_to_ids = {}
    ats.num_choices = model.nr_choices
    if ats.num_players > 0:
        ats.state_to_choice = []
        ats.choice_to_action = []
    ats.choice_to_branch = []
    ats.branch_to_target = []
    ats.branch_probabilities = []
    has_interval = False
    # TODO support exact
    if model.supports_rates():
        ats.state_is_markovian: [True] * model.nr_states
        ats.state_exit_rate = []
    for state in model.states.values():
        if ats.num_players > 0:
            assert ats.state_to_choice is not None, "If players exist, states must have choices."
            ats.state_to_choice.append(len(ats.choice_to_action))
        if ats.state_exit_rate is not None:
            # Model has rates.
            ats.state_exit_rate.append(model.get_rate(state))
            assert model.get_rate(state) > 0 or len(model.get_successor_states(state)) == 0, f"States need positive exit rates, but state {state.id} has exit rate {model.get_rate(state)}"
        for action, choice in state.get_choices():
            if ats.num_players > 0:
                ats.choice_to_action.append(actions_to_ids[action])
            ats.choice_to_branch.append(len(ats.branch_to_target))
            for branch in choice:
                value, target = branch  # Unpacking.
                if type(value) == stormvogel.Interval:
                    value = to_umbi_interval(value)
                print(value)
                ats.branch_to_target.append(target.id)
                if ats.state_exit_rate is not None:
                    # Stormvogel stores rates, UMB stores probabilities.
                    ats.branch_probabilities.append(value/ats.state_exit_rate[-1])
                else:
                    ats.branch_probabilities.append(value)
    if ats.num_players > 0:
        assert ats.state_to_choice is not None, "If players exist, states must have choices."
        ats.state_to_choice.append(len(ats.choice_to_action))
    ats.choice_to_branch.append(len(ats.branch_to_target))
    ats.num_branches = len(ats.branch_to_target)

    for label in model.get_labels():
        labeled_states = {s.id for s in model.get_states_with_label(label)}
        ats.add_ap_annotation(umbi.ats.AtomicPropositionAnnotation(
            name=label,
            alias=label,
            description=label,
            state_to_value=[True if s_id in labeled_states else False for s_id in range(ats.num_states)]
        ))

    # Now create reward structures:
    # for reward_model in model.rewards:
    #
    #     reward_model.
    # umbi.ats.RewardAnnotation(
    #     name="steps",
    #     type=umbi.ats.CommonType.DOUBLE,
    #     alias="step cost",
    #     description="Cost incurred at each step.",
    #     choice_to_value=[1 for c in range(ats.num_choices)]
    # )
    # ats.add_reward_annotation()

    ats.validate()
    return ats


def get_model_type(
    time: umbi.ats.TimeType, players: int, has_observations: bool
) -> stormvogel.ModelType:
    if time == umbi.ats.TimeType.DISCRETE:
        if players == 0:
            if has_observations:
                raise ValueError("Stormvogel does not support MCs with observations")
            return stormvogel.ModelType.DTMC
        if players == 1:
            if has_observations:
                return stormvogel.ModelType.POMDP
            else:
                return stormvogel.ModelType.MDP
    elif time == umbi.ats.TimeType.STOCHASTIC:
        if players == 0:
            if has_observations:
                raise ValueError("Stormvogel does not support CTMCs with observations")
            return stormvogel.ModelType.CTMC
    raise NotImplementedError(
        f"The combination {time}, {players}, {has_observations} is not supported by this translation."
    )

def translate_to_stormvogel(ats: umbi.ats.ExplicitAts) -> stormvogel.Model:
    modeltype = get_model_type(ats.time, ats.num_players, False)
    logger.debug(f"modeltype: {modeltype}")
    model = stormvogel.new_model(
        modeltype=modeltype,
        create_initial_state=False
    )
    # TODO handle the case where action strings do not exist.
    if model.supports_rates() and ats.state_exit_rate is None:
        raise RuntimeError("Translating a CTMC requires state-exit-rates.")
    stormvogel_actions = {}
    if ats.action_strings is not None:
        stormvogel_actions = {
            action_id: model.new_action(emtpy_string_to_none(action))
            for action_id, action in enumerate(ats.action_strings)
        }
    initial_state_found = False
    for s_id in range(ats.num_states):
        state = model.new_state()
        if ats.state_is_initial[s_id]:
            if initial_state_found:
                raise RuntimeError(
                    "Stormvogel supports models with at most one initial state."
                )
            initial_state_found = True
            state.add_label("init")
    for name, ap_structure in ats.ap_annotations.items():
        if name == "init":
            # We handle the init label separately.
            continue
        if ap_structure.has_choice_values:
            raise RuntimeError("Choice labels are not supported by Stormvogel.")
        if ap_structure.has_branch_values:
            raise RuntimeError("Branch labels are not supported by Stormvogel.")

        for s_id in range(ats.num_states):
            if ap_structure.get_state_value(s_id):
                model.get_state_by_id(s_id).add_label(ap_structure.name)
            # We currently ignore the description and the alias.
    if not initial_state_found:
        raise RuntimeError("Stormvogel requires models to have an initial state.")

    for s_id in range(ats.num_states):
        state = model.get_state_by_id(s_id)
        state_choices = {}
        for c_id in ats.state_choice_range(s_id):
            choice_branches = []
            for b_id in ats.choice_branch_range(c_id):
                branch_value = ats.get_branch_probability(b_id)
                if model.type == stormvogel.ModelType.CTMC:
                    # Note that in stormvogel, CTMCs are represented with the transition rates.
                    branch_value *= ats.state_exit_rate[s_id]
                print(branch_value)
                choice_branches.append(
                    (branch_value, ats.get_branch_target(b_id))
                )
            state_choices[
                stormvogel_actions.get(ats.get_choice_action(c_id)) if ats.choice_to_action is not None else stormvogel.EmptyAction
            ] = choice_branches
        model.add_choice(state, state_choices)
    return model
