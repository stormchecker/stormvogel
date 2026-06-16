from __future__ import annotations

import logging
from fractions import Fraction
from typing import TYPE_CHECKING

try:
    import umbi.ats
    import umbi.datatypes
    from umbi.ats.simple_ats import SimpleAts
    from umbi.ats.entity_space import EntityClass

    _UMBI_AVAILABLE = True
except ImportError:
    _UMBI_AVAILABLE = False
    if TYPE_CHECKING:
        import umbi.ats
        import umbi.datatypes
        from umbi.ats.simple_ats import SimpleAts
        from umbi.ats.entity_space import EntityClass

from stormvogel.model.model import Model, ModelType, new_model
from stormvogel.model.action import Action, EmptyAction
from stormvogel.model.distribution import Distribution
from stormvogel.model.observation import Observation
from stormvogel.model.value import Interval
from stormvogel.model.variable import (
    Variable as SvVariable,
    Predicate,
    VariableKey,
    IntDomain,
    BoolDomain,
    CategoricalDomain,
)

logger = logging.getLogger("stormvogel.umbi")


def none_to_empty_string(x: str | None) -> str:
    return x if x is not None else ""


def empty_string_to_none(x: str) -> str | None:
    return x if x != "" else None


def to_nr_players(model_type: ModelType) -> int:
    if model_type in (ModelType.DTMC, ModelType.CTMC, ModelType.HMM):
        return 0
    return 1


def to_time_type(model_type: ModelType) -> umbi.ats.TimeType:
    if model_type in (ModelType.CTMC, ModelType.MA):
        return umbi.ats.TimeType.STOCHASTIC
    return umbi.ats.TimeType.DISCRETE


def to_umbi_interval(interval: Interval) -> umbi.datatypes.Interval:
    return umbi.datatypes.Interval(left=interval.lower, right=interval.upper)


def from_umbi_interval(interval: umbi.datatypes.Interval) -> Interval:
    return Interval(lower=interval.left, upper=interval.right)


def _domain_from_umbi_var(
    v: umbi.ats.Variable,
) -> IntDomain | BoolDomain | CategoricalDomain | None:
    """Infer a stormvogel domain from a UMBI variable's observed value set."""
    if not v.has_domain:
        return None
    from umbi.datatypes.primitive import PrimitiveType
    from umbi.datatypes.numeric_primitive import NumericPrimitiveType

    pt = v.promotion_type
    if pt == PrimitiveType.BOOL:
        return BoolDomain()
    if pt in (NumericPrimitiveType.INT, NumericPrimitiveType.UINT):
        return IntDomain(int(v.lower), int(v.upper))  # type: ignore[arg-type]
    return CategoricalDomain(tuple(v.domain.sorted_domain))


def get_model_type(
    time: umbi.ats.TimeType, players: int, has_observations: bool
) -> ModelType:
    if time == umbi.ats.TimeType.DISCRETE:
        if players == 0:
            if has_observations:
                return ModelType.HMM
            return ModelType.DTMC
        if players == 1:
            if has_observations:
                return ModelType.POMDP
            return ModelType.MDP
    elif time == umbi.ats.TimeType.STOCHASTIC:
        if players == 0:
            if has_observations:
                raise ValueError("Stormvogel does not support CTMCs with observations")
            return ModelType.CTMC
    raise NotImplementedError(
        f"The combination {time}, {players}, {has_observations} is not supported."
    )


def translate_to_umbi(
    model: Model, ignore_unsupported_rewards: bool = False
) -> SimpleAts:
    """Export a stormvogel Model to a UMBI SimpleAts.

    Args:
        model: a stormvogel Model (DTMC, CTMC, MDP, or POMDP)
        ignore_unsupported_rewards: if True, silently skip transition reward models
            instead of raising an error.

    Returns: a umbi.ats.SimpleAts
    """
    if not _UMBI_AVAILABLE:
        raise ImportError(
            "umbi is required for UMBI translation. Install it with: pip install umbi"
        )
    if model.is_parametric():
        raise ValueError(
            "Parametric models cannot be translated to UMBI. "
            "Evaluate all parameters before calling translate_to_umbi."
        )
    if model.model_type == ModelType.MA:
        raise NotImplementedError(
            "Markov automata are not yet supported by the UMBI translator."
        )

    ats = SimpleAts()
    ats.time = to_time_type(model.model_type)
    ats.num_players = to_nr_players(model.model_type)

    ats.new_states(model.nr_states)
    state_to_id: dict = {s: i for i, s in enumerate(model.states)}

    ats.initial_states = [state_to_id[model.initial_state]]

    # CTMC exit rates: sum of all outgoing rates per state
    if model.supports_rates():
        for state in model.states:
            s_id = state_to_id[state]
            if state.has_choices():
                exit_rate = sum(
                    rate for _, branch in state.choices for rate, _ in branch
                )
            else:
                exit_rate = 0.0
            ats.state_to_exit_rate[s_id] = exit_rate

    # MDP/POMDP choice actions
    action_to_ca: dict[Action, int] = {}
    if model.supports_actions():
        all_actions = list(model.actions)
        ats.num_choice_actions = len(all_actions)
        ats.new_choice_action_to_name()
        for ca_id, action in enumerate(all_actions):
            action_to_ca[action] = ca_id
            ats.choice_action_to_name[ca_id] = none_to_empty_string(action.label)

    # Transitions
    for state in model.states:
        s_id = state_to_id[state]
        if not state.has_choices():
            continue
        exit_rate = ats.state_to_exit_rate[s_id] if model.supports_rates() else None
        for action, branch in state.choices:
            choice = ats.new_state_choice(s_id)
            if action_to_ca:
                ats.choice_to_choice_action[choice] = action_to_ca[action]
            for value, target in branch:
                t_id = state_to_id[target]
                if isinstance(value, Interval):
                    prob = to_umbi_interval(value)
                elif exit_rate is not None and exit_rate > 0:
                    prob = value / exit_rate
                else:
                    prob = value
                ats.new_choice_branch(choice, t_id, prob)

    # AP labels
    for label, labeled_states in model.state_labels.items():
        ann = ats.new_ap_annotation(name=label)
        ids = {state_to_id[s] for s in labeled_states}
        ann.state_values = [i in ids for i in range(ats.num_states)]

    # Observations (POMDP) — build obs_to_id here so valuations section can use it
    obs_to_id: dict[object, int] = {}
    if model.supports_observations():
        state_obs_ids: list[int] = []
        for state in model.states:
            obs = model.state_observations.get(state)
            if obs is None:
                raise RuntimeError(
                    f"State {state} has no observation in an observation model."
                )
            if isinstance(obs, Distribution):
                raise NotImplementedError(
                    f"State {state} has a distribution over observations, which UMBI does not support. "
                )
            if obs not in obs_to_id:
                obs_to_id[obs] = len(obs_to_id)
            state_obs_ids.append(obs_to_id[obs])
        ats.num_observations = len(obs_to_id)
        ats.observation_annotation.state_values = state_obs_ids

    # State rewards (transition rewards are not supported)
    for reward_model in model.rewards:
        if reward_model.has_transition_rewards():
            if ignore_unsupported_rewards:
                continue
            raise ValueError(
                f"Reward model '{reward_model.name}' has transition rewards, which are not "
                f"supported. Use ignore_unsupported_rewards=True to skip."
            )
        ann = ats.new_reward_annotation(name=reward_model.name)
        ann.add_state_values()
        for i, state in enumerate(model.states):
            r = reward_model.get_state_reward(state)
            ann.state_values[i] = r if r is not None else 0

    # State and observation valuations share a single EntityClassValuations container
    has_state_vals = any(model.state_valuations.values())
    obs_valuations: dict[Observation, dict[VariableKey, object]] = (
        {obs: v for obs, v in model.observation_valuations.items() if v}
        if model.supports_observations()
        else {}
    )
    has_obs_vals = bool(obs_valuations)

    if has_state_vals or has_obs_vals:
        var_valuations = ats.add_variable_valuations()

        if has_state_vals:
            sv_ent = var_valuations.add_state_valuations()
            all_sv_vars: list[SvVariable] = sorted(
                {var for vals in model.state_valuations.values() for var in vals},
                key=lambda v: v.label,
            )
            sv_var_to_umbi: dict[SvVariable, umbi.ats.Variable] = {
                var: sv_ent.new_variable(var.label) for var in all_sv_vars
            }
            for i, state in enumerate(model.states):
                if model.state_valuations[state]:
                    sv_ent.set_entity_valuation(
                        i,
                        {
                            sv_var_to_umbi[var]: val
                            for var, val in model.state_valuations[state].items()
                        },
                    )

        if has_obs_vals:
            ov_ent = var_valuations.add_observation_valuations()
            all_ov_vars: list[VariableKey] = sorted(
                {
                    var
                    for vals in obs_valuations.values()
                    for var in vals
                    if isinstance(var, (SvVariable, Predicate))
                },
                key=lambda v: v.label,
            )
            ov_var_to_umbi: dict[VariableKey, umbi.ats.Variable] = {
                var: ov_ent.new_variable(var.label) for var in all_ov_vars
            }
            for obs, vals in obs_valuations.items():
                if obs not in obs_to_id:
                    logger.warning(
                        "Observation with valuations is not assigned to any state and will be skipped."
                    )
                    continue
                sv_vals = {
                    var: val
                    for var, val in vals.items()
                    if isinstance(var, (SvVariable, Predicate))
                }
                if sv_vals:
                    ov_ent.set_entity_valuation(
                        obs_to_id[obs],
                        {  # type: ignore[arg-type]
                            ov_var_to_umbi[var]: val for var, val in sv_vals.items()
                        },
                    )

    ats.validate()
    return ats


def translate_to_stormvogel(
    ats: SimpleAts,
    ignore_unsupported_rewards: bool = False,
    ignore_choice_annotations: bool = False,
    ignore_branch_annotations: bool = False,
) -> Model:
    """Import a UMBI SimpleAts as a stormvogel Model.

    Args:
        ats: a umbi.ats.SimpleAts
        ignore_unsupported_rewards: if True, silently skip reward annotations that have
            only choice- or branch-level values (not supported by stormvogel state rewards)
            instead of raising an error.
        ignore_choice_annotations: if True, silently skip choice-level variable valuations
            instead of raising an error.
        ignore_branch_annotations: if True, silently skip branch-level variable valuations
            instead of raising an error.

    Returns: a stormvogel Model
    """
    if not _UMBI_AVAILABLE:
        raise ImportError(
            "umbi is required for UMBI translation. Install it with: pip install umbi"
        )
    if ats.time == umbi.ats.TimeType.URGENT_STOCHASTIC:
        raise NotImplementedError(
            "Markov automata are not yet supported by the UMBI translator."
        )
    model_type = get_model_type(
        ats.time, ats.num_players, ats.has_observation_annotation
    )
    model = new_model(model_type, create_initial_state=False)

    # Create stormvogel Observations first, so they can be passed to new_state
    umbi_obs_to_sv_obs: dict[int, Observation] = {}
    obs_values: list[int] = []
    if ats.has_observation_annotation:
        obs_values = [int(v) for v in ats.observation_annotation.state_values]  # type: ignore[arg-type]
        for obs_id in obs_values:
            if obs_id not in umbi_obs_to_sv_obs:
                umbi_obs_to_sv_obs[obs_id] = model.new_observation(alias=str(obs_id))

    for s_id in range(ats.num_states):
        obs: Observation | None = (
            umbi_obs_to_sv_obs[obs_values[s_id]]
            if ats.has_observation_annotation
            else None
        )
        model.new_state(observation=obs)

    if len(ats.initial_states) != 1:
        raise RuntimeError(
            f"Stormvogel supports models with exactly one initial state, "
            f"but this ATS has {len(ats.initial_states)}."
        )
    model.states[ats.initial_states[0]].add_label("init")

    # AP labels (skip "init", handled above)
    for name, ann in ats.ap_annotations.items():
        if name == "init":
            continue
        if not ann.has_state_values:
            continue
        for s_id in range(ats.num_states):
            if ann.state_values[s_id]:
                model.states[s_id].add_label(name)

    # Build action map for MDPs
    ca_to_action: dict[int, Action] = {}
    if ats.has_choice_actions:
        for ca_id in range(ats.num_choice_actions):
            label = (
                empty_string_to_none(ats.choice_action_to_name[ca_id])
                if ats.has_choice_action_to_name
                else None
            )
            ca_to_action[ca_id] = Action(label)

    # Exit rates for CTMCs
    exit_rates = None
    if model.supports_rates():
        if not ats.has_state_to_exit_rate:
            raise RuntimeError("Translating a CTMC requires state exit rates.")
        exit_rates = ats.state_to_exit_rate

    # Transitions
    for s_id in range(ats.num_states):
        state = model.states[s_id]
        choices_shorthand: dict[Action, list] = {}
        state_choice_ids = list(ats.get_state_choices(s_id))
        if exit_rates is not None and state_choice_ids:
            raw_exit_rate = exit_rates[s_id]
            if isinstance(raw_exit_rate, umbi.datatypes.Interval):
                raise RuntimeError(
                    f"CTMC state {s_id} has an interval-valued exit rate, which is not supported."
                )
            s_exit_rate: int | float | Fraction | None = raw_exit_rate
            if s_exit_rate == 0:
                raise RuntimeError(
                    f"CTMC state {s_id} has transitions but exit rate is zero."
                )
        else:
            s_exit_rate = None
        for c_id in state_choice_ids:
            action = (
                ca_to_action[ats.choice_to_choice_action[c_id]]
                if ats.has_choice_to_choice_action
                else EmptyAction
            )
            branches = []
            for b_id in ats.get_choice_branches(c_id):
                raw_prob = ats.branch_to_probability[b_id]
                target = model.states[ats.branch_to_target[b_id]]
                if isinstance(raw_prob, umbi.datatypes.Interval):
                    sv_interval = from_umbi_interval(raw_prob)
                    if s_exit_rate is not None:
                        value = Interval(
                            lower=sv_interval.lower * s_exit_rate,
                            upper=sv_interval.upper * s_exit_rate,
                        )
                    else:
                        value = sv_interval
                elif isinstance(raw_prob, (int, float, Fraction)):
                    if s_exit_rate is not None:
                        value = raw_prob * s_exit_rate
                    else:
                        value = raw_prob
                else:
                    value = raw_prob
                branches.append((value, target))
            choices_shorthand[action] = branches
        if choices_shorthand:
            state.set_choices(choices_shorthand)

    # State rewards
    for name, ann in ats.reward_annotations.items():
        if not ann.has_state_values:
            if ann.has_choice_values or ann.has_branch_values:
                if ignore_unsupported_rewards:
                    continue
                raise ValueError(
                    f"Reward annotation '{name}' has choice- or branch-level values, which "
                    f"stormvogel does not support. Use ignore_unsupported_rewards=True to skip."
                )
            continue
        reward_model = model.new_reward_model(name)
        for s_id in range(ats.num_states):
            val = ann.state_values[s_id]
            if isinstance(val, (int, float, Fraction)) and val != 0:
                reward_model.set_state_reward(model.states[s_id], val)

    # State and observation valuations
    if ats.has_variable_valuations:
        vv = ats.variable_valuations

        if vv.has_values_for(EntityClass.CHOICES):
            if not ignore_choice_annotations:
                raise ValueError(
                    "This ATS has choice-level variable valuations, which stormvogel does not "
                    "support. Use ignore_choice_annotations=True to skip."
                )

        if vv.has_values_for(EntityClass.BRANCHES):
            if not ignore_branch_annotations:
                raise ValueError(
                    "This ATS has branch-level variable valuations, which stormvogel does not "
                    "support. Use ignore_branch_annotations=True to skip."
                )

        if vv.has_values_for(EntityClass.STATES):
            sv = vv.state_valuations
            sv_vars = {
                v.name: SvVariable(label=v.name, domain=_domain_from_umbi_var(v))
                for v in sv.variables
            }
            for s_id in range(ats.num_states):
                umbi_val = sv.get_entity_valuation(s_id)
                if umbi_val:
                    model.state_valuations[model.states[s_id]] = {
                        sv_vars[v.name]: val
                        for v, val in umbi_val.items()
                        if val is not None
                    }

        if vv.has_values_for(EntityClass.OBSERVATIONS):
            ov = vv.observation_valuations
            ov_vars = {
                v.name: SvVariable(label=v.name, domain=_domain_from_umbi_var(v))
                for v in ov.variables
            }
            for obs_id, sv_obs in umbi_obs_to_sv_obs.items():
                umbi_val = ov.get_entity_valuation(obs_id)
                if umbi_val:
                    model.observation_valuations[sv_obs] = {
                        ov_vars[v.name]: val
                        for v, val in umbi_val.items()
                        if val is not None
                    }

    return model
