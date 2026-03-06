import stormvogel.model
import stormvogel.result
from typing import Union

try:
    import stormpy
except ImportError:
    stormpy = None


def convert_scheduler_to_stormvogel(
    model: stormvogel.model.Model, stormpy_scheduler: "stormpy.storage.Scheduler"
):
    """Converts a stormpy scheduler to a stormvogel scheduler"""
    taken_actions = {}
    for stormpy_state_id, state in enumerate(model.states):
        av_act = state.available_actions()
        choice = stormpy_scheduler.get_choice(stormpy_state_id)
        action_index = choice.get_deterministic_choice()
        taken_actions[state] = av_act[action_index]
    return stormvogel.result.Scheduler(model, taken_actions)


def convert_model_checking_result(
    model: stormvogel.model.Model,
    stormpy_result: Union[
        "stormpy.ExplicitQuantitativeCheckResult",
        "stormpy.ExplicitQualitativeCheckResult",
        "stormpy.ExplicitParametricQuantitativeCheckResult",
    ],
    with_scheduler: bool = True,
) -> stormvogel.result.Result | None:
    """
    Takes a model checking result from stormpy and its associated model and converts it to a stormvogel representation
    """
    assert stormpy is not None

    # we distinguish between quantitative and qualitative results
    # (determines what kind of values our result contains)
    if (
        type(stormpy_result) == stormpy.ExplicitQuantitativeCheckResult
        or type(stormpy_result) == stormpy.ExplicitParametricQuantitativeCheckResult
    ):
        values = {
            model.states[index]: value
            for (index, value) in enumerate(stormpy_result.get_values())
        }
    elif type(stormpy_result) == stormpy.ExplicitQualitativeCheckResult:
        values = {
            model.states[i]: stormpy_result.at(i) for i in range(0, len(model.states))
        }
    else:
        raise RuntimeError("Unsupported result type")

    # we check if our results and expected converted results come with a scheduler
    if stormpy_result.has_scheduler and with_scheduler:
        # we build the results object containing a converted scheduler
        stormvogel_result = stormvogel.result.Result(
            model,
            values,
            scheduler=convert_scheduler_to_stormvogel(model, stormpy_result.scheduler),
        )
    else:
        # we build the results object without a scheduler
        stormvogel_result = stormvogel.result.Result(
            model,
            values,
        )

    return stormvogel_result


def map_result_to_original_model(
    result: stormvogel.result.Result,
    original_model: stormvogel.model.Model,
    recreated_model: stormvogel.model.Model,
) -> stormvogel.result.Result:
    """
    Maps a model checking result (which uses reconstructed states) back to the
    original model states to preserve object identities.
    """
    if len(original_model.states) != len(recreated_model.states):
        return result

    mapped_values = {}
    for index, state in enumerate(original_model.states):
        eq_state = recreated_model.states[index]
        mapped_values[state] = result.values[eq_state]

    mapped_scheduler = None
    if result.scheduler is not None:
        mapped_taken_actions = {}
        for index, state in enumerate(original_model.states):
            eq_state = recreated_model.states[index]
            eq_action = result.scheduler.taken_actions[eq_state]
            # Map action label and branch counts, fallback if action structure changes
            mapped_action = None
            for a in state.available_actions():
                if a.label == eq_action.label:
                    mapped_action = a
                    break
            if mapped_action is None and len(state.available_actions()) > 0:
                # Fallback: Just take the same index if order didn't change
                try:
                    act_idx = eq_state.available_actions().index(eq_action)
                    mapped_action = state.available_actions()[act_idx]
                except ValueError:
                    mapped_action = state.available_actions()[0]
            if mapped_action is not None:
                mapped_taken_actions[state] = mapped_action

        mapped_scheduler = stormvogel.result.Scheduler(
            original_model, mapped_taken_actions
        )

    return stormvogel.result.Result(original_model, mapped_values, mapped_scheduler)
