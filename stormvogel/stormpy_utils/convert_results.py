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
    for state in model.states:
        av_act = state.available_actions()
        stormpy_state_id = model.states.index(state)
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
            model.states[index]: value for (index, value) in enumerate(stormpy_result.get_values())
        }
    elif type(stormpy_result) == stormpy.ExplicitQualitativeCheckResult:
        values = {model.states[i]: stormpy_result.at(i) for i in range(0, len(model.states))}
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
