import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.stormpy_utils.convert_results as convert_results
import stormvogel.model
import stormvogel.property_builder

try:
    import stormpy
except ImportError:
    stormpy = None


def model_checking(
    model: stormvogel.model.Model, prop: str | None = None, scheduler: bool = True
) -> stormvogel.result.Result | None:
    """Perform model checking on a stormvogel model using stormpy.

    Convert the model to stormpy, run model checking, and convert the result back.
    If no property string is provided, a widget for building one is displayed.

    :param model: The stormvogel model to check.
    :param prop: A property string to check, or ``None`` to display a property builder widget.
    :param scheduler: Whether to extract a scheduler from the result.
    :returns: The model checking result, or ``None`` if no property was provided.
    :raises RuntimeError: If the model is not stochastic.
    """

    assert stormpy is not None

    if not model.is_stochastic():
        raise RuntimeError("Model checking only works on stochastic models.")

    # the user must provide a property string, otherwise we provide the widget for building one
    if prop:
        # we first map the model to a stormpy model
        stormpy_model = mapping.stormvogel_to_stormpy(model)

        # we perform the model checking operation
        prop = stormpy.parse_properties(prop)
        assert prop is not None
        if model.supports_actions() and scheduler:
            stormpy_result = stormpy.model_checking(
                stormpy_model, prop[0], extract_scheduler=True
            )
        else:
            stormpy_result = stormpy.model_checking(stormpy_model, prop[0])

        # to get the correct action labels, we need to convert the model back to stormvogel instead of
        # using the initial one for now. (otherwise schedulers won't work)
        stormvogel_model = mapping.stormpy_to_stormvogel(stormpy_model)

        # we convert the results
        assert stormvogel_model is not None
        stormvogel_result = convert_results.convert_model_checking_result(
            stormvogel_model, stormpy_result
        )
        assert stormvogel_result is not None

        # Map back to the original model to preserve UUIDs and references
        stormvogel_result = convert_results.map_result_to_original_model(
            stormvogel_result, model, stormvogel_model
        )
        return stormvogel_result
    else:
        print(
            "You have not proved a property string. You can create a simple one using this widget."
        )
        stormvogel.property_builder.build_property_string(model)
        return None


if __name__ == "__main__":
    import examples.monty_hall

    mdp = examples.monty_hall.create_monty_hall_mdp()

    rewardmodel = mdp.new_reward_model("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))
    rewardmodel2 = mdp.new_reward_model("rewardmodel2")
    rewardmodel2.set_from_rewards_vector(list(range(67)))

    print(model_checking(mdp))
