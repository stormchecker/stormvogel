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
    """
    Instead of calling this function, the storm model checker can be used.
    In order to to that, convert this model using the mapping and then do storm model checking.
    After that, map it back.

    This function just performs this procedure automatically.
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
