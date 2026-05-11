import warnings

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
) -> stormvogel.result.Result | stormvogel.result.ParetoResult | None:
    """Perform model checking on a stormvogel model using stormpy.

    Convert the model to stormpy, run model checking, and convert the result back.
    If no property string is provided, a widget for building one is displayed.

    For multiobjective properties of the form ``multi(Pmax=? [...], ...)`` a
    :class:`~stormvogel.result.ParetoResult` is returned instead of a
    :class:`~stormvogel.result.Result`.

    :param model: The stormvogel model to check.
    :param prop: A property string to check, or ``None`` to display a property builder widget.
    :param scheduler: Whether to extract a scheduler from the result (ignored for multiobjective).
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

        if prop[0].raw_formula.is_multi_objective_formula:
            stormpy_result = stormpy.model_checking(stormpy_model, prop[0])
            return convert_results.convert_pareto_result(stormpy_result, prop[0])

        if model.is_interval_model():
            if model.model_type == stormvogel.model.ModelType.DTMC:
                warnings.warn(
                    "stormpy.check_interval_mdp is being called on a DTMC interval model. "
                    "At the time of writing, stormy has no check_interval_dtmc, results should be correct but may be inefficient. ",
                    stacklevel=2,
                )
            if not model.has_fixed_graph():
                warnings.warn(
                    "Interval model has transitions with lower bound 0: the graph "
                    "is not fixed and stormpy model checking may be ill-supported.",
                    stacklevel=2,
                )
            task = stormpy.CheckTask(prop[0].raw_formula, True)
            task.set_uncertainty_resolution_mode(
                stormpy.UncertaintyResolutionMode.ROBUST
            )
            stormpy_result = stormpy.check_interval_mdp(
                stormpy_model, task, stormpy.Environment()
            )
        elif model.supports_actions() and scheduler:
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
            "No property string provided. You can create a simple one using this widget."
        )
        stormvogel.property_builder.build_property_string(model)
        return None
