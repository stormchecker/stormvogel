# import stormvogel.model_checking
import stormvogel.examples.monty_hall
import stormvogel.examples.die
import stormvogel.stormpy_utils.model_checking
import stormvogel.model
import pytest

try:
    import stormpy
except ImportError:
    stormpy = None


def test_model_checking():
    if stormpy is not None:
        # we get our result using the stormvogel model checker function indirectly
        mdp = stormvogel.examples.monty_hall.create_monty_hall_mdp()
        prop = 'Pmax=? [F "done"]'
        result = stormvogel.stormpy_utils.model_checking.model_checking(mdp, prop, True)

        # and directly
        prop = stormpy.parse_properties(prop)

        stormpy_model = stormvogel.stormpy_utils.mapping.stormvogel_to_stormpy(mdp)
        stormpy_result = stormpy.model_checking(
            stormpy_model, prop[0], extract_scheduler=True
        )

        stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(
            stormpy_model
        )

        stormvogel_result = (
            stormvogel.stormpy_utils.convert_results.convert_model_checking_result(
                stormvogel_model, stormpy_result
            )
        )

        # now we do it for a dtmc:
        dtmc = stormvogel.examples.die.create_die_dtmc()
        prop = 'P=? [F "rolled1"]'
        result = stormvogel.stormpy_utils.model_checking.model_checking(
            dtmc, prop, True
        )

        # indirectly:
        prop = stormpy.parse_properties(prop)
        stormpy_model = stormvogel.stormpy_utils.mapping.stormvogel_to_stormpy(dtmc)
        stormpy_result = stormpy.model_checking(stormpy_model, prop[0])

        stormvogel_model = stormvogel.stormpy_utils.mapping.stormpy_to_stormvogel(
            stormpy_model
        )

        stormvogel_result = (
            stormvogel.stormpy_utils.convert_results.convert_model_checking_result(
                stormvogel_model, stormpy_result
            )
        )

        assert result == stormvogel_result

        # we check if it fails in the case of non-stochastic models
        dtmc.remove_state(dtmc.states[1], normalize=False)
        prop = 'P=? [F "rolled2"]'
        with pytest.raises(RuntimeError):
            assert stormvogel.stormpy_utils.model_checking.model_checking(
                dtmc, prop, True
            )
