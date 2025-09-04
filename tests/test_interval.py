import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.model
import pytest

try:
    import stormpy
except ImportError:
    stormpy = None


@pytest.mark.tags("stormpy")
def test_convert_imc():
    imc = stormvogel.model.new_dtmc()
    init = imc.get_initial_state()

    imc.new_state(labels="A")
    imc.new_state(labels="B")

    init.set_choice(
        [
            (
                stormvogel.model.Interval(1 / 3, 2 / 3),
                imc.get_states_with_label("A")[0],
            ),
            (
                stormvogel.model.Interval(1 / 2, 5 / 6),
                imc.get_states_with_label("B")[0],
            ),
        ]
    )

    # we add self loops to all states with no outgoing choices
    imc.add_self_loops()

    # we test the mapping
    stormpy_imc = mapping.stormvogel_to_stormpy(imc)
    new_imc = mapping.stormpy_to_stormvogel(stormpy_imc)

    assert imc == new_imc
