# ruff: noqa: F405
"""The stormvogel package"""

__all__ = [
    # layout
    "Layout",
    # model
    "Action",
    "EmptyAction",
    "Choices",
    "ChoicesShorthand",
    "choices_from_shorthand",
    "Distribution",
    "ModelType",
    "Model",
    "new_dtmc",
    "new_mdp",
    "new_ctmc",
    "new_pomdp",
    "new_hmm",
    "new_ma",
    "new_model",
    "State",
    "Number",
    "Interval",
    "Value",
    "is_zero",
    "value_to_string",
    "Observation",
    "RewardModel",
    "IntDomain",
    "BoolDomain",
    "CategoricalDomain",
    "VariableDomain",
    "Variable",
    "Predicate",
    "VariableKey",
    # property_builder
    "build_property_string",
    # result
    "Scheduler",
    "random_scheduler",
    "Result",
    "ParetoResult",
    "plot_pareto_result",
    # show
    "show",
    "show_bird",
    # simulator
    "Path",
    "step",
    "simulate_path",
    "simulate",
    "get_action_at_state",
    # visualization
    "JSVisualization",
    # stormpy_utils
    "model_checking",
    # utilities
    "is_in_notebook",
    # submodules (keep accessible after `from stormvogel import *`)
    "layout",
    "bird",
    "examples",
    "extensions",
    "stormpy_utils",
]

from stormvogel import layout  # NOQA
from stormvogel.layout import Layout  # NOQA

# from stormvogel.stormpy_utils.mapping import *  # NOQA
# from stormvogel.stormpy_utils.model_checking import model_checking  # NOQA
from stormvogel.model import *  # NOQA
from stormvogel.property_builder import build_property_string  # NOQA
from stormvogel.result import *  # NOQA
from stormvogel.show import *  # NOQA
from stormvogel.show import show  # NOQA
from stormvogel.simulator import *  # NOQA
from stormvogel import bird  # NOQA
from stormvogel import examples  # NOQA
from stormvogel import extensions  # NOQA
from stormvogel import stormpy_utils  # NOQA
from stormvogel.visualization import JSVisualization  # NOQA
from stormvogel.stormpy_utils.model_checking import *  # NOQA
import stormvogel.communication_server  # NOQA


def is_in_notebook():
    try:
        import sys

        shell = globals().get("__IPYTHON__", None)
        if shell is not None:
            return True
        if "ipykernel" not in sys.modules:
            return False

        from IPython.core.getipython import get_ipython

        ipython = get_ipython()

        if ipython is None or "IPKernelApp" not in ipython.config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if is_in_notebook():
    # Import and init magic
    from stormvogel.stormpy_utils import magic as magic

try:
    # Running stormvogel in the playground imports the playground's overwrite for show
    import playground

    _show = show  # Save original show so playground.py can call it without recursion
    show = playground.show  # type: ignore[assignment]
except ImportError:
    pass
