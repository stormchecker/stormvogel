"""The stormvogel package"""

from stormvogel import layout  # NOQA
from stormvogel.layout import Layout  # NOQA

# from stormvogel.stormpy_utils.mapping import *  # NOQA
# from stormvogel.stormpy_utils.model_checking import model_checking  # NOQA
from stormvogel.model import *  # NOQA
from stormvogel.property_builder import build_property_string  # NOQA
from stormvogel.result import *  # NOQA
from stormvogel.show import *  # NOQA
from stormvogel.simulator import *  # NOQA
from stormvogel import bird  # NOQA
from stormvogel import examples  # NOQA
from stormvogel import extensions  # NOQA
from stormvogel import stormpy_utils  # NOQA
from stormvogel.visualization import JSVisualization  # NOQA
from stormvogel.stormpy_utils.model_checking import *  # NOQA

import sys


def is_in_notebook():
    return (
        "ipykernel" in sys.modules
        or "IPython" in sys.modules
        and hasattr(sys.modules["IPython"], "get_ipython")
        and getattr(sys.modules["IPython"].get_ipython(), "config", {}).get(
            "IPKernelApp", False
        )
    )


if is_in_notebook():
    # Import and init magic only if in notebook
    from stormvogel.stormpy_utils import magic as magic
