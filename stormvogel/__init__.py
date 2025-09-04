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


def is_in_notebook():
    try:
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
