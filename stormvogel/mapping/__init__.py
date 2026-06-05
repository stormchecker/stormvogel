from warnings import warn

from stormvogel.stormpy_utils.mapping import *  # NOQA

# The mapping top level module is deprecated
warn(
    "The 'mapping' module is deprecated. Please import from 'stormvogel.stormpy_utils.mapping' instead.",
    DeprecationWarning,
    stacklevel=2,
)
