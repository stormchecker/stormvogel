"""Cell magic for writing models directly in notebooks."""

import tempfile
from warnings import warn
from IPython.core.magic import register_cell_magic
from IPython.core.getipython import get_ipython

try:
    import stormpy
except ImportError:
    stormpy = None


def parse_program(line, cell, parser_function, name):
    """Parse a program using stormpy.

    :param line: The IPython magic line argument (variable name to store result).
    :param cell: The cell content containing the program source.
    :param parser_function: The stormpy parser function to use.
    :param name: The magic command name (for error messages).
    :returns: The parsed program.
    """
    store_to_var = True
    # The line should be one word, which is the variable we write the result into
    if len(line.split()) != 1:
        warn(f"Write the program to a variable by doing %%{name} <variable>.")
        store_to_var = False

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(cell.encode("utf-8"))
        prism_filename = temp_file.name

    program = parser_function(prism_filename)
    if store_to_var:
        ipython = get_ipython()
        if ipython is not None:
            ipython.user_ns[line] = program
        else:
            raise RuntimeError("IPython not available, but variable provided")
    return program


@register_cell_magic
def prism(line, cell):
    """Parse a PRISM program from a notebook cell.

    :param line: The variable name to store the parsed program.
    :param cell: The cell content containing PRISM code.
    :returns: The parsed PRISM program.
    """
    assert stormpy is not None
    return parse_program(line, cell, stormpy.parse_prism_program, "prism")


@register_cell_magic
def jani(line, cell):
    """Parse a JANI model from a notebook cell.

    :param line: The variable name to store the parsed model.
    :param cell: The cell content containing JANI code.
    :returns: The parsed JANI model.
    """
    assert stormpy is not None
    return parse_program(line, cell, stormpy.parse_jani_model, "jani")
