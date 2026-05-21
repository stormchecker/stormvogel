# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
from pathlib import Path
import tomllib

project = "stormvogel"
copyright = "2024, stormvogel team"
author = "stormvogel team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "autoapi.extension",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]
autoapi_dirs = [Path("../stormvogel")]
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    # "imported-members" omitted: prevents duplicate object descriptions and
    # sympy docstring leakage. TODO: fix properly with __all__ in model submodules.
]
autosummary_generate = True

nbsphinx_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/orchard/**", "conf.py"]

# Suppress ambiguous cross-reference warnings caused by the intentional
# three-level re-export chain (stormvogel.X / stormvogel.model.X / stormvogel.model.x.X).
# TODO: fix properly by adding __all__ to model submodules and collapsing to one re-export level.
suppress_warnings = ["ref.python"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# When building the "latest" (development) docs, show a banner linking to the
# last stable release.  RELEASE_DOCS_URL and RELEASE_TAG are injected by CI.
_release_url = os.environ.get("RELEASE_DOCS_URL", "")
_release_tag = os.environ.get("RELEASE_TAG", "")
if _release_url and _release_tag:
    html_theme_options = {
        "announcement": (
            f"You are viewing the <strong>development</strong> version of the docs. "
            f'<a href="{_release_url}">View the latest stable release ({_release_tag}) →</a>'
        )
    }

# Load current version

pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject = tomllib.load(f)

if _release_url and _release_tag:
    html_title = "stormvogel"
else:
    html_title = f"stormvogel v{_release_tag}"
