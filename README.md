# Stormvogel 🐦: An interactive approach to probabilistic model checking in Python

[![Coverage](https://stormchecker.github.io/stormvogel/latest/coverage.svg)](https://stormchecker.github.io/stormvogel/latest/)

The state-of-the-art model checking tools that are currently available are optimized to be efficient. The result of this is that they are quite hard to learn and use. Stormvogel flattens the learning cuve by providing easy and user-friendly APIs for creating probabilistic Markov models, and tools to visualize and debug them. It supports seemless conversion to the powerful [Storm(py) model checker](https://stormchecker.github.io/stormpy/api) out of the box.

## Features
* Easy APIs for constructing Markov models in dedicated data structures. Currently, DTMCs, MDPs, CTMCs, POMDPs and Markov Automata are supported. This also includes parametric and interval models.

* Seamless conversion between stormvogel and stormpy models with some runtime overhead. This allows, e.g., also using formats such as JANI and PRISM that are not supported by stormvogel directly. It is also possible to add support for a different model checker.

* Visualization of Markov models as an interactive graph and into SVGs via dot. This includes extensive layout options, and displaying model checking results and simulations in an interactive way.
* Support for gymnasium environments
* An extensive documentation with clear examples.

Check out the [the stormvogel documentation](https://stormchecker.github.io/stormvogel/) for examples of how to use stormvogel.

## Installation

### Pip (release version, recommended for users)

1. Run `pip install stormvogel`.
2. To also install stormpy, run `pip install stormpy`.
3. Run `jupyter lab`
4. Now a browser window should open that runs jupyter lab with stormvogel installed.

### Docker (release version)

1. Install `docker`. Run:
2. `docker run -it -p 8080:8080 stormvogel/stormvogel`
3. Now a browser window should open that runs jupyter lab with stormvogel and stormpy installed.

### For contributors (latest version)
Contributors need [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or newer to manage the development environment.

1. Clone the stormvogel repo (or your own fork) in a separate folder
2. In the stormvogel folder:
    ```
    uv sync --locked --extra storm
    uv run jupyter lab
    ```
    This creates `.venv` and installs the project plus development, test, lint, and documentation tools. Omit `--extra storm` for core-only development, or use `--all-extras` for all optional integrations (needed for documentation builds). Some extras require system libraries such as Cairo and Graphviz; docs also need Pandoc.
3. Install the `pre-commit` hook: `uv run pre-commit install`

Commit `uv.lock` when changing dependencies with `uv add` or `uv remove`. Use `uv lock --upgrade` to update locked versions and `uv sync --locked` to install them.

## Testing
```
uv run nox -s tests   # run test suite
uv run nox -s lint    # ruff + pyright
uv run nox -s docs    # sphinx-build (executes doc notebooks)
```
Or run `uv run pytest` directly without nox.

To test without development tools or optional integrations:
```
uv sync --locked --no-default-groups --group test
uv run --no-sync pytest
```
Use `--no-sync` here so uv does not reinstall the default development groups.

Notice that part of the tests will be skipped if stormpy is not installed.
## Authors
Stormvogel was mainly developed at Radboud University by Linus Heck, Pim Leerkes, and Ivo Melse under supervision from Sebastian Junges and Matthias Volk.

Thank you to our contributors: Luko van der Maas, Nicklas Osmers.

## License
Stormvogel is licenced under the [GPL-3.0 license](https://github.com/stormchecker/stormvogel?tab=GPL-3.0-1-ov-file).
