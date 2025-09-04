# Stormvogel 🐦: An interactive approach to probabilistic model checking in Python

The state-of-the-art model checking tools that are currently available are optimized to be efficient, but not much effort goes into making them user-friendly. The result of this is that they are quite hard to learn and use. Stormvogel solves this problem by providing easy and user-friendly APIs for creating probabilistic Markov models, and tools to visualize and debug them. It supports seemless conversion to the powerful [Storm(py) model checker](https://moves-rwth.github.io/stormpy/#stormpy-api-reference) out of the box.

## Features
* Easy APIs for constructing Markov models in dedicated data structures. Currently, DTMCs, MDPs, CTMCs, POMDPs and Markov Automata are supported. This also includes parametric variants. Interval models are in development.

* Seamless conversion between stormvogel and stormpy models with some runtime overhead. This allows, e.g., also using formats such as JANI and PRISM that are not supported by stormvogel directly. It is also possible to add support for a different model checker.

* Visualization of Markov models as an interactive graph. This includes extensive layout options, and displaying model checking results and simulations in an interactive way.
* Support for gymnasium environments
* An extensive documentation with clear examples.

Check out the [the stormvogel documentation](https://stormchecker.github.io/stormvogel/) for examples of how to use stormvogel.

## Installation

### Pip (release version, recommended for users)

1. Run `pip install stormvogel`.
2. To also install stormpy, run `pip install stormpy`.

### Docker (release version)

1. Install `docker`. Run:
2. `docker run -it -p 8080:8080 stormvogel/stormvogel`
3. Now a browser window should open that runs jupyter lab with stormvogel and stormpy installed.

### For development (latest version)
Note that you might have to tweak these steps a bit to get it to work on your particular system, but here is an outline.

1. [Install the poetry package manager](https://python-poetry.org/docs/#installing-with-pipx)
2. Clone the stormvogel repo (or your own fork) in a separate folder
3. In the stormvogel folder:
    ```
    poetry install
    poetry shell # Activate poetry virtual environment
    pip install stormpy
    pip install . # Install stormvogel
    ```
    If installing stormpy fails in poetry, you can also try to follow the official [stormpy installation instructions](https://moves-rwth.github.io/stormpy/installation.html), and run `poetry shell` on top of the `virtualenv` environment that they describe there.
4. Install `pre-commit` hook: `pre-commit install`

## Testing
Notice that part of the tests will fail if stormpy is not installed.
```
pytest
```
## Authors
Stormvogel was mainly developed at Radboud University by Linus Heck, Pim Leerkes, and Ivo Melse under supervision from Sebastian Junges and Matthias Volk.

Thank you to our contributors: Luko van der Maas, Nicklas Osmers.

## License
Stormvogel is licenced under the [GPL-3.0 license](https://github.com/stormchecker/stormvogel?tab=GPL-3.0-1-ov-file).
