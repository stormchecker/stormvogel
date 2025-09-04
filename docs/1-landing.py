# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Welcome to Stormvogel!
#
#
# This notebook provides an introduction to the stormvogel project, a first example, and pointers to further material. We recommend to also look at
#
# - The source code, hosted at [Github](https://github.com/moves-rwth/stormvogel)
# - The python packages, hosted at [Pypi](https://pypi.org/project/stormvogel/)
# - User documentation, which includes this notebook [Docs](https://moves-rwth.github.io/stormvogel/)
# - Reach out to us at [Discord Server](https://discord.gg/byeKSasJY6)
#

# %% [markdown]
# ## Installing with pip
#
# The easiest way to install stormvogel is by pip:
# ```
# pip install stormvogel
# pip install stormpy # Optional, for efficient model checking algorithms
# ```

# %% [markdown]
# ## Running the Docker container
#
# Another way to run stormvogel is by running our [Docker](https://www.docker.com) container on your local machine.
# ```
# docker pull stormvogel/stormvogel
# docker run -it -p 8080:8080 stormvogel/stormvogel
# ```
#
# See the GitHub repository for more advanced installation options.

# %% [markdown]
# ## What is probabilistic model checking?
#
# ### Qualitative model checking
# Model checking, in this context, refers to methods and techniques to exhaustively analyze whether a model of system behavior satisfies a given formal specification. For the (initial) development of such techniques, Clarke, Emerson and Sifakis were awarded the ACM Turing Award in 2007.   Specifically, these models are typically state-based formalism that describe the dynamics of a system, such as a hardware circuit or a computer program. Simple specifications ask: *Is it possible to reach this error state?*, *Is it possible to simultaneously open the microwave while it is active?*, *Can one throw 3 sixes in a row?* etc. Traditionally, the outcome is a yes/no answer, but typically, model checking techniques output some kind of witness that certifies why the answer is yes or provides a counterexample when the answer is no.
#
# Below we have a toy model of a car. The idea is that the car always choose whether to 'wait' (that is, continue driving), or to 'brake'. This car does not satisfy the specification to never reach the state 'accident' as a driver could choose to consistently continue driving.

# %% jupyter={"is_executing": true}
from stormvogel import *
vis = show(examples.create_car_mdp(), layout=stormvogel.layout.Layout("layouts/car.json"))

# %% [markdown]
# ### Probabilistic Model Checking
# Model checking asks whether something is possible (or dually, whether something always holds). Its perspective cannot distinguish between likely and unlikely events. However, often it is not possible to completely rule out unwanted behavior. In those cases, one may still want to assert that it the unwanted behavior is unlikely to occur: It is possible that all packets are dropped when communicating over ethernet, but it is highly unlikely, just like it is possible but highly unlikely to throw 100 sixes in a row.
#
# Probabilistic model checking therefore analyses Markov models, state-based formalisms that describe the dynamics of probabilistic processes. It yields guarantees that, e.g., *the probability to reach a dangerous state is sufficiently small*, but also that *the expected energy consumption is low*, or that *the probability that a battery depletes during a drone lands* is tiny.
#
# Probabilistic model checking is a general technique and can be applied to any problem which can be modelled as Markov model, such as an Markov decision process (MDP). Given that these models are used in many different domains, the use of probabilistic model checking is not limited to software and hardware verification.
#
# ## Tools for Probabilistic Model Checking
# The research community in probabilistic model checking is largely tool driven. While many smaller prototype implementations exist, some larger ecosystems have emerged, in particular PRISM, Modest, and Storm. For an overview of the field, we refer to this recent paper.  Stormvogel, as the name suggests, has been primarily developed on top of the ecosystem that the model checker Storm provides.
#
#
# ### What is Storm, Stormpy and Stormvogel?
# * [Storm](https://www.stormchecker.org/) is a probabilistic model checker, written in C++, whose primary goal is to provide the fastest algorithms for standard probabilistic model checking queries. This goal comes at a cost of a steep learning curve.
# * [Stormpy](https://moves-rwth.github.io/stormpy/) is an low-level API with Python bindings for Storm that allows developers working with Storm to quickly prototype new algorithms.  It is structured mostly similarly to Storm itself.
# * Stormvogel is a collection of Python APIs and visualization tools with ease-of-use as top priority: The goal of Stormvogel is to provide an accessible way to do probabilistic model checking and to provide educational tools about model checking. It can be easily interfaced with Stormpy to profit from high-performance model checking.
#
# ## What does Stormvogel provide?
# While Stormvogel is still under active development, these are the current key features.
#
# * APIs for constructing Markov models in dedicated data structures. Currently, DTMCs, MDPs, CTMCs, POMDPs and Markov Automata are supported.
#   - The `model` API can be used to construct a model directly by explicitely defining a set of states and transitions.
#   - The `bird` API can be used to construct a model by defining a delta function. The structure is similar to PRISM.
#   - The `PRISM` API can be used to construct a model using the PRISM syntax, which has been the standard in model checking over the last years.
# * Seamless conversion between stormvogel and stormpy models with some runtime overhead. This allows, e.g., also using formats such as JANI that are not supported by stormvogel directly.
# * Visualization in notebooks.
#   - Visualize your models by displaying the states, actions and edges.
#   - Edit and save/load the layout of your models interactively using a GUI.
#   - Display model checking results in an interactive way.
#   - Visualize a simulation of a model.
# * An interface for model checking.
#   - A function for model checking that takes a `PRISM` property string. Note that this uses Storm under the hood, so it is probably fast.
#   - A GUI for making it easier to construct these property strings for beginners (the *property builder*)
# * An extensive documentation, with examples and documentation
#
# Details about these can be found in the remaining notebooks.
#
# ## Questions/help, bugs & contributing
# To suggest a feature or report a bug, simply create an issue on [Github](https://github.com/moves-rwth/stormvogel). If you would like to contribute to the project yourself, you can always create a pull request or join our [public Discord server](https://discord.gg/byeKSasJY6).
#
# Stormvogel was developed at the Radboud University in 2024-2025 by Linus Heck, Pim Leerkes and Ivo Melse under the supervision of Sebastian Junges (Radboud) and Matthias Volk (Eindhoven University of Technology). We would like to thank the Dutch Research Council (NWO) for providing funding for the project via the NWO Open Science Fund Grant StormAE.
#
# Good luck using stormvogel! And if you ever get bored or frustrated, the bird is here to cheer you up! (This bird is in fact a singleton DTMC!)

# %%
bird = show_bird()
