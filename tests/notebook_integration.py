# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Notebook integration tests
#
# Exercises the Jupyter-specific parts of stormvogel in a real kernel.
# Any cell raising an exception fails the test.
# Intended to be run via `pytest tests/test_notebook_runner.py`.

# %%
import os
import tempfile

import matplotlib
import stormvogel.examples as examples
import stormvogel.layout_editor as le
import stormvogel.model as model
from stormvogel.show import show
from stormvogel.visualization import JSVisualization, MplVisualization

# %%
# show() writes model.html to cwd — work in a temp dir so the test stays clean.
os.chdir(tempfile.mkdtemp())
matplotlib.use("Agg")  # headless — no display required

# %% [markdown]
# ## JS engine

# %%
m = model.new_dtmc()
s1 = m.new_state(labels=["end"])
m.set_choices(m.initial_state, [(1, s1)])
m.add_self_loops()

vis = show(m, engine="js")
assert isinstance(vis, JSVisualization)

# %%
html = vis.generate_html()
assert html
assert "<script" in html
assert "NetworkWrapper" in html

# %%
iframe = vis.generate_iframe()
assert "<iframe" in iframe

# %% [markdown]
# ## Matplotlib engine

# %%
vis_mpl = show(m, engine="mpl")
assert isinstance(vis_mpl, MplVisualization)

# %% [markdown]
# ## Additional model types

# %%
for create_model in [
    examples.create_die_dtmc,
    examples.create_car_mdp,
    examples.create_nuclear_fusion_ctmc,
]:
    m_ex = create_model()
    vis_ex = show(m_ex, engine="js")
    assert isinstance(vis_ex, JSVisualization)
    assert vis_ex.generate_html()

# %% [markdown]
# ## JSVisualization does not start a server in DOCUMENTATION mode

# %%
# JSVisualization.__init__ checks DOCUMENTATION and skips server init.
# The env var is set to "1" by the pytest runner.
assert os.environ.get("DOCUMENTATION") == "1"
assert vis.server is None

# %% [markdown]
# ## LayoutEditor construction

# %%
editor = le.LayoutEditor(vis.layout, visualization=None, do_display=False)
assert editor is not None

# %% [markdown]
# ## show() with layout editor

# %%
# This path creates a LayoutEditor, wraps it and the vis in a widgets.HBox,
# and calls ipd.display(box) — only meaningful to test in a real kernel.
vis_with_editor = show(m, engine="js", show_editor=True)
assert isinstance(vis_with_editor, JSVisualization)
