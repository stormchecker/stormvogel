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
# # Model visualization
#
# When visualizing a model using, there are a lot of layout options. In this notebook we will show how to edit the way that a model looks.
#
# Remember the car model from the introduction? You will notice that it does not have the same colors as it had in the introduction notebook. The reason for this is that in the introduction, we specified a different **Layout**.
#
# **Note:** that this notebook does not work properly in the HTML docs. It will likely work in any Jupyter notebook server, but we recommend Jupyter lab, because we used this to create and test stormvogel.

# %%
from stormvogel import *
vis = show(examples.create_car_mdp())

# %% [markdown]
# ## Basic editing
# Let's edit the layout! The first thing that you need to do is open the layout editor by adding `show_editor = True` to the call to `show`.
# A menu will open up with a lot of different sections that can be opened by clicking on them. First of all, let's try to alter the colors of states and actions. Click on 'states' or 'actions' and then adjust 'bg color' by clicking on the box with a color. A color picker will open up. You will notice that there are a lot of different options, feel free to play around with them and see the changes to the model in real time.
#
# ### Dragging nodes
# You can also change the position of nodes by selecting them in the displayed network, and moving them around. This does not require the editor, but your changes here can only be saved if you are using it.
#
# ### Exploring
# Go to 'misc', set 'explore' to True and press the 'Reload' button. Now you can explore the model. Clicking on a node reveals its successors.
#
# ### Edit groups
# In the model in the introduction notebook, states with different labels had different colors, but right now we only have a single group for states, and a single group for actions. You can change this by going to 'edit_groups'. By default, there are two groups: one for states and one for actions.
# * Remove both of them from the list by pressing the cross next to them. Now press the reload button at the bottom, and you will see that 'states' and 'actions' are removed from the editing menu.
# * Now open 'edit_groups' again and press the empty box. It will give you some options that correspond to the labels of the states that are present. Select 'green_light', 'red_light' and 'accident'. Press reload again. Now you can edit these groups and create a similar layout to the introduction notebook.
#
# ### Saving layouts
# It is always useful to save your layout so that it won't be gone once you re-execute the cell. An easy way of doing this is by going to 'saving', and providing a path to store the json file that contains your layout. Then press 'Save'. If you re-execute the cell, and go to saving, enter the same name, and press 'Load', your layout will be back. In practice, it is more useful to also be able to load a layout without having to go through the menu. You can do this easily by adding `layout=Layout("file/name.json")` the call to show. Try it!
#
# ### Exporting visualizations
# Visualizations in stormvogel are rendered to HTML. This means that you can also export them and embed them in other web pages. You can also export to svg, pdf, and even latex. Give it a try!
#
# **Note:** There are additional model visualization features for schedulers and results, but they will be explained in their respective notebooks.

# %%
#vis = stormvogel.show.show(stormvogel.examples.car.car, show_editor=True)

#vis.export("html", "vis_html")
#vis.export("svg", "vis_svg")
#vis.export("pdf", "vis_pdf")
#vis.export("latex", "vis_latex")

# %% [markdown]
# ## Advanced editing
# If the layout editor does not provide a visualization option that you really want, it is also possible to edit the layout manually. The following code shows you what the default layout looks like. `Layout.layout` is a dictionary that stores all the layout information. You could edit this directly, but this can get annoying because it is a nested dictionary. We provide a method `set_value` to make this easier. Alternatively, If you have a lot of options that you want to change at the same time, it is probably easier to overwrite `Layout.layout` directly.
#
# Try commenting in the uncommented line and see the changes. under "groups". Then see the changes in the model by re-executing the final cell.
#
# The structure of the layout is derived from the vis.js library (that is also used to display the models). There are many other options that can be changed but which are not in the layout editor. For more information, we refer to the [vis.js documentation](https://visjs.github.io/vis-network/docs/network/#options).

# %%
my_layout = layout.DEFAULT()
my_layout.layout["groups"]["states"] = {}
#my_layout.set_value(["groups", "states", "color", "background"], "orange")
my_layout.layout

# %%
vis = show(examples.create_car_mdp(), layout=my_layout)

# %% [markdown]
# ## Javascript injection
# If you want to customize the visualization even more, you can modify the source code, or you can use Javascript injection from a notebook. Here is a small example of javascript that destroys the network. Also see `html_generation.py` and the documentation of `vis.js`.

# %%
import IPython.display as ipd
# ipd.display(ipd.Javascript(
# f"""{vis.nt.network_wrapper}.network.destroy()"""))
