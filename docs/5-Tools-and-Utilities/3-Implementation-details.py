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
# # Implementation details
# This notebook explains the way that some features are implemented in order to help with some issues and for future maintainability.

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Communication server remark
# **Note: Only read this if you got the warning, and you want to save node positions from a visualization.
# The warning also shows up in the docs, but that is normal.**
#
# Stormvogel uses a local server (or multiple local servers if multiple kernels are running) to communicate between visualization elements (Javascript frontend) and Python. Sometimes this communication fails and when this happens the user gets a warning. Stormvogel is perfectly usable without this server, but you can no longer save the node positions of a layout. If you really need to save node positions, here are some pointers to fixing the issue.

# %% [markdown]
# ### Could not start server / test request failed / no free port.
# 1) Restart the kernel and re-run.
# 2) If you are working using docker or remotely, ports 8889-8905 (at least some of them) need to be available.
#    * **SSH**: `ssh -N -L 8889:localhost:8889 YOUR_SSH_CONFIG_NAME`.
#    * **Docker**: `docker run --network=host -it stormvogel/stormvogel`
#    * **Both**: If you cannot use this range (8889-8905), try setting `stormvogel.communication_server.min_port` and `max_port` to a port range that is available.
# 4) You might also want to consider changing `stormvogel.communication_server.localhost_address` to the IPv6 loopback address if you are using IPv6.
#
# Please contact the stormvogel developpers if you keep running into issues.
#
# You can also disable the warning message by setting `use_server` to `False` in your call to `show`
#
# Example:

# %%
import stormvogel
stormvogel.communication_server.min_port = 3000
stormvogel.communication_server.max_port = 3020
# Stormvogel looks in range 3000-3020 for an available port
stormvogel.communication_server.localhost_address = "::1/128" # IPv6 loopback address
stormvogel.communication_server.use_server = False # Stormvogel no longer tries to start the server.

# %% [markdown]
# # Visualization
# The visualization part of Stormvogel generates a html file which is then displayed in the notebook and rendered in the browser. Here is an outline of the main modules and how they are related. In order to understand the code, it might be useful to read up on IPython widgets first, as they are used extensively. The most important thing to understand is that widgets can *capture* output. This means that the output in the block is not shown in the notebook, but saved in the capturing widget. If you want to see the output, the widget that is capturing needs to be displayed.
#
# ## Note on debugging
# Debugging can be difficult because we are working in Jupyter notebook and in both Python and Javascript which need to communicate. Sometimes overly eager browser security policies also can also get in the way. IPython and Jupyter notebooks can also be flaky in general: I have had multiple problems where my solution would work only 50-70% of the times for unknown reasons. I will provide tips for debugging when applicable.
#
# ## Python
# ### show.py
# Contians a method show() that displays a stormvogel model as a directed graph. show() creates a Visualization object, and calls Visualization.show().
# The most important argument is show_editor. If this is enabled, not only the model will be displayed, but also a GUI next to it where the user can control how the model looks (LayoutEditor). In this case, show wraps the visualization and the editor in a Horizontal box (HBox). Show returns the Visualization object.
#
# ### Displayable (displayable.py)
# Any class with a show() method is a subclass from Displayable. A Displayable has an output widget. All things that the object displays should be *captured* by this output widget. If do_display is enabled, then the object shows its output, otherwise it doesn't and some other object should manage this. An example of this can be found in show() in show.py: do_display is disabled in the call to Visualziation. If the edtior is enabled, it is wrapped in a box which is then displayed.
#
# ### Visualization (visualization.py)
# The Visualization object 'reads' the provided model and displays the graph using a Network object. It also uses a layout object for knowing what the graph should look like. It also exposes some methods to the user for highlighting states/actions in the graph, and exporting the visualization in different formats (e.g. svg).
#
# ### Network (network.py)
# Generates a visual graph by generating html/javascript code and displaying it in the notebook. More on the JavaScript part later.
#
# ### Layout (layout.py)
# Responsible for storing the current layout of the network and saving/loading the layout from/to files. Also responsible for storing the schema, which is used in the LayoutEditor.
#
# ### LayoutEditor (layout_editor.py)
# Displays an interactive layout editor with a GUI. What this layout editor should look is specified in layout.schema. Under the hood, it uses DictEditor.
#
# **Debugging:** print and log statements that are triggered by user input in the GUI are (ususally) hidden. To overcome this, use debug_output; capture your print/log statements with debug_output, and then display debug_output somewhere in your notebook.
#
# ### DictEditor (dict_editor.py)
# Provided with a nested dictionary of values, and a schema with the same structure, display a GUI editor for this dictionary using IPython widgets.
#
# ### Communication Server (communication_server.py)
# Communicating from Python to Javascript is easy; it suffices to generate a piece of Javascript code and call IPython.Javascript(...). Communicating from Javascript to Python is annoying. You can essentially see the Javascript part in the user's browser as a front end and the python kernel as a backend. They communicate through a server that is hosted on the pc that is running the kernel. The communication server provides an abstraction to make communicating back easier, see methods add_event and result.
#
# Futhermore, it is important to know that multiple visualizations in the same notebook share the same server, while visualizations in separate notebooks use separate python processes, and therefore they have to use different servers. This also means that they need different ports.
#
# ## Javascript/HTML
# **Debugging/editing:** If you want to edit anything that is only related to Javascript, I would recommend just generating a visualization, using Visualziation.export('html', 'model'), and then opening model.html in a separate folder using your browser. Then you can edit the Javascript here directly and see if it works before bringing it into html_generation.py, which can be really annoying due to caching. (You could be seeing an old version that doesn't have your changes yet in Jupyter notebooks.)
#
# The main library which is used for plotting the graph is called [vis.js](https://visjs.org/). In order to enable exporting the network to svg/pdf, we also have a library called [svgcanvas](https://github.com/zenozeng/svgcanvas/tree/v2.6.0). If you use the standard version of vis.js, exporting to svg does not work properly (the borders of nodes do not display). In order to fix this, I patched the vis.js version that is used in stormvogel. If you want to find the places where it was patched, use ctrl+f "SVG-PATCH".
#
# Please don't judge my hacky solutions too much, thank you.
