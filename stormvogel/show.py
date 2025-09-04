"""Shorter api for showing a model."""

import stormvogel.model
import stormvogel.layout
import stormvogel.visualization
import stormvogel.layout_editor
import stormvogel.result
import networkx as nx

import ipywidgets as widgets
import IPython.display as ipd


def show(
    model: stormvogel.model.Model,
    engine: str = "js",
    nx_pos: bool = False,
    nx_scale: int = 500,
    result: stormvogel.result.Result | None = None,
    scheduler: stormvogel.result.Scheduler | None = None,
    layout: stormvogel.layout.Layout | None = None,
    show_editor: bool = False,
    debug_output: widgets.Output = widgets.Output(),
    use_iframe: bool = False,
    do_init_server: bool = True,
    max_states: int = 1000,
    max_physics_states: int = 500,
) -> (
    stormvogel.visualization.JSVisualization
    | stormvogel.visualization.MplVisualization
    | None
):
    """Create and show a visualization of a Model using a visjs Network

    Args:
        model (Model): The stormvogel model to be displayed.
        engine (str): The engine that should be used for the visualization.
            Can be either "js" for the interactive html/JavaScript visualization, or "mpl" for matplotlib.
        nx_pos (bool): Whether networkx should be used to determine the node positions in the graph.
            Highly recommended for bigger and acyclic models!
        nx_scaling_factor (int): Scaling factor for the positions when using networkx positions. Defaults to 500.
        result (Result, optional): A result associatied with the model.
            The results are displayed as numbers on a state. Enable the layout editor for options.
            If this result has a scheduler, then the scheduled actions will have a different color etc. based on the layout
        scheduler (Scheduler, optional): The scheduled actions will have a different color etc. based on the layout
            If both result and scheduler are set, then scheduler takes precedence.
        layout (Layout): Layout used for the visualization.
        show_editor (bool): For interactive visualizaiton. Show an interactive layout editor.
        debug_output (widgets.Output): For interactive visualization. Output widget that can be used to debug interactive features.
        use_iframe(bool): For interactive visualziation. Wrap the generated html inside of an IFrame.
            In some environments, the visualization works better with this enabled.
        do_init_server(bool): For interactive visualization. Initialize a local server that is used for communication between Javascript and Python.
            If this is set to False, then exporting network node positions and svg/pdf/latex is impossible.
        max_states (int): If the model has more states, then the network is not displayed.
        max_physics_states (int): If the model has more states, then physics are disabled.
    Returns: Visualization object.
    """
    if layout is None:
        layout = stormvogel.layout.DEFAULT()

    # Use networkx positions if the user wants it.
    if nx_pos:
        G = stormvogel.visualization.ModelGraph.from_model(model)
        pos = nx.bfs_layout(G, start=model.get_initial_state().id)
        layout = layout.set_nx_pos(pos, scale=nx_scale)

    if engine == "js":
        vis = stormvogel.visualization.JSVisualization(
            model=model,
            result=result,
            scheduler=scheduler,
            layout=layout,
            debug_output=debug_output,
            do_init_server=do_init_server,
            use_iframe=use_iframe,
            max_states=max_states,
            max_physics_states=max_physics_states,
        )
        if show_editor:
            e = stormvogel.layout_editor.LayoutEditor(
                layout, vis, do_display=False, debug_output=debug_output
            )
            e.show()
            box = widgets.HBox(children=[vis.output, e.output])
            ipd.display(box)
            vis.update()
        else:  # Unfortunately, the sphinx docs only work if we save the html as a file and embed.
            if use_iframe:
                iframe = vis.generate_iframe()
            else:
                iframe = vis.generate_html()
            with open("model.html", "w") as f:
                f.write(iframe)
            ipd.display(ipd.HTML(filename="model.html"))
        return vis
    elif engine == "mpl":
        vis = stormvogel.visualization.MplVisualization(
            model=model, result=result, scheduler=scheduler, layout=layout
        )
        vis.show()
        return vis
    else:
        print(f"Unkown engine: {engine}. Choose 'js' or 'mlp'.")


def show_bird():
    m = stormvogel.model.new_dtmc(create_initial_state=False)
    m.new_state("🐦")
    m.add_self_loops()
    return show(
        m, show_editor=False, do_init_server=False, layout=stormvogel.layout.SV()
    )
