"""Shorter api for showing a model."""

from typing import Callable, Any, TYPE_CHECKING
import stormvogel.model
from stormvogel.layout import Layout, DEFAULT, SV
import stormvogel.result
from stormvogel.visualization import JSVisualization, MplVisualization

if TYPE_CHECKING:
    import ipywidgets as widgets
    from stormvogel.graph import ModelGraph


def show(
    model: stormvogel.model.Model,
    result: stormvogel.result.Result | None = None,
    engine: str = "js",
    pos_function: Callable[["ModelGraph"], dict[int, Any]] | None = None,
    pos_function_scaling: int = 750,
    scheduler: stormvogel.result.Scheduler | None = None,
    layout: Layout | None = None,
    show_editor: bool = False,
    debug_output: "widgets.Output | None" = None,
    use_iframe: bool = False,
    do_init_server: bool = True,
    max_states: int = 1000,
    max_physics_states: int = 500,
) -> JSVisualization | MplVisualization | None:
    """Create and show a visualization of a Model using a visjs Network.

    :param model: The stormvogel model to be displayed.
    :param result: A result associated with the model.
        The results are displayed as numbers on a state. Enable the layout editor for options.
        If this result has a scheduler, then the scheduled actions will have a different color etc. based on the layout.
    :param engine: The engine that should be used for the visualization.
        Can be either ``"js"`` for the interactive HTML/JavaScript visualization, or ``"mpl"`` for matplotlib.
    :param pos_function: Function that takes a ModelGraph and maps it to a dictionary of node positions.
        It is often useful to import these from networkx, see https://networkx.org/documentation/stable/_modules/networkx/drawing/layout.html for some examples.
        In particular, ``nx.bfs_layout`` seems to work great for models with a directed acyclic graph structure.
    :param pos_function_scaling: Scaling factor for the positions when using networkx positions.
    :param scheduler: The scheduled actions will have a different color etc. based on the layout.
        If both result and scheduler are set, then scheduler takes precedence.
    :param layout: Layout used for the visualization.
    :param show_editor: For interactive visualization. Show an interactive layout editor.
    :param debug_output: For interactive visualization. Output widget that can be used to debug interactive features.
    :param use_iframe: For interactive visualization. Wrap the generated HTML inside of an IFrame.
        In some environments, the visualization works better with this enabled.
    :param do_init_server: For interactive visualization. Initialize a local server that is used for communication between JavaScript and Python.
        If this is set to ``False``, then exporting network node positions and SVG/PDF/LaTeX is impossible.
    :param max_states: If the model has more states, then the network is not displayed.
    :param max_physics_states: If the model has more states, then physics are disabled.
    :returns: A :class:`JSVisualization` or :class:`MplVisualization` object, or ``None`` on error.
    :raises ValueError: If the engine is not recognized.
    """
    import ipywidgets as widgets
    import IPython.display as ipd

    if layout is None:
        layout = DEFAULT()

    # Use networkx positions if the user wants it.
    if pos_function:
        from stormvogel.graph import ModelGraph

        G = ModelGraph.from_model(model)
        try:
            pos = pos_function(G)
            layout = layout.set_nx_pos(pos, scale=pos_function_scaling)
        finally:
            pass

    if engine == "js":
        vis = JSVisualization(
            model=model,
            result=result,
            scheduler=scheduler,
            layout=layout,
            debug_output=debug_output or widgets.Output(),
            do_init_server=do_init_server,
            use_iframe=use_iframe,
            max_states=max_states,
            max_physics_states=max_physics_states,
        )
        if show_editor:
            import stormvogel.layout_editor

            e = stormvogel.layout_editor.LayoutEditor(
                layout,
                vis,
                do_display=False,
                debug_output=debug_output or widgets.Output(),
            )
            e.show()
            box = widgets.HBox(children=[vis.output, e.output])
            ipd.display(box)
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
        vis = MplVisualization(
            model=model, result=result, scheduler=scheduler, layout=layout
        )
        vis.show()
        return vis
    else:
        print(f"Unkown engine: {engine}. Choose 'js' or 'mlp'.")


def show_bird() -> JSVisualization:
    """Show a simple model with a bird state."""
    m = stormvogel.model.new_dtmc(create_initial_state=False)
    m.new_state(labels=["init", "🐦"])
    m.add_self_loops()
    vis = show(m, show_editor=False, do_init_server=False, layout=SV())
    assert isinstance(vis, JSVisualization)
    return vis
