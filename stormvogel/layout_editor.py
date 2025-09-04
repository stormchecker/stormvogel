"""Layout editor."""

import stormvogel.communication_server
import stormvogel.dict_editor
import stormvogel.displayable
import stormvogel.layout
import stormvogel.visualization

import IPython.display as ipd
import ipywidgets as widgets
import logging


class LayoutEditor(stormvogel.displayable.Displayable):
    def __init__(
        self,
        layout: stormvogel.layout.Layout,
        visualization: stormvogel.visualization.JSVisualization | None = None,
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
    ) -> None:
        """Create an interactive layout editor, according to the schema. Display it using the show() method.
        Args:
            layout (Layout): The layout to be edited.
            visualization (Visualization, optional): A visualization that uses said layout. Defaults to None. Used to update the layout.
            output (widgets.Output, optional): An output widget within which the layout editor should be displayed. Defaults to None.
            do_display (bool, optional): Set to true iff you want the LayoutEditor to display. Defaults to True.
            debug_output (widgets.Output, optional): Debug information is displayed in this output. Leave to default if that doesn't interest you.
        """
        super().__init__(output, do_display, debug_output)
        self.vis: stormvogel.visualization.JSVisualization | None = visualization
        self.layout: stormvogel.layout.Layout = layout
        if self.vis is not None:
            self.vis.show(hidden=True)
        self.loaded: bool = False  # True iff the layout is done loading.
        self.editor = stormvogel.dict_editor.DictEditor(
            schema=self.layout.schema,
            update_dict=self.layout.layout,
            on_update=self.try_update,
            do_display=False,
        )

    def save_node_positions(self):
        """Try to save the positions of the nodes in the graph to the layout.
        The user is informed if this fails."""
        with self.debug_output:
            logging.debug(f"Status of vis {self.vis}")
        if self.vis is not None:
            with self.output:
                if stormvogel.communication_server.server is None:
                    with self.debug_output:
                        logging.info(
                            "Node positions won't be saved because the server is disabled."
                        )
                    with self.output:
                        print(
                            "Node positions won't be saved because the server is disabled."
                        )
                else:
                    try:
                        positions = self.vis.get_positions()
                        with self.debug_output:
                            logging.debug(positions)
                        self.layout.layout["positions"] = positions
                    except TimeoutError:
                        with self.debug_output:
                            logging.warning(
                                "Failed to save node positions in layout file."
                            )
                        with self.output:
                            self.__warn_failed_positions_save()

    def process_save_button(self):
        """Triggered whenever something is changed in the layout editor,
        but only does something if the save button was pressed."""
        self.layout.layout["saving"]["save_button"] = False
        # Also save the node positions.
        self.save_node_positions()
        try:
            self.layout.save(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )
        except RuntimeError:
            with self.output:
                print("Filename should end in .json")
        except OSError:
            with self.output:
                print(
                    f"Bad or inaccessible path or filename: {self.layout.layout['saving']['filename']}"
                )

    def process_load_button(self):
        """Triggered whenever something is changed in the layout editor,
        but only does something if the load button was pressed."""
        self.layout.layout["saving"]["load_button"] = False
        try:
            self.layout.load(
                self.layout.layout["saving"]["filename"],
                path_relative=self.layout.layout["saving"]["relative_path"],
            )
        except OSError:
            logging.warning("Loaded file does not exist.")
            with self.output:
                print(
                    f"Loaded file does not exist. {self.layout.layout['saving']['filename']}"
                )

    def process_reload_button(self):
        """Triggered whenever something is changed in the layout editor,
        but only does something if the reload button was pressed."""
        self.layout.layout["reload_button"] = False
        with self.debug_output:
            print("Received reload button request.")
            print(self.vis)

    def try_update(self):
        """Process the updates from the layout editor where required."""
        self.layout.copy_settings()
        if not self.loaded:
            return
        save = self.layout.layout["saving"]["save_button"]
        load = self.layout.layout["saving"]["load_button"]
        reload = self.layout.layout["reload_button"]
        if save:
            self.process_save_button()
        if load:
            self.process_load_button()
        if reload:
            self.process_reload_button()
        # The preceeding methods should never call self.show() or self.try_show_vis() since it's already called here.
        if load or reload:
            if self.vis is not None:
                self.vis.recreate()  # This also updates the edit groups as a side effect, so it should be called first.
                self.vis.show()
            self.show()
        elif self.vis is not None:
            self.vis.update()

    def show(self) -> None:
        """Display an interactive layout editor, according to the schema."""
        self.loaded = False
        with self.editor.output:  # Clear existing editor.
            ipd.clear_output()
        self.editor = stormvogel.dict_editor.DictEditor(
            schema=self.layout.schema,
            update_dict=self.layout.layout,
            on_update=self.try_update,
            do_display=False,
            output=widgets.Output(),
        )
        self.editor.show()
        box = widgets.VBox(children=[self.editor.output])
        with self.output:
            ipd.clear_output()
            ipd.display(box)
        self.maybe_display_output()
        self.loaded = True

    def __warn_failed_positions_save(self):
        print(f"""Could not save the node positions of this graph in {self.layout.layout["saving"]["filename"]}
Sorry for the inconvenience. See 'Implementation details' in docs.""")
