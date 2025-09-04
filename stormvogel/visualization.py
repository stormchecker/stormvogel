"""Contains the code responsible for model visualization."""

from collections.abc import Callable
from typing import Any
import cairosvg
import pathlib
import warnings
from time import sleep

from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import PathCollection
import stormvogel.model
import stormvogel.layout
import stormvogel.result
import stormvogel.html_generation
from stormvogel.autoscale_svg import autoscale_svg
from .graph import ModelGraph, NodeType
from . import simulator

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

import logging
import json
import html
import ipywidgets as widgets
import IPython.display as ipd
import random
import string
from fractions import Fraction


def und(x: str) -> str:
    """Replace spaces by underscores."""
    return x.replace(" ", "_")


def random_word(k: int) -> str:
    """Random word of lenght k"""
    return "".join(random.choices(string.ascii_letters, k=k))


def random_color() -> str:
    """Return a random HEX color."""
    return "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])


def blend_colors(c1: str, c2: str, factor: float) -> str:
    """Blend two colors in HEX format. #RRGGBB.
    Args:
        color1 (str): Color 1 in HEX format #RRGGBB
        color2 (str): Color 2 in HEX format #RRGGBB
        factor (float): The fraction of the resulting color that should come from color1."""
    r1 = int("0x" + c1[1:3], 0)
    g1 = int("0x" + c1[3:5], 0)
    b1 = int("0x" + c1[5:7], 0)
    r2 = int("0x" + c2[1:3], 0)
    g2 = int("0x" + c2[3:5], 0)
    b2 = int("0x" + c2[5:7], 0)
    r_res = int(factor * r1 + (1 - factor) * r2)
    g_res = int(factor * g1 + (1 - factor) * g2)
    b_res = int(factor * b1 + (1 - factor) * b2)
    return "#" + "".join("%02x" % i for i in [r_res, g_res, b_res])


class VisualizationBase:
    """Base class for visualizing a Stormvogel MDP model.

    This class sets up a visual representation of a Stormvogel model, optionally
    incorporating the result of a model checking operation and a scheduler
    (i.e., a strategy for selecting actions). It constructs an internal graph
    of the model and manages visual layout settings, such as active and
    possible display groups.

    If a scheduler is not explicitly provided but is available in the model
    checking result, it will be used automatically. When a scheduler is set,
    the "scheduled_actions" group is activated in the layout; otherwise, it
    is deactivated.

    Attributes:
        model (stormvogel.model.Model): The MDP model being visualized.
        layout (stormvogel.layout.Layout): The layout configuration for the visualization.
        result (stormvogel.result.Result | None): The result of a model checking operation.
        scheduler (stormvogel.result.Scheduler | None): A scheduler representing a path
            through the model.
        G (ModelGraph): The internal graph structure representing the model.

    Args:
        model (stormvogel.model.Model): The MDP model to visualize.
        layout (stormvogel.layout.Layout, optional): Layout settings for the visualization.
            Defaults to `stormvogel.layout.DEFAULT()`.
        result (stormvogel.result.Result, optional): The result of a model checking
            operation, which may contain a scheduler.
        scheduler (stormvogel.result.Scheduler, optional): An explicit scheduler
            defining actions to take in each state.
    """

    def __init__(
        self,
        model: stormvogel.model.Model,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        result: stormvogel.result.Result | None = None,
        scheduler: stormvogel.result.Scheduler | None = None,
    ) -> None:
        self.model = model
        self.layout = layout
        self.result = result
        self.scheduler = scheduler
        # If a scheduler was not set explictly, but a result was set, then take the scheduler from the results.
        if self.scheduler is None:
            if self.result is not None:
                self.scheduler = self.result.scheduler

        # Set "scheduler" as an active group iff it is present.
        if self.scheduler is not None:
            self.layout.add_active_group("scheduled_actions")
        else:  # Otherwise, disable it
            self.layout.remove_active_group("scheduled_actions")
        self.recreate()

    def recreate(self):
        """Recreate the ModelGraph and set the edit groups."""
        self.G = ModelGraph.from_model(
            self.model,
            state_properties=self._create_state_properties,
            action_properties=self._create_action_properties,
            transition_properties=self._create_transition_properties,
        )
        underscored_labels = set(map(und, self.model.get_labels()))
        possible_groups = underscored_labels.union(
            {"states", "actions", "scheduled_actions"}
        )
        self.layout.set_possible_groups(possible_groups)

    def _format_number(self, n: stormvogel.model.Value) -> str:
        """Call number_to_string in model.py while accounting for the settings specified in the layout object."""
        return stormvogel.model.value_to_string(
            n,
            self.layout.layout["numbers"]["fractions"],
            self.layout.layout["numbers"]["digits"],
            self.layout.layout["numbers"]["denominator_limit"],
        )

    def _format_result(self, s: stormvogel.model.State) -> str:
        """Create a string that shows the result for this state. Starts with newline.
        If results are not enabled, then it returns the empty string."""
        if self.result is None or not self.layout.layout["results"]["show_results"]:
            return ""
        result_of_state = self.result.get_result_of_state(s)
        if result_of_state is None:
            return ""
        return (
            "\n"
            + self.layout.layout["results"]["result_symbol"]
            + " "
            + self._format_number(result_of_state)
        )

    def _format_observations(self, s: stormvogel.model.State) -> str:
        """Create a String that shows the observation for this state (FOR POMDPs).
        Starts with newline."""
        if (
            s.observation is None
            or not self.layout.layout["state_properties"]["show_observations"]
        ):
            return ""
        else:
            return (
                "\n"
                + self.layout.layout["state_properties"]["observation_symbol"]
                + " "
                + str(s.observation.observation)
            )

    def _group_state(self, s: stormvogel.model.State, default: str) -> str:
        """Return the group of this state.
        That is, the label of s that has the highest priority, as specified by the user under edit_groups"""
        und_labels = set(map(lambda x: und(x), s.labels))
        res = list(
            filter(
                lambda x: x in und_labels, self.layout.layout["edit_groups"]["groups"]
            )
        )
        return und(res[0]) if res != [] else default

    def _group_action(self, s_id: int, a: stormvogel.model.Action, default: str) -> str:
        """Return the group of this action. Only relevant for scheduling"""
        # Put the action in the group scheduled_actions if appropriate.
        if self.scheduler is None:
            return default

        choice = self.scheduler.get_choice_of_state(self.model.get_state_by_id(s_id))
        return "scheduled_actions" if a == choice else default

    def _format_rewards(
        self, s: stormvogel.model.State, a: stormvogel.model.Action
    ) -> str:
        """Create a string that contains either the state exit reward (if actions are not supported)
        or the reward of taking this action from this state. (if actions ARE supported)
        Starts with newline"""
        if not self.layout.layout["state_properties"]["show_rewards"]:
            return ""
        EMPTY_RES = "\n" + self.layout.layout["state_properties"]["reward_symbol"]
        res = EMPTY_RES
        for reward_model in self.model.rewards:
            if self.model.supports_actions():
                if a in s.available_actions():
                    reward = reward_model.get_state_action_reward(s, a)
                else:
                    reward = None
            else:
                reward = reward_model.get_state_reward(s)
            if reward is not None and not (
                not self.layout.layout["state_properties"]["show_zero_rewards"]
                and reward == 0
            ):
                res += f"\t{reward_model.name}: {self._format_number(reward)}"
        if res == EMPTY_RES:
            return ""
        return res

    def _create_state_properties(self, state: stormvogel.model.State):
        """Generates visualization properties for a given state in the model.

        This method assembles the visual representation of a state, including
        its label, group assignment, color (based on model checking results),
        and other textual annotations like rewards and observations. These
        properties are used when constructing the `ModelGraph` for visualization.

        If result coloring is enabled and a model checking result is available,
        the state's color is interpolated between two configured colors based
        on the state's result value relative to the maximum result.

        Args:
            state (stormvogel.model.State): The model state for which to generate properties.

        Returns:
            dict: A dictionary containing the state's visualization properties with keys:
                - `"label"`: A string representing the state label, including ID (if enabled),
                  labels, rewards, result, and observations.
                - `"group"`: A string identifying the layout group this state belongs to.
                - `"color"`: A blended RGB color string or None, based on result values.
        """
        res = self._format_result(state)
        observations = self._format_observations(state)
        rewards = self._format_rewards(state, stormvogel.model.EmptyAction)
        group = self._group_state(state, "states")
        id_label_part = (
            f"{state.id}\n"
            if self.layout.layout["state_properties"]["show_ids"]
            else ""
        )

        color = None

        result_colors = self.layout.layout["results"]["result_colors"]
        if result_colors and self.result is not None:
            result = self.result.get_result_of_state(state)
            max_result = self.result.maximum_result()
            if isinstance(result, (int, float, Fraction)) and isinstance(
                max_result, (int, float, Fraction)
            ):
                color1 = self.layout.layout["results"]["max_result_color"]
                color2 = self.layout.layout["results"]["min_result_color"]
                factor = result / max_result if max_result != 0 else 0
                color = blend_colors(color1, color2, float(factor))
        properties = {
            "label": id_label_part
            + ",".join(state.labels)
            + rewards
            + res
            + observations,
            "group": group,
            "color": color,
        }
        return properties

    def _create_action_properties(
        self, state: stormvogel.model.State, action: stormvogel.model.Action
    ) -> dict:
        """Generates visualization properties for a given action in the model.

        This method creates a label for the action, optionally including any
        associated reward, and includes the original `Action` object for use
        in the visualization or interaction.

        Args:
            state (stormvogel.model.State): The state from which the action originates.
            action (stormvogel.model.Action): The action being evaluated.

        Returns:
            dict: A dictionary containing the action's visualization properties:
                - `"label"`: A string combining the action's labels and reward.
                - `"model_action"`: The original `Action` object.
        """
        reward = self._format_rewards(self.model.get_state_by_id(state.id), action)

        properties = {"label": ",".join(action.labels) + reward, "model_action": action}
        return properties

    def _create_transition_properties(self, state, action, next_state) -> dict:
        """Generates visualization properties for a transition between states.

        This method finds the transition probability for a specific state-action–
        next-state triplet and formats it as a label. If the transition exists,
        the formatted probability is included; otherwise, an empty dictionary is returned.

        Args:
            state (stormvogel.model.State): The source state of the transition.
            action (stormvogel.model.Action): The action taken from the source state.
            next_state (stormvogel.model.State): The target state of the transition.

        Returns:
            dict: A dictionary containing the transition's visualization properties:
                - `"label"`: The formatted transition probability, if found.
                  Otherwise, the dictionary will be empty.
        """
        properties = dict()
        choices = state.get_outgoing_choice(action)
        if choices is None:
            return properties
        for prob, target in choices:
            if next_state.id == target.id:
                properties["label"] = self._format_number(prob)
                return properties
        return properties


class JSVisualization(VisualizationBase):
    """Handles visualization of a Model using a Network from stormvogel.network."""

    EXTRA_PIXELS: int = 20  # To prevent the scroll bar around the Network.

    def __init__(
        self,
        model: stormvogel.model.Model,
        name: str | None = None,
        result: stormvogel.result.Result | None = None,
        scheduler: stormvogel.result.Scheduler | None = None,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        output: widgets.Output | None = None,
        debug_output: widgets.Output = widgets.Output(),
        use_iframe: bool = False,
        do_init_server: bool = True,
        max_states: int = 1000,
        max_physics_states: int = 500,
        spam: widgets.Output = widgets.Output(),
    ) -> None:
        """Create and show a visualization of a Model using a visjs Network
        Args:
            model (Model): The stormvogel model to be displayed.
            name (str): Used to name the iframe. ONLY SPECIFY IF YOU KNOW WHAT YOU ARE DOING. You should never create two networks with the same name, they might clash.
            result (Result, optional): A result associatied with the model.
                The results are displayed as numbers on a state. Enable the layout editor for options.
                If this result has a scheduler, then the scheduled actions will have a different color etc. based on the layout
            scheduler (Scheduler, optional): The scheduled actions will have a different color etc. based on the layout
                If both result and scheduler are set, then scheduler takes precedence.
            layout (Layout): Layout used for the visualization.
            output (widgets.Output): The output widget in which the network is rendered.
                Whether this widget is also displayed automatically depends on do_display.
            debug_output (widgets.Output): Output widget that can be used to debug interactive features.
            use_iframe (bool): Set to true iff you want to use an iframe. Defaults to False.
            do_init_server (bool): Set to true iff you want to initialize the server. Defaults to True.
            max_states (int): If the model has more states, then the network is not displayed.
            max_physics_states (int): If the model has more states, then physics are disabled.
        """
        super().__init__(model, layout, result, scheduler)
        self.initial_state_id = model.get_initial_state().id
        if output is None:
            self.output = widgets.Output()
        else:
            self.output = output
        self.debug_output: widgets.Output = debug_output
        self.spam = spam
        with self.output:
            ipd.display(self.spam)

        # vis stuff
        self.name: str = name or random_word(10)
        self.use_iframe: bool = use_iframe
        self.max_states: int = max_states
        self.max_physics_states: int = max_physics_states

        self.do_init_server: bool = do_init_server
        self.network_wrapper: str = ""  # Use this for javascript injection.
        if self.use_iframe:
            self.network_wrapper: str = (
                f"document.getElementById('{self.name}').contentWindow.nw_{self.name}"
            )
        else:
            self.network_wrapper: str = f"nw_{self.name}"
        self.new_nodes_hidden: bool = False
        if do_init_server:
            self.server: stormvogel.communication_server.CommunicationServer = (
                stormvogel.communication_server.initialize_server()
            )

    def _generate_node_js(self) -> str:
        """Generate the required js script for node definition"""
        node_js = ""
        for node in self.G.nodes():
            node_attr = self.G.nodes[node]
            label = node_attr.get("label", None)
            color = "black"
            group = None
            layout_group_color = None
            match self.G.nodes[node]["type"]:
                case NodeType.STATE:
                    group = self._group_state(
                        self.model.get_state_by_id(node), "states"
                    )
                    layout_group_color = self.layout.layout["groups"].get(group)
                case NodeType.ACTION:
                    in_edges = list(self.G.in_edges(node))
                    assert (
                        len(in_edges) == 1
                    ), "An action node should only have a single incoming edge"
                    state, _ = in_edges[0]
                    group = self._group_action(
                        state, self.G.nodes[node]["model_action"], "actions"
                    )
                    layout_group_color = self.layout.layout["groups"].get(group)
            if layout_group_color is not None:
                color = layout_group_color.get("color", {"background": color}).get(
                    "background"
                )
                # HACK: This is necessary for the selection highlighting to work
                # and should not be here
                color = None
            current = "{ id: " + str(node)
            if label is not None:
                current += f", label: `{label}`"
            if group is not None:
                current += f', group: "{group}"'
            if node in self.layout.layout["positions"]:
                current += (
                    f", x: {self.layout.layout['positions'][node]['x']}, "
                    f"y: {self.layout.layout['positions'][node]['y']}"
                )
            if self.layout.layout["misc"]["explore"] and node != self.initial_state_id:
                current += ", hidden: true"
                current += ", physics: false"
            if color is not None:
                current += f', color: "{color}"'
            current += " },\n"
            node_js += current
        return node_js

    def _generate_edge_js(self) -> str:
        """Generate the required js script for edge definition"""
        edge_js = ""
        # preprocess scheduled actions
        scheduled_action_nodes = []
        for node in self.G.nodes:
            if self.G.nodes[node]["type"] == NodeType.ACTION:
                continue
            for _, target in self.G.out_edges(node):
                if self.G.nodes[target]["type"] == NodeType.STATE:
                    continue
                action = self.G.nodes[target]["model_action"]
                group = self._group_action(node, action, "actions")
                if group == "scheduled_actions":
                    scheduled_action_nodes.append(target)

        for from_, to in self.G.edges():
            edge_attr = self.G.edges[(from_, to)]
            # TODO: in order for the layout to have an effect color should be
            # self.layout.layout["edges"]["color"]["color"]
            # however this breaks the highlighing on selection
            color = None
            scheduled_color = self.layout.layout["groups"].get(
                "scheduled_actions", {"color": {"border": color}}
            )["color"]["border"]
            match [self.G.nodes[from_]["type"], self.G.nodes[to]["type"]]:
                case [NodeType.STATE, NodeType.ACTION]:
                    if to in scheduled_action_nodes:
                        color = scheduled_color
                case [NodeType.ACTION, NodeType.STATE]:
                    if from_ in scheduled_action_nodes:
                        color = scheduled_color
            label = edge_attr.get("label", None)
            current = "{ from: " + str(from_) + ", to: " + str(to)
            if label is not None:
                current += f', label: "{label}"'
            if color is not None:
                current += f', color: "{color}"'
            if self.layout.layout["misc"]["explore"]:
                current += ", hidden: true"
                current += ", physics: false"
            current += " },\n"
            edge_js += current
        return edge_js

    def _get_options(self) -> str:
        """Returns the current layout configuration as a JSON-formatted string.

        This method serializes the layout dictionary used for visualization
        into a readable JSON format with indentation for clarity.

        Returns:
            str: A pretty-printed JSON string representing the current layout configuration.
        """
        return json.dumps(self.layout.layout, indent=2)

    def set_options(self, options: str) -> None:
        """Sets the layout configuration from a JSON-formatted string.

        This method replaces the current layout with a new one defined by the
        given JSON string. It should be called only before visualization is rendered
        (i.e., before calling `show()`), as it reinitializes the layout.

        Args:
            options (str): A JSON-formatted string representing the layout configuration.
        """
        options_dict = json.loads(options)
        self.layout = stormvogel.layout.Layout(layout_dict=options_dict)

    def generate_html(self) -> str:
        """Generate an html page representing the current state of the `ModelGraph`"""
        return stormvogel.html_generation.generate_html(
            self._generate_node_js(),
            self._generate_edge_js(),
            self._get_options(),
            self.name,
            self.layout.layout["misc"].get("width", 800),
            self.layout.layout["misc"].get("height", 600),
        )

    def generate_iframe(self) -> str:
        """Generate an iframe for the network, using the html."""
        return f"""
          <iframe
                id="{self.name}"
                width="{self.layout.layout["misc"].get("width", 800) + self.EXTRA_PIXELS}"
                height="{self.layout.layout["misc"].get("height", 600) + self.EXTRA_PIXELS}"
                sandbox="allow-scripts allow-same-origin"
                frameborder="0"
                srcdoc="{html.escape(self.generate_html())}"
                border:none !important;
                allowfullscreen webkitallowfullscreen mozallowfullscreen
          ></iframe>"""

    def generate_svg(self, width: int = 800) -> str:
        """Generate an svg rendering for the network."""
        js = f"RETURN({self.network_wrapper}.getSvg());"
        res = self.server.result(js)[1:-1]
        unescaped = res.encode("utf-8").decode("unicode_escape")
        scaled = autoscale_svg(unescaped, width)
        return scaled

    def enable_exploration_mode(self, initial_node_id: int):
        """Enables exploration mode starting from a specified initial state.

        This method activates interactive exploration mode in the visualization
        and sets the starting point for exploration to the given state ID.
        `show()` needs to be called after this method is executed to have an effect.

        Args:
            initial_node_id (int): The ID of the state from which exploration should begin.
        """
        self.initial_state_id = initial_node_id
        self.layout.set_value(["misc", "explore"], True)

    def get_positions(self) -> dict:
        """Get the current positions of the nodes on the canvas. Returns empty dict if unsucessful.
        Example result: {0: {"x": 5, "y": 10}}"""
        if self.server is None:
            with self.debug_output:
                logging.warning(
                    "Server not initialized. Could not retrieve position data."
                )
            raise TimeoutError("Server not initialized.")
        try:
            positions: dict = json.loads(
                self.server.result(
                    f"""RETURN({self.network_wrapper}.network.getPositions())"""
                )
            )
            return {int(k): v for k, v in positions.items()}
        except TimeoutError:
            with self.debug_output:
                logging.warning("Timed out. Could not retrieve position data.")
            raise TimeoutError("Timed out. Could not retrieve position data.")

    def show(self, hidden: bool = False) -> None:
        with self.output:  ## If there was already a rendered network, clear it.
            ipd.clear_output()
        if len(self.model.get_states()) > self.max_states:
            with self.output:
                print(
                    f"This model has more than {self.max_states} states. If you want to proceed, set max_states to a higher value."
                    f"This is to prevent the browser from crashing, be careful."
                )
            return
        if len(self.model.get_states()) > self.max_physics_states:
            with self.output:
                print(
                    f"This model has more than {self.max_physics_states} states. If you want physics, set max_physics_states to a higher value."
                    f"Physics are disabled to prevent the browser from crashing, be careful."
                )
            self.layout.layout["physics"] = False
            self.layout.copy_settings()
        if self.use_iframe:
            iframe = self.generate_iframe()
        else:
            iframe = self.generate_html()
        with self.output:  # Display the iframe within the Output.
            ipd.clear_output()
            ipd.display(ipd.HTML(iframe))
        if not hidden:
            ipd.display(self.output)
        with self.debug_output:
            logging.info("Called show")

    def update(self) -> None:
        """Updates the visualization with the current layout options.

        This method sends updated layout configuration to the frontend visualization
        by injecting JavaScript code. It is typically used to reflect changes made
        to layout settings after the initial rendering.

        Note:
            This should be called after modifying layout properties if the visualization
            has already been shown, to apply those changes interactively.
        """
        js = f"""{self.network_wrapper}.network.setOptions({self._get_options()});"""
        ipd.display(ipd.Javascript(js))

    def set_node_color(self, node_id: int, color: str | None) -> None:
        """Sets the color of a specific node in the visualization.

        This method updates the visual appearance of a node by changing its color
        via JavaScript. It only takes effect once the network has been fully loaded
        in the frontend.

        Args:
            node_id (int): The ID of the node whose color should be changed.
            color (str | None): The color to apply (e.g., "#ff0000" for red).
                If None, the node color is reset or cleared.

        Note:
            This function requires that the visualization is already rendered
            (i.e., `show()` has been called and completed asynchronously).
        """
        if color is None:
            color = "null"
        else:
            color = f'"{color}"'

        js = f"""{self.network_wrapper}.setNodeColor({node_id}, {color});"""
        ipd.display(ipd.Javascript(js))
        ipd.clear_output()

    def highlight_state(self, state_id: int, color: str | None = "red"):
        """Highlights a single state in the model by changing its color.

        This method changes the color of the specified state node in the visualization.
        Pass `None` to reset or clear the highlight.

        Args:
            state_id (int): The ID of the state to highlight.
            color (str | None, optional): The color to use for highlighting (e.g., "red", "#00ff00").
                Defaults to "red".

        Raises:
            AssertionError: If the state ID does not exist in the model graph.
        """
        assert self.G.nodes.get(state_id) is not None, "State id not in ModelGraph"
        self.set_node_color(state_id, color)

    def highlight_action(
        self, state_id: int, action: stormvogel.model.Action, color: str | None = "red"
    ):
        """Highlights a single action in the model by changing its color.

        This method changes the color of the node representing a specific action
        taken from a given state. Pass `None` to remove the highlight.

        Args:
            state_id (int): The ID of the state from which the action originates.
            action (stormvogel.model.Action): The action to highlight.
            color (str | None, optional): The color to use for highlighting. Defaults to "red".

        Warns:
            UserWarning: If the specified (state, action) pair is not found in the model graph.
        """
        try:
            nt_id = self.G.state_action_id_map[(state_id, action)]
            self.set_node_color(nt_id, color)
        except KeyError:
            warnings.warn(
                "Tried to highlight an action that is not present in this model."
            )

    def highlight_state_set(self, state_ids: set[int], color: str | None = "blue"):
        """Highlights a set of states in the model by changing their color.

        Iterates over each state ID in the provided set and applies the given
        color. Pass `None` to clear highlighting for all specified states.

        Args:
            state_ids (set[int]): A set of state IDs to highlight.
            color (str | None, optional): The color to apply. Defaults to "blue".
        """
        for s_id in state_ids:
            self.set_node_color(s_id, color)

    def highlight_action_set(
        self,
        state_action_set: set[tuple[int, stormvogel.model.Action]],
        color: str = "red",
    ):
        """Highlights a set of actions in the model by changing their color.

        Applies the specified color to all (state, action) pairs in the given set.
        Pass `None` as the color to clear the current highlighting.

        Args:
            state_action_set (set[tuple[int, stormvogel.model.Action]]): A set of
                (state ID, action) pairs to highlight.
            color (str, optional): The color to apply. Defaults to "red".
        """
        for s_id, a in state_action_set:
            self.highlight_action(s_id, a, color)

    def highlight_decomposition(
        self,
        decomp: list[tuple[set[int], set[tuple[int, stormvogel.model.Action]]]],
        colors: list[str] | None = None,
    ):
        """Highlight a set of tuples of (states and actions) in the model by changing their color.
        Args:
            decomp: A list of tuples (states, actions)
            colors (optional): A list of colors for the decompossitions. Random colors are picked by default."""
        for n, v in enumerate(decomp):
            if colors is None:
                color = random_color()
            else:
                color = colors[n]
            self.highlight_state_set(v[0], color)
            self.highlight_action_set(v[1], color)

    def clear_highlighting(self):
        """Clear all highlighting that is currently active, returing all states to their original colors."""
        for s_id, _ in self.model:
            self.set_node_color(s_id, None)
        for a_id in self.G.state_action_id_map.values():
            self.set_node_color(a_id, None)

    def highlight_path(
        self,
        path: simulator.Path,
        color: str,
        delay: float = 1,
        clear: bool = True,
    ) -> None:
        """Highlight the path that is provided as an argument in the model.
        Args:
            path (simulator.Path): The path to highlight.
            color (str | None): The color that the highlighted states should get (in HTML color standard).
                Set to None, in order to clear existing highlights on this path.
            delay (float): If not None, there will be a pause of a specified time before highlighting the next state in the path.
            clear (bool): Clear the highlighting of a state after it was highlighted. Only works if delay is not None.
                This is particularly useful for highlighting paths with loops."""
        seq = path.to_state_action_sequence()
        for i, v in enumerate(seq):
            if isinstance(v, stormvogel.model.State):
                self.set_node_color(v.id, color)
                sleep(delay)
                if clear:
                    self.set_node_color(v.id, None)
            elif (
                isinstance(v, stormvogel.model.Action)
                and (seq[i - 1].id, v) in self.G.state_action_id_map
            ):
                node_id = self.G.state_action_id_map[seq[i - 1].id, v]
                self.set_node_color(node_id, color)
                sleep(delay)
                if clear:
                    self.set_node_color(node_id, None)

    def export(self, output_format: str, filename: str = "export") -> None:
        """
        Export the visualization to your preferred output format.
        The appropriate file extension will be added automatically.

        Parameters:
            output_format (str): Desired export format.
            filename (str): Base name for the exported file.

        Supported output formats (not case-sensitive):

            "HTML"    → An interactive .html file (e.g., draggable nodes)
            "IFrame"  → Exports as an <iframe> wrapped HTML in a .html file
            "PDF"     → Exports to .pdf (via conversion from SVG)
            "SVG"     → Exports to .svg vector image
        """
        output_format = output_format.lower()
        filename_base = pathlib.Path(filename).with_suffix(
            ""
        )  # remove extension if present

        if output_format == "html":
            html = self.generate_html()
            (filename_base.with_suffix(".html")).write_text(html, encoding="utf-8")

        elif output_format == "iframe":
            iframe = self.generate_iframe()
            (filename_base.with_suffix(".html")).write_text(iframe, encoding="utf-8")

        elif output_format == "svg":
            svg = self.generate_svg()
            (filename_base.with_suffix(".svg")).write_text(svg, encoding="utf-8")

        elif output_format == "pdf":
            svg = self.generate_svg()
            cairosvg.svg2pdf(
                bytestring=svg.encode("utf-8"), write_to=filename_base.name + ".pdf"
            )

        elif output_format == "latex":
            svg = self.generate_svg()
            # Create the 'export' folder if it doesn't exist
            export_folder = pathlib.Path(filename_base)
            export_folder.mkdir(parents=True, exist_ok=True)
            pdf_filename = filename_base.with_suffix(".pdf")
            # Convert SVG to PDF
            cairosvg.svg2pdf(
                bytestring=svg.encode("utf-8"),
                write_to=str(export_folder / pdf_filename),
            )

            # Create the LaTeX file
            latex_content = f"""\\documentclass{{article}}
\\usepackage{{graphicx}}
\\begin{{document}}
\\begin{{figure}}[h!]
\\centering
\\includegraphics[width=\\textwidth]{{{pdf_filename.name}}}
\\caption{{Generated using Stormvogel. TODO insert citing instructions}}
\\end{{figure}}
\\end{{document}}
"""
            # Write the LaTeX code to a .tex file
            (export_folder / filename_base.with_suffix(".tex")).write_text(
                latex_content, encoding="utf-8"
            )

        else:
            raise RuntimeError(f"Export format not supported: {output_format}")


class MplVisualization(VisualizationBase):
    """Matplotlib-based visualization for Stormvogel models.

    Extends the base visualization class to render the model, results, and
    scheduler using Matplotlib. Supports interactive features like node
    highlighting and custom hover behavior.

    This class manages figure creation, state and edge highlighting, and
    optionally allows interaction callbacks when hovering over nodes.

    Args:
        model (stormvogel.model.Model): The model to visualize.
        layout (stormvogel.layout.Layout, optional): Layout configuration for
            the visualization. Defaults to `stormvogel.layout.DEFAULT()`.
        result (stormvogel.result.Result | None, optional): The result of a model
            checking operation, which may contain a scheduler. Defaults to None.
        scheduler (stormvogel.result.Scheduler | None, optional): Explicit scheduler
            defining actions to take in each state. Defaults to None.
        title (str | None, optional): Title of the visualization figure. Defaults to None.
        interactive (bool, optional): Whether to enable interactive features such
            as node hover callbacks. Defaults to False.
        hover_node (Callable | None, optional): Callback function invoked when
            hovering over nodes. Receives parameters
            `(PathCollection, PathCollection, MouseEvent, Axes)`. Defaults to None.
    """

    def __init__(
        self,
        model: stormvogel.model.Model,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        result: stormvogel.result.Result | None = None,
        scheduler: stormvogel.result.Scheduler | None = None,
        title: str | None = None,
        interactive: bool = False,
        hover_node: Callable[[PathCollection, PathCollection, MouseEvent, Axes], None]
        | None = None,
    ):
        super().__init__(model, layout, result, scheduler)
        self.title = title or ""
        self.interactive = interactive
        self.hover_node = hover_node
        self._highlights: dict[int, str] = dict()
        self._edge_highlights: dict[tuple[int, int], str] = dict()
        self._fig = None
        if self.scheduler is not None:
            self.highlight_scheduler(self.scheduler)

    def highlight_state(self, state: stormvogel.model.State | int, color: str = "red"):
        """Highlights a state node in the visualization by setting its color.

        Args:
            state (stormvogel.model.State | int): The state object or state ID to highlight.
            color (str, optional): The color to apply. Defaults to "red".

        Raises:
            AssertionError: If the state node is not present in the model graph.
        """
        if isinstance(state, stormvogel.model.State):
            state = state.id
        node = state
        assert node in self.G.nodes, f"Node {node} not in graph"
        self._highlights[state] = color

    def highlight_action(
        self,
        state: stormvogel.model.State | int,
        action: stormvogel.model.Action,
        color: str = "red",
    ):
        """Highlights an action node associated with a state by setting its color.

        Args:
            state (stormvogel.model.State | int): The state object or state ID from which the action originates.
            action (stormvogel.model.Action): The action to highlight.
            color (str, optional): The color to apply. Defaults to "red".

        Raises:
            AssertionError: If the state node is not present in the model graph.
        """
        if isinstance(state, stormvogel.model.State):
            state = state.id
        state_node = state
        assert state_node in self.G.nodes, f"Node {state_node} not in graph"
        action_node = self.G.state_action_id_map[state_node, action]
        self._highlights[action_node] = color

    def highlight_edge(self, from_: int, to_: int, color: str = "red"):
        """Highlights an edge between two nodes by setting its color.

        Args:
            from_ (int): The source node ID of the edge.
            to_ (int): The target node ID of the edge.
            color (str, optional): The color to apply. Defaults to "red".
        """
        self._edge_highlights[from_, to_] = color

    def clear_highlighting(self):
        """Clear all nodes that are marked for highlighting in the visualization"""
        self._highlights.clear()
        self._edge_highlights.clear()

    def highlight_scheduler(self, scheduler: stormvogel.result.Scheduler):
        """Highlights states, actions, and edges according to the given scheduler.

        Applies a specific highlight color defined by the layout to all states and
        actions specified by the scheduler’s taken actions, as well as the edges connecting them.
        The color is derived from the layout’s configured group colors for scheduled actions.

        Args:
            scheduler (stormvogel.result.Scheduler): The scheduler containing
                state-action mappings to highlight.
        """
        default_color = self.layout.layout["edges"]["color"]["color"]
        color = self.layout.layout["groups"].get(
            "scheduled_actions", {"color": {"border": default_color}}
        )["color"]["border"]
        for state_id, taken_action in scheduler.taken_actions.items():
            self.highlight_state(state_id, color)
            if taken_action == stormvogel.model.EmptyAction:
                continue
            action_node = self.G.state_action_id_map[(state_id, taken_action)]
            self.highlight_action(state_id, taken_action, color)
            self.highlight_edge(state_id, action_node, color)
            for start, end in self.G.out_edges(action_node):
                self.highlight_edge(start, end, color)

    def add_to_ax(
        self,
        ax,
        node_size: int | dict[int, int] = 300,
        node_kwargs: dict[str, Any] | None = None,
        edge_kwargs: dict[str, Any] | None = None,
    ):
        """Draws the model graph onto a given Matplotlib Axes.

        This method renders nodes and edges of the model graph on the
        provided Matplotlib `ax` object. It uses layout positions, colors from
        the current layout configuration, and any highlights applied to nodes
        or edges. Node sizes can be specified either as a fixed integer or as a
        dictionary mapping node IDs to sizes.

        Args:
            ax (matplotlib.axes.Axes): The Matplotlib axes to draw the graph on.
            node_size (int or dict[int, int], optional): Size(s) of nodes. If an
                int is given, all nodes are drawn with that size. If a dictionary,
                it must provide sizes for all nodes. Defaults to 300.
            node_kwargs (dict[str, Any], optional): Additional keyword arguments
                passed to `nx.draw_networkx_nodes()`. Defaults to None.
            edge_kwargs (dict[str, Any], optional): Additional keyword arguments
                passed to `nx.draw_networkx_edges()`. Defaults to None.

        Returns:
            tuple: A tuple `(nodes, edges)` where `nodes` is the
                `matplotlib.collections.PathCollection` of drawn nodes and `edges`
                is the `matplotlib.collections.LineCollection` of drawn edges.
        """
        if node_kwargs is None:
            node_kwargs = dict()
        if edge_kwargs is None:
            edge_kwargs = dict()

        if isinstance(node_size, dict):
            assert all(
                [n in node_size for n in self.G.nodes]
            ), "Not all nodes are present in node_size"
        else:
            node_size = {n: node_size for n in self.G.nodes}

        # fetch the colors from the layout
        node_colors = dict()
        for node in self.G.nodes:
            color = "black"
            layout_group_color = None
            match self.G.nodes[node]["type"]:
                case NodeType.STATE:
                    group = self._group_state(
                        self.model.get_state_by_id(node), "states"
                    )
                    layout_group_color = self.layout.layout["groups"].get(group)
                case NodeType.ACTION:
                    in_edges = list(self.G.in_edges(node))
                    assert (
                        len(in_edges) == 1
                    ), "An action node should only have a single incoming edge"
                    state, _ = in_edges[0]
                    group = self._group_action(
                        state, self.G.nodes[node]["model_action"], "actions"
                    )
                    layout_group_color = self.layout.layout["groups"].get(group)
            if layout_group_color is not None:
                color = layout_group_color.get("color", {"background": color}).get(
                    "background"
                )
            node_colors[node] = color

        edge_colors = dict()
        for edge in self.G.edges:
            edge_colors[edge] = self.layout.layout["edges"]["color"]["color"]

        # Now add highlights
        for node, color in self._highlights.items():
            node_colors[node] = color
        for edge, color in self._edge_highlights.items():
            edge_colors[edge] = color

        pos = {
            node: np.array((pos["x"], pos["y"]))
            for node, pos in self.layout.layout["positions"].items()
        }
        if len(pos) != len(self.G.nodes):
            pos = nx.random_layout(self.G)
        edges = nx.draw_networkx_edges(
            self.G,
            pos=pos,
            ax=ax,
            edge_color=[edge_colors[e] for e in self.G.edges],  # type: ignore
            **edge_kwargs,
        )
        nodes = nx.draw_networkx_nodes(
            self.G,
            pos=pos,
            ax=ax,
            node_color=[node_colors[n] for n in self.G.nodes],  # type: ignore
            node_size=[node_size[n] for n in self.G.nodes],  # type: ignore
            **node_kwargs,
        )
        return nodes, edges

    def update(
        self,
        node_size: int | dict[int, int] = 300,
        node_kwargs: dict[str, Any] | None = None,
        edge_kwargs: dict[str, Any] | None = None,
    ):
        """Updates or creates the Matplotlib figure displaying the model graph.

        This method sets up the figure size based on layout settings, draws the
        graph nodes and edges using `add_to_ax`, and applies highlights and titles.
        If `interactive` is enabled, it connects a hover event handler to update
        the plot title dynamically when the mouse moves over nodes.

        Args:
            node_size (int or Sequence[int], optional): Size(s) for the nodes. Can be
                a single integer or a sequence of sizes for each node. Defaults to 300.
            node_kwargs (dict[str, Any], optional): Additional keyword arguments
                passed to `nx.draw_networkx_nodes()`. Defaults to None.
            edge_kwargs (dict[str, Any], optional): Additional keyword arguments
                passed to `nx.draw_networkx_edges()`. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The Matplotlib figure object containing the visualization.
        """
        px = 1 / plt.rcParams["figure.dpi"]
        figsize = (
            self.layout.layout["misc"].get("width", 800) * px,
            self.layout.layout["misc"].get("height", 600) * px,
        )
        if self._fig is None:
            self._fig, ax = plt.subplots(figsize=figsize)
        else:
            w, h = figsize
            self._fig.set_figwidth(w)
            self._fig.set_figheight(h)
            ax = self._fig.gca()
            ax.clear()
        fig = self._fig
        nodes, edges = self.add_to_ax(
            ax,
            node_size=node_size,
            node_kwargs=node_kwargs,
            edge_kwargs=edge_kwargs,
        )
        ax.set_title(self.title)
        node_list = list(self.G.nodes)

        def update_title(ind):
            idx = ind["ind"][0]
            node = node_list[idx]
            node_attr = self.G.nodes[node]
            ax.set_title(f"{node_attr['type'].name}: {node_attr['label']}")

        def hover(event):
            cont, ind = nodes.contains(event)
            if self.hover_node is not None:
                self.hover_node(nodes, edges, event, ax)
            else:
                if cont:
                    update_title(ind)
                else:
                    ax.set_title(self.title)
            fig.canvas.draw_idle()

        if self.interactive:
            fig.canvas.mpl_connect("motion_notify_event", hover)
        return fig

    def show(
        self,
    ):
        self.update()
        plt.show()
