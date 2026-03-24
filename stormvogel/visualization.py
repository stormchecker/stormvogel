"""Contains the code responsible for model visualization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import stormvogel.model
from stormvogel.model.value import value_to_string

import stormvogel.layout
import stormvogel.result
import stormvogel.html_generation

if TYPE_CHECKING:
    # Type-only imports
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.collections import PathCollection
    from matplotlib.axes import Axes
    import ipywidgets as widgets
    from . import simulator


def _escape_js_template(s: str) -> str:
    """Escape a Python string for safe embedding inside a JS template literal (backticks)."""
    return s.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")


def _escape_js_dquote(s: str) -> str:
    """Escape a Python string for safe embedding inside a JS double-quoted string."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


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

    :param model: The MDP model to visualize.
    :param layout: Layout settings for the visualization.
    :param result: The result of a model checking operation, which may contain a scheduler.
    :param scheduler: An explicit scheduler defining actions to take in each state.

    :ivar model: The MDP model being visualized.
    :ivar layout: The layout configuration for the visualization.
    :ivar result: The result of a model checking operation.
    :ivar scheduler: A scheduler representing a path through the model.
    :ivar G: The internal graph structure representing the model.
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
        # If a scheduler was not set explicitly, but a result was set, then take the scheduler from the results.
        if self.scheduler is None:
            if self.result is not None:
                self.scheduler = self.result.scheduler

        # Set "scheduler" as an active group iff it is present.
        if self.scheduler is not None:
            self.layout.add_active_group("scheduled_actions")
        else:  # Otherwise, disable it
            self.layout.remove_active_group("scheduled_actions")
        self.recreate()

    @staticmethod
    def _und(x: str) -> str:
        """Replace spaces by underscores."""
        return x.replace(" ", "_")

    @staticmethod
    def _random_word(k: int) -> str:
        """Generate a random word of length *k*."""
        import random
        import string

        return "".join(random.choices(string.ascii_letters, k=k))

    @staticmethod
    def _random_color() -> str:
        """Return a random HEX color."""
        import random

        return "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])

    @staticmethod
    def _blend_colors(c1: str, c2: str, factor: float) -> str:
        """Blend two colors in HEX format (#RRGGBB).

        :param c1: Color 1 in HEX format #RRGGBB.
        :param c2: Color 2 in HEX format #RRGGBB.
        :param factor: The fraction of the resulting color that should come from *c1*.
        """
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

    def recreate(self):
        """Recreate the ModelGraph and set the edit groups."""
        from .graph import ModelGraph

        self.G = ModelGraph.from_model(
            self.model,
            state_properties=self._create_state_properties,
            action_properties=self._create_action_properties,
            transition_properties=self._create_transition_properties,
        )
        underscored_labels = set(map(self._und, self.model.state_labels.keys()))
        possible_groups = underscored_labels.union(
            {"states", "actions", "scheduled_actions"}
        )
        self.layout.set_possible_groups(possible_groups)

    def _format_number(self, n: stormvogel.model.Value) -> str:
        """Format a number using the settings specified in the layout object."""
        if self.layout.layout["numbers"]["visible"] is False:
            return ""
        return value_to_string(
            n,
            self.layout.layout["numbers"]["fractions"],
            self.layout.layout["numbers"]["digits"],
            self.layout.layout["numbers"]["denominator_limit"],
        )

    def _format_result(self, s: stormvogel.model.State) -> str:
        """Create a string that shows the result for this state.

        Start with a newline. If results are not enabled, return the empty string."""
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
        """Create a string that shows the observation for this state (for POMDPs).

        Start with a newline."""
        if (
            not self.model.supports_observations()
            or not self.layout.layout["state_properties"]["show_observations"]
        ):
            return ""
        elif isinstance(s.observation, list):
            string = ""
            for prob, obs in s.observation:
                string += (
                    "\n"
                    + self.layout.layout["state_properties"]["observation_symbol"]
                    + " "
                    + str(obs.display())
                    + ": "
                    + self._format_number(prob)
                )
            return string
        else:
            return (
                "\n"
                + self.layout.layout["state_properties"]["observation_symbol"]
                + " "
                + str(s.observation)
            )

    def _group_state(self, s: stormvogel.model.State, default: str) -> str:
        """Determine the group of a state.

        The group is the label of *s* that has the highest priority,
        as specified by the user under edit_groups.
        """
        und_labels = set(map(self._und, s.labels))
        res = list(
            filter(
                lambda x: x in und_labels, self.layout.layout["edit_groups"]["groups"]
            )
        )
        return self._und(res[0]) if res != [] else default

    def _group_action(
        self, state: stormvogel.model.State, a: stormvogel.model.Action, default: str
    ) -> str:
        """Return the group of this action. Only relevant for scheduling."""
        # Put the action in the group scheduled_actions if appropriate.
        if self.scheduler is None:
            return default

        action = self.scheduler.get_action_at_state(state)
        return "scheduled_actions" if a == action else default

    def _format_rewards(
        self, s: stormvogel.model.State, a: stormvogel.model.Action
    ) -> str:
        """Create a string that contains the state reward. Actions no longer have individual rewards."""
        if not self.layout.layout["state_properties"]["show_rewards"]:
            return ""
        if a != stormvogel.model.EmptyAction:
            return ""

        EMPTY_RES = "\n" + self.layout.layout["state_properties"]["reward_symbol"]
        res = EMPTY_RES
        for reward_model in self.model.rewards:
            reward = reward_model.get_state_reward(s)
            if reward is not None and not (
                not self.layout.layout["state_properties"]["show_zero_rewards"]
                and reward == 0
            ):
                res += f"\t{reward_model.name}: {self._format_number(reward)}"
        if res == EMPTY_RES:
            return ""
        return res

    def _create_state_properties(self, state: stormvogel.model.State) -> dict:
        """Generate visualization properties for a given state in the model.

        Create the visual representation of a state, including its label,
        group assignment, color (based on model checking results), and other
        textual annotations like rewards and observations. These properties
        are used when constructing the ``ModelGraph`` for visualization.

        If result coloring is enabled and a model checking result is available,
        the state's color is interpolated between two configured colors based
        on the state's result value relative to the maximum result.

        :param state: The model state for which to generate properties.
        :returns: A dictionary containing the state's visualization properties with keys
            ``"label"``, ``"group"``, and ``"color"``.
        """
        from fractions import Fraction

        res = self._format_result(state)
        observations = self._format_observations(state)
        rewards = self._format_rewards(state, stormvogel.model.EmptyAction)
        group = self._group_state(state, "states")
        id_label_part = (
            f"{state}\n" if self.layout.layout["state_properties"]["show_ids"] else ""
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
                color = self._blend_colors(color1, color2, float(factor))
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
        """Generate visualization properties for a given action in the model.

        Create a label for the action, optionally including any associated
        reward, and include the original ``Action`` object for use in the
        visualization or interaction.

        :param state: The state from which the action originates.
        :param action: The action being evaluated.
        :returns: A dictionary containing the action's visualization properties
            with keys ``"label"`` and ``"model_action"``.
        """
        reward = self._format_rewards(state, action)

        properties = {"label": (action.label or "") + reward, "model_action": action}
        return properties

    def _create_transition_properties(self, state, action, next_state) -> dict:
        """Generate visualization properties for a transition between states.

        Find the transition probability for a specific state-action-next-state
        triplet and format it as a label. If the transition exists, the
        formatted probability is included; otherwise, an empty dictionary is returned.

        :param state: The source state of the transition.
        :param action: The action taken from the source state.
        :param next_state: The target state of the transition.
        :returns: A dictionary containing the transition's visualization properties.
        """
        properties = dict()
        transitions = state.get_outgoing_transitions(action)
        if transitions is None:
            return properties
        for prob, target in transitions:
            if next_state == target:
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
        output: widgets.Output | None = None,  # type: ignore[name-defined]
        debug_output: widgets.Output | None = None,  # type: ignore[name-defined]
        use_iframe: bool = False,
        do_init_server: bool = True,
        max_states: int = 1000,
        max_physics_states: int = 500,
    ) -> None:
        """Create and show a visualization of a Model using a visjs Network.

        :param model: The stormvogel model to be displayed.
        :param name: Used to name the iframe. Only specify if you know what you are
            doing. You should never create two networks with the same name, they might clash.
        :param result: A result associated with the model. The results are displayed as
            numbers on a state. Enable the layout editor for options. If this result has a
            scheduler, then the scheduled actions will have a different color etc. based on
            the layout.
        :param scheduler: The scheduled actions will have a different color etc. based on
            the layout. If both result and scheduler are set, then scheduler takes precedence.
        :param layout: Layout used for the visualization.
        :param output: The output widget in which the network is rendered.
        :param debug_output: Output widget that can be used to debug interactive features.
        :param use_iframe: Set to ``True`` to use an iframe.
        :param do_init_server: Set to ``True`` to initialize the server.
        :param max_states: If the model has more states, the network is not displayed.
        :param max_physics_states: If the model has more states, physics are disabled.
        """
        import ipywidgets as widgets  # local, heavy
        import os

        super().__init__(model, layout, result, scheduler)

        self.initial_state = model.initial_state

        if output is None:
            self.output = widgets.Output()
        else:
            self.output = output
        if debug_output is None:
            self.debug_output: widgets.Output = widgets.Output()
        else:
            self.debug_output = debug_output

        # vis stuff
        self.name: str = name or self._random_word(10)
        self.use_iframe: bool = use_iframe
        self.max_states: int = max_states
        self.max_physics_states: int = max_physics_states

        self.do_init_server: bool = do_init_server
        self.network_wrapper: str = ""  # Use this for javascript injection.
        if self.use_iframe:
            self.network_wrapper = (
                f"document.getElementById('{self.name}').contentWindow.nw_{self.name}"
            )
        else:
            self.network_wrapper = f"nw_{self.name}"
        self.new_nodes_hidden: bool = False
        # If the user wants to initialize the server, and we are not building the documentation.
        if do_init_server and os.environ.get("DOCUMENTATION", "0").lower() != "1":
            import stormvogel.communication_server  # ensure submodule is loaded

            self.server: stormvogel.communication_server.CommunicationServer = (
                stormvogel.communication_server.initialize_server()
            )
        else:
            self.server = None  # type: ignore[assignment]

    def _generate_node_js(self) -> str:
        """Generate the required JS script for node definition."""
        from .graph import NodeType, node_key

        node_js = ""
        for node in self.G.nodes():
            node_attr = self.G.nodes[node]
            label = node_attr.get("label", None)
            color = node_attr.get("color", None)
            group = None
            match self.G.nodes[node]["type"]:
                case NodeType.STATE:
                    group = self._group_state(node, "states")
                case NodeType.ACTION:
                    in_edges = list(self.G.in_edges(node))
                    assert (
                        len(in_edges) == 1
                    ), "An action node should only have a single incoming edge"
                    state, _ = in_edges[0]
                    group = self._group_action(
                        state, self.G.nodes[node]["model_action"], "actions"
                    )
                    # layout_group_color = self.layout.layout["groups"].get(group)
            # if layout_group_color is not None:
            #     color = layout_group_color.get("color", {"background": color}).get(
            #         "background"
            #     )
            #     # HACK: This is necessary for the selection highlighting to work
            #     # and should not be here
            #     color = None
            # TODO ask Ossmoss what this code was supposed to do exactly. It only seems to waste time?
            key = node_key(node)
            current = f'{{ id: "{key}"'
            if label is not None:
                current += f", label: `{_escape_js_template(label)}`"
            if group is not None:
                current += f', group: "{group}"'
            if key in self.layout.layout["positions"]:
                current += (
                    f", x: {self.layout.layout['positions'][key]['x']}, "
                    f"y: {self.layout.layout['positions'][key]['y']}"
                )
            if self.layout.layout["misc"]["explore"] and node != self.initial_state:
                current += ", hidden: true"
                current += ", physics: false"
            if color is not None:
                current += f', color: "{color}"'
            current += " },\n"
            node_js += current
        return node_js

    def _generate_edge_js(self) -> str:
        """Generate the required JS script for edge definition."""
        from .graph import NodeType, node_key

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
            current = f'{{ from: "{node_key(from_)}", to: "{node_key(to)}"'
            if label is not None:
                current += f', label: "{_escape_js_dquote(label)}"'
            if color is not None:
                current += f', color: "{color}"'
            if self.layout.layout["misc"]["explore"]:
                current += ", hidden: true"
                current += ", physics: false"
            current += " },\n"
            edge_js += current
        return edge_js

    def _get_options(self) -> str:
        """Return the current layout configuration as a JSON-formatted string.

        Serialize the layout dictionary used for visualization into a readable
        JSON format with indentation for clarity.

        :returns: A pretty-printed JSON string representing the current layout configuration.
        """
        import json

        return json.dumps(self.layout.layout, indent=2)

    def set_options(self, options: str) -> None:
        """Set the layout configuration from a JSON-formatted string.

        Replace the current layout with a new one defined by the given JSON
        string. Call only before visualization is rendered (i.e., before
        calling ``show()``), as it reinitializes the layout.

        :param options: A JSON-formatted string representing the layout configuration.
        """
        import json

        options_dict = json.loads(options)
        self.layout = stormvogel.layout.Layout(layout_dict=options_dict)

    def generate_html(self) -> str:
        """Generate an HTML page representing the current state of the ``ModelGraph``."""
        return stormvogel.html_generation.generate_html(
            self._generate_node_js(),
            self._generate_edge_js(),
            self._get_options(),
            self.name,
        )

    def generate_iframe(self) -> str:
        """Generate an iframe for the network, using the HTML."""
        import html as _html

        return f"""
          <iframe
                id="{self.name}"
                width="{self.layout.layout["misc"].get("width", 800) + self.EXTRA_PIXELS}"
                height="{self.layout.layout["misc"].get("height", 600) + self.EXTRA_PIXELS}"
                sandbox="allow-scripts allow-same-origin"
                frameborder="0"
                srcdoc="{_html.escape(self.generate_html())}"
                border:none !important;
                allowfullscreen webkitallowfullscreen mozallowfullscreen
          ></iframe>"""

    def generate_svg(self, width: int = 800) -> str:
        """Generate an SVG rendering for the network."""
        from stormvogel.autoscale_svg import autoscale_svg

        js = f"RETURN({self.network_wrapper}.getSvg());"
        res = self.server.result(js)[1:-1]
        unescaped = res.encode("utf-8").decode("unicode_escape")
        scaled = autoscale_svg(unescaped, width)
        return scaled

    def enable_exploration_mode(self, initial_state: stormvogel.model.State):
        """Enable exploration mode starting from a specified initial state.

        Activate interactive exploration mode in the visualization and set the
        starting point for exploration to the given state.
        ``show()`` needs to be called after this method to have an effect.

        :param initial_state: The state from which exploration should begin.
        """
        self.initial_state = initial_state
        self.layout.set_value(["misc", "explore"], True)

    def get_positions(self) -> dict[str, dict[str, int]]:
        """Get the current positions of the nodes on the canvas.

        :returns: A dict mapping node keys to position dicts, e.g.
            ``{"uuid-string": {"x": 5, "y": 10}}``. Return empty dict if unsuccessful.
        :raises TimeoutError: If the server is not initialized or communication times out.
        """
        import json
        import logging

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
            return positions
        except TimeoutError:
            with self.debug_output:
                logging.warning("Timed out. Could not retrieve position data.")
            raise TimeoutError("Timed out. Could not retrieve position data.")

    def show(self, hidden: bool = False) -> None:
        import logging
        import IPython.display as ipd

        with self.output:  # If there was already a rendered network, clear it.
            ipd.clear_output()
        if len(self.model.states) > self.max_states:
            with self.output:
                print(
                    f"This model has more than {self.max_states} states. If you want to proceed, set max_states to a higher value."
                    f"This is to prevent the browser from crashing, be careful."
                )
            return
        if len(self.model.states) > self.max_physics_states:
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
        """Update the visualization with the current layout options.

        Send updated layout configuration to the frontend visualization by
        injecting JavaScript code. Typically used to reflect changes made to
        layout settings after the initial rendering.

        .. note::
            Call this after modifying layout properties if the visualization has
            already been shown, to apply those changes interactively.
        """
        import IPython.display as ipd

        js = f"""{self.network_wrapper}.network.setOptions({self._get_options()});"""
        ipd.display(ipd.Javascript(js))

    def set_node_color(
        self,
        obj: (
            stormvogel.model.State
            | tuple[stormvogel.model.State, stormvogel.model.Action]
        ),
        color: str | None,
    ) -> None:
        """Set the color of a specific node in the visualization.

        Update the visual appearance of a node by changing its color via
        JavaScript. Only takes effect once the network has been fully loaded
        in the frontend.

        :param obj: The state or (state, action) pair whose node color should be changed.
        :param color: The color to apply (e.g., ``"#ff0000"`` for red).
            If ``None``, the node color is reset.

        .. note::
            This requires that the visualization is already rendered
            (i.e., ``show()`` has been called and completed asynchronously).
        """
        from .graph import node_key
        import IPython.display as ipd

        if color is None:
            color = "null"
        else:
            color = f'"{ color }"'

        js = f"""{self.network_wrapper}.setNodeColor("{node_key(obj)}", {color});"""
        ipd.display(ipd.Javascript(js))
        ipd.clear_output()

    def highlight_state(self, state: stormvogel.model.State, color: str | None = "red"):
        """Highlight a single state in the model by changing its color.

        Change the color of the specified state node in the visualization.
        Pass ``None`` to reset or clear the highlight.

        :param state: The state to highlight.
        :param color: The color to use for highlighting (e.g., ``"red"``, ``"#00ff00"``).
        :raises AssertionError: If the state does not exist in the model graph.
        """
        assert self.G.nodes.get(state) is not None, "State id not in ModelGraph"
        self.set_node_color(state, color)

    def highlight_action(
        self,
        state: stormvogel.model.State,
        action: stormvogel.model.Action,
        color: str | None = "red",
    ):
        """Highlight a single action in the model by changing its color.

        Change the color of the node representing a specific action taken from
        a given state. Pass ``None`` to remove the highlight.

        :param state: The state from which the action originates.
        :param action: The action to highlight.
        :param color: The color to use for highlighting.
        :raises UserWarning: If the specified (state, action) pair is not found in the model graph.
        """
        import warnings

        try:
            self.set_node_color((state, action), color)
        except KeyError:
            warnings.warn(
                "Tried to highlight an action that is not present in this model."
            )

    def highlight_state_set(
        self, states: set[stormvogel.model.State], color: str | None = "blue"
    ):
        """Highlight a set of states in the model by changing their color.

        Iterate over each state in the provided set and apply the given color.
        Pass ``None`` to clear highlighting for all specified states.

        :param states: A set of states to highlight.
        :param color: The color to apply.
        """
        for state in states:
            self.set_node_color(state, color)

    def highlight_action_set(
        self,
        state_action_set: set[tuple[stormvogel.model.State, stormvogel.model.Action]],
        color: str = "red",
    ):
        """Highlight a set of actions in the model by changing their color.

        Apply the specified color to all (state, action) pairs in the given set.
        Pass ``None`` as the color to clear the current highlighting.

        :param state_action_set: A set of (state, action) pairs to highlight.
        :param color: The color to apply.
        """
        for state, a in state_action_set:
            self.highlight_action(state, a, color)

    def highlight_decomposition(
        self,
        decomp: list[
            tuple[
                set[stormvogel.model.State],
                set[tuple[stormvogel.model.State, stormvogel.model.Action]],
            ]
        ],
        colors: list[str] | None = None,
    ):
        """Highlight a set of tuples of (states and actions) in the model by changing their color.

        :param decomp: A list of tuples (states, actions).
        :param colors: A list of colors for the decompositions. Random colors are picked by default.
        """
        for n, v in enumerate(decomp):
            if colors is None:
                color = self._random_color()
            else:
                color = colors[n]
            self.highlight_state_set(v[0], color)
            self.highlight_action_set(v[1], color)

    def clear_highlighting(self):
        """Clear all highlighting that is currently active, returning all nodes to their original colors."""
        for state in self.model:
            self.set_node_color(state, None)
        for a_id in self.G.nodes:
            self.set_node_color(a_id, None)

    def highlight_path(
        self,
        path: simulator.Path,  # type: ignore[name-defined]
        color: str,
        delay: float = 1,
        clear: bool = True,
    ) -> None:
        """Highlight the path that is provided as an argument in the model.

        :param path: The path to highlight.
        :param color: The color that the highlighted states should get (in HTML color standard).
            Set to ``None`` to clear existing highlights on this path.
        :param delay: Pause for the specified time before highlighting the next state in the path.
        :param clear: Clear the highlighting of a state after it was highlighted. Only works if
            delay is not ``None``. Particularly useful for highlighting paths with loops.
        """
        from time import sleep

        seq = path.to_state_action_sequence()
        for i, v in enumerate(seq):
            if isinstance(v, stormvogel.model.State):
                self.set_node_color(v, color)
                sleep(delay)
                if clear:
                    self.set_node_color(v, None)
            elif isinstance(v, stormvogel.model.Action):
                last_state = seq[i - 1]
                if isinstance(last_state, stormvogel.model.State):
                    node_id = (last_state, v)
                    self.set_node_color(node_id, color)
                    sleep(delay)
                    if clear:
                        self.set_node_color(node_id, None)

    def export(self, output_format: str, filename: str = "export") -> None:
        """Export the visualization to the preferred output format.

        The appropriate file extension will be added automatically.

        :param output_format: Desired export format. Supported values (not case-sensitive):
            ``"HTML"``, ``"IFrame"``, ``"PDF"``, ``"SVG"``, ``"LaTeX"``.
        :param filename: Base name for the exported file.
        :raises RuntimeError: If the export format is not supported.
        """
        import pathlib

        output_format = output_format.lower()
        filename_base = pathlib.Path(filename).with_suffix(
            ""
        )  # remove extension if present

        if output_format == "html":
            html_txt = self.generate_html()
            (filename_base.with_suffix(".html")).write_text(html_txt, encoding="utf-8")

        elif output_format == "iframe":
            iframe = self.generate_iframe()
            (filename_base.with_suffix(".html")).write_text(iframe, encoding="utf-8")

        elif output_format == "svg":
            svg = self.generate_svg()
            (filename_base.with_suffix(".svg")).write_text(svg, encoding="utf-8")

        elif output_format == "pdf":
            svg = self.generate_svg()
            import cairosvg  # local, heavy

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
            import cairosvg  # local, heavy

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

    Extend the base visualization class to render the model, results, and
    scheduler using Matplotlib. Support interactive features like node
    highlighting and custom hover behavior.

    :param model: The model to visualize.
    :param layout: Layout configuration for the visualization.
    :param result: The result of a model checking operation, which may contain a scheduler.
    :param scheduler: Explicit scheduler defining actions to take in each state.
    :param title: Title of the visualization figure.
    :param interactive: Whether to enable interactive features such as node hover callbacks.
    :param hover_node: Callback function invoked when hovering over nodes. Receives parameters
        ``(PathCollection, PathCollection, MouseEvent, Axes)``.
    """

    def __init__(
        self,
        model: stormvogel.model.Model,
        layout: stormvogel.layout.Layout = stormvogel.layout.DEFAULT(),
        result: stormvogel.result.Result | None = None,
        scheduler: stormvogel.result.Scheduler | None = None,
        title: str | None = None,
        interactive: bool = False,
        hover_node: (
            Callable[[PathCollection, PathCollection, MouseEvent, Axes], None] | None
        ) = None,
    ):
        super().__init__(model, layout, result, scheduler)
        self.title = title or ""
        self.interactive = interactive
        self.hover_node = hover_node
        self._highlights: dict[
            stormvogel.model.State
            | tuple[stormvogel.model.State, stormvogel.model.Action],
            str,
        ] = dict()
        self._edge_highlights: dict[
            tuple[
                stormvogel.model.State
                | tuple[stormvogel.model.State, stormvogel.model.Action],
                stormvogel.model.State
                | tuple[stormvogel.model.State, stormvogel.model.Action],
            ],
            str,
        ] = dict()
        self._fig = None
        if self.scheduler is not None:
            self.highlight_scheduler(self.scheduler)

    def highlight_state(
        self, state: stormvogel.model.State, color: str = "red"
    ) -> None:
        """Highlight a state node in the visualization by setting its color.

        :param state: The state to highlight.
        :param color: The color to apply.
        :raises AssertionError: If the state node is not present in the model graph.
        """
        assert state in self.G.nodes, f"Node {state} not in graph"
        self._highlights[state] = color

    def highlight_action(
        self,
        state: stormvogel.model.State,
        action: stormvogel.model.Action,
        color: str = "red",
    ) -> None:
        """Highlight an action node associated with a state by setting its color.

        :param state: The state from which the action originates.
        :param action: The action to highlight.
        :param color: The color to apply.
        :raises AssertionError: If the state node is not present in the model graph.
        """
        assert state in self.G.nodes, f"Node {state} not in graph"
        action_node = (state, action)
        self._highlights[action_node] = color

    def highlight_edge(
        self,
        from_: (
            stormvogel.model.State
            | tuple[stormvogel.model.State, stormvogel.model.Action]
        ),
        to_: (
            stormvogel.model.State
            | tuple[stormvogel.model.State, stormvogel.model.Action]
        ),
        color: str = "red",
    ) -> None:
        """Highlight an edge between two nodes by setting its color.

        :param from_: The source node of the edge.
        :param to_: The target node of the edge.
        :param color: The color to apply.
        """
        self._edge_highlights[from_, to_] = color

    def clear_highlighting(self) -> None:
        """Clear all nodes that are marked for highlighting in the visualization."""
        self._highlights.clear()
        self._edge_highlights.clear()

    def highlight_scheduler(self, scheduler: stormvogel.result.Scheduler) -> None:
        """Highlight states, actions, and edges according to the given scheduler.

        Apply a specific highlight color defined by the layout to all states and
        actions specified by the scheduler's taken actions, as well as the edges
        connecting them.

        :param scheduler: The scheduler containing state-action mappings to highlight.
        """
        default_color = self.layout.layout["edges"]["color"]["color"]
        color = self.layout.layout["groups"].get(
            "scheduled_actions", {"color": {"border": default_color}}
        )["color"]["border"]
        for state_id, taken_action in scheduler.taken_actions.items():
            self.highlight_state(state_id, color)
            if taken_action == stormvogel.model.EmptyAction:
                continue
            action_node = (state_id, taken_action)
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
    ) -> tuple[Any, Any]:
        """Draw the model graph onto a given Matplotlib Axes.

        Render nodes and edges of the model graph on the provided Matplotlib
        ``ax`` object. Use layout positions, colors from the current layout
        configuration, and any highlights applied to nodes or edges.

        :param ax: The Matplotlib axes to draw the graph on.
        :param node_size: Size(s) of nodes. If an int is given, all nodes are drawn
            with that size. If a dictionary, it must provide sizes for all nodes.
        :param node_kwargs: Additional keyword arguments passed to
            ``nx.draw_networkx_nodes()``.
        :param edge_kwargs: Additional keyword arguments passed to
            ``nx.draw_networkx_edges()``.
        :returns: A tuple ``(nodes, edges)`` where ``nodes`` is the
            ``PathCollection`` of drawn nodes and ``edges`` is the
            ``LineCollection`` of drawn edges.
        """
        import numpy as np
        import networkx as nx
        from .graph import NodeType, node_key

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
                    group = self._group_state(node, "states")
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

        pos = {}
        for nx_node in self.G.nodes:
            key = node_key(nx_node)
            if key in self.layout.layout["positions"]:
                node_pos = self.layout.layout["positions"][key]
                pos[nx_node] = np.array((node_pos["x"], node_pos["y"]))
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
        """Update or create the Matplotlib figure displaying the model graph.

        Set up the figure size based on layout settings, draw the graph nodes
        and edges using ``add_to_ax``, and apply highlights and titles. If
        ``interactive`` is enabled, connect a hover event handler to update the
        plot title dynamically when the mouse moves over nodes.

        :param node_size: Size(s) for the nodes. Can be a single integer or a
            dictionary mapping nodes to sizes.
        :param node_kwargs: Additional keyword arguments passed to
            ``nx.draw_networkx_nodes()``.
        :param edge_kwargs: Additional keyword arguments passed to
            ``nx.draw_networkx_edges()``.
        :returns: The Matplotlib figure object containing the visualization.
        """
        import matplotlib.pyplot as plt

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
                self.hover_node(nodes, edges, event, ax)  # type: ignore
            else:
                if cont:
                    update_title(ind)
                else:
                    ax.set_title(self.title)
            fig.canvas.draw_idle()

        if self.interactive:
            fig.canvas.mpl_connect("motion_notify_event", hover)
        return fig

    def show(self) -> None:
        import matplotlib.pyplot as plt

        self.update()
        plt.show()
