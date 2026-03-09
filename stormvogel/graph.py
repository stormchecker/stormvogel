"""Contains the code responsible for representing the structure of a model as a graph."""

from base64 import urlsafe_b64encode
from collections.abc import Callable
from enum import Enum
from typing import Any, Self
import networkx

from stormvogel.model import Action, EmptyAction, Model, State, Value


def node_key(node: State | tuple[State, Action]) -> str:
    """Convert a graph node to a unique, JS/JSON-safe string key.

    State nodes map to their UUID string.
    Action ``(State, Action)`` tuple nodes map to
    ``{state_uuid}__{base64url(action_label)}``.

    The action label is base64url-encoded so that the result is safe for
    embedding inside JS/JSON double-quoted strings and free of separator
    collisions (the UUID hex+hyphen alphabet and the base64url alphabet never
    produce the ``__`` bigram).

    :param node: A state or ``(State, Action)`` tuple to convert.
    :returns: A unique string key for the node.
    :raises TypeError: If *node* is not a ``State`` or ``(State, Action)`` tuple.
    """
    if isinstance(node, State):
        return str(node.state_id)
    elif isinstance(node, tuple) and len(node) == 2:
        state, action = node
        raw_label = action.label or ""
        encoded = urlsafe_b64encode(raw_label.encode()).decode()
        return f"{state.state_id}__{encoded}"
    raise TypeError(f"node_key expects State or (State, Action), got {type(node)}")


class NodeType(Enum):
    STATE = 0
    ACTION = 1
    UNDEFINED = 2


class ModelGraph(networkx.DiGraph):
    """A directed networkx graph describing the structure of a model.

    States and actions (except EmptyActions) are represented as nodes in the graph.
    All outgoing edges of a state node describe the available actions for that state.
    The outgoing edges from an action describe the possible next states (possible choices)
    and hold the probability of each transition as a node attribute.
    """

    def add_state(self, state: State, **attr):
        """Add a state node to the graph.

        :param state: The state to add.
        :param \\**attr: Arbitrary keyword arguments representing attributes to associate with the state node.
        """
        self.add_node(state, type=NodeType.STATE, **attr)

    def add_action(self, state: State, action: Action, **action_attr):
        """Add an action node to the graph and connect it to a given state.

        The action node is uniquely identified and linked from the source state.
        The action is skipped if it is an ``EmptyAction``.

        :param state: The source state from which the action originates.
        :param action: The action to add.
        :param \\**action_attr: Arbitrary keyword arguments representing attributes to associate with the action node.
        :raises AssertionError: If the source state is not already in the graph.
        """
        assert state in self.nodes, f"State {state} not in graph yet"
        if action == EmptyAction:
            return
        self.add_node((state, action), type=NodeType.ACTION, **action_attr)
        self.add_edge(state, (state, action))

    def add_transition(
        self,
        state: State,
        action: Action,
        next_state: State,
        probability: Value,
        **attr,
    ) -> None:
        """Add a transition to the graph with an associated probability.

        For non-empty actions, this adds an edge from the action node to the target
        state. For ``EmptyAction``, the edge is added directly from the source state
        to the target state.

        :param state: The source state.
        :param action: The action that causes the transition.
        :param next_state: The target state reached by the transition.
        :param probability: The probability associated with the transition.
        :param \\**attr: Arbitrary keyword arguments representing attributes to associate with the transition edge.
        :raises AssertionError: If the source state or target state is not in the graph,
            or if the action node is missing (for non-empty actions).
        """

        assert state in self.nodes, f"State {state} not in graph."
        assert (
            action == EmptyAction or (state, action) in self.nodes
        ), f"Action node for action {action} in state {state} not in graph."
        assert next_state in self.nodes, f"Next state {next_state} not in graph."

        if action == EmptyAction:
            self.add_edge(state, next_state, probability=probability, **attr)
        else:
            self.add_edge((state, action), next_state, probability=probability, **attr)

    @classmethod
    def from_model(
        cls,
        model: Model,
        state_properties: Callable[[State], dict[str, Any]] | None = None,
        action_properties: Callable[[State, Action], dict[str, Any]] | None = None,
        transition_properties: (
            Callable[[State, Action, State], dict[str, Any]] | None
        ) = None,
    ) -> Self:
        """Construct a directed graph representation of a model.

        Initialize the graph from the provided *model* by adding all states,
        actions, and transitions. Optional callbacks allow customization of
        properties for states, actions, and transitions.

        :param model: The model containing states and choices.
        :param state_properties: A callable that returns a dictionary of properties
            for a given state.
        :param action_properties: A callable that returns a dictionary of properties
            for a given action from a state.
        :param transition_properties: A callable that returns a dictionary of properties
            for a transition from a source state via an action to a target state.
        :returns: An instance of the graph populated with states, actions, and
            transitions from the model.

        .. doctest::

            >>> import stormvogel.examples as examples
            >>> mdp = examples.create_lion_mdp()
            >>> G = ModelGraph.from_model(mdp, state_properties = lambda s: {"labels": s.labels})
            >>> G.nodes[mdp.initial_state]
            {'type': <NodeType.STATE: 0>, 'labels': ['init']}
        """
        G = cls()
        for state in model:
            props = dict()
            if state_properties is not None:
                props = state_properties(state)
            G.add_state(state, **props)

        for state, choice in model.choices.items():
            for action, branch in choice:
                action_props = dict()
                if action_properties is not None:
                    action_props = action_properties(state, action)
                G.add_action(state, action, **action_props)
                for probability, target in branch:
                    transition_props = dict()
                    if transition_properties is not None:
                        transition_props = transition_properties(state, action, target)
                    G.add_transition(
                        state,
                        action,
                        target,
                        probability=probability,
                        **transition_props,
                    )
        return G
