"""Contains the code responsible for representing the structure of a model as a graph"""

from collections.abc import Callable
from enum import Enum
from typing import Any, Self
from networkx import DiGraph

from stormvogel.model import Action, EmptyAction, Model, State, Value

ACTION_ID_OFFSET = 10**10


class NodeType(Enum):
    STATE = 0
    ACTION = 1
    UNDEFINED = 2


class ModelGraph(DiGraph):
    """A directed graph describing the structure of a model.

    States and actions (except EmptyActions) are represented as nodes in the graph.
    All outgoing edges of a state node describe the available actions for that state.
    The outgoing edges from an action describe the possible next states (possible choices)
    and hold the probability of each transition as a node attribute.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__(self)
        self._state_action_id_map: dict[tuple[int, Action], int] = dict()
        self._next_action_id = ACTION_ID_OFFSET

    @property
    def state_action_id_map(self):
        """
        dict[tuple[int, Action], int]: A mapping from (state ID, Action) pairs to internal action node IDs.

        This dictionary allows lookup of the internal graph node ID corresponding to a given (state, action) pair.
        It is used to associate actions with their respective nodes in the graph structure.
        """
        return self._state_action_id_map

    def add_state(self, state: State | int, **attr):
        """
        Adds a state node to the graph.

        If a `State` object is provided, its `id` is used as the node identifier.
        Additional attributes can be assigned to the state node via keyword arguments.

        Args:
            state (State | int): The state to add, either as a `State` object or its ID.
            **attr: Arbitrary keyword arguments representing attributes to associate with the state node.
        """
        if isinstance(state, State):
            state = state.id
        self.add_node(state, type=NodeType.STATE, **attr)

    def add_action(self, state: State | int, action: Action, **action_attr):
        """
        Adds an action node to the graph and connects it to a given state.

        The action node is uniquely identified and linked from the source state.
        The action is skipped if it is an `EmptyAction`. Additional attributes
        can be assigned to the action node via keyword arguments.

        Args:
            state (State | int): The source state (either a `State` object or its ID) from which the action originates.
            action (Action): The action to add.
            **action_attr: Arbitrary keyword arguments representing attributes to associate with the action node.

        Raises:
            AssertionError: If the source state is not already in the graph.
        """
        if isinstance(state, State):
            state = state.id
        assert state in self.nodes, f"State {state} not in graph yet"
        if action == EmptyAction:
            return
        self.add_node(self._next_action_id, type=NodeType.ACTION, **action_attr)
        self.add_edge(state, self._next_action_id)
        self._state_action_id_map[(state, action)] = self._next_action_id
        self._next_action_id += 1

    def add_transition(
        self,
        state: State | int,
        action: Action,
        next_state: int | State,
        probability: Value,
        **attr,
    ):
        """
        Adds a transition to the graph with an associated probability.

        For non-empty actions, this adds an edge from the action node to the target state.
        For `EmptyAction`, the edge is added directly from the source state to the target state.
        Additional attributes can be provided for the transition edge.

        Args:
            state (State | int): The source state (either a `State` object or its ID).
            action (Action): The action that causes the transition.
            next_state (int | State): The target state reached by the transition.
            probability (Value): The probability associated with the transition.
            **attr: Arbitrary keyword arguments representing attributes to associate with the transition edge.

        Raises:
            AssertionError: If the source state or target state is not in the graph,
                or if the action node is missing (for non-empty actions).
        """
        if isinstance(state, State):
            state = state.id
        if isinstance(next_state, State):
            next_state = next_state.id
        action_id = self._state_action_id_map.get((state, action), None)
        assert state in self.nodes
        assert action == EmptyAction or action_id is not None
        assert next_state in self.nodes
        if action == EmptyAction:
            self.add_edge(state, next_state, probability=probability, **attr)
        else:
            self.add_edge(action_id, next_state, probability=probability, **attr)

    @classmethod
    def from_model(
        cls,
        model: Model,
        state_properties: Callable[[State], dict[str, Any]] | None = None,
        action_properties: Callable[[State, Action], dict[str, Any]] | None = None,
        transition_properties: Callable[[State, Action, State], dict[str, Any]]
        | None = None,
    ) -> Self:
        """
        Constructs a directed graph representation of a Markov Decision Process (MDP) from a model instance.

        This method initializes the graph from the provided `model` by adding all states, actions, and choices.
        Optional callbacks allow customization of properties for states, actions, and choices.

        Args:
            model (Model): The MDP model containing states and choices.
            state_properties (Callable[[State], dict[str, Any]], optional): A callable that returns a dictionary
                of properties for a given state. Defaults to None.
            action_properties (Callable[[State, Action], dict[str, Any]], optional): A callable that returns a
                dictionary of properties for a given action from a state. Defaults to None.
            transition_properties (Callable[[State, Action, State], dict[str, Any]], optional): A callable that
                returns a dictionary of properties for a transition from a source state via an action to a target
                state. Defaults to None.

        Returns:
            Self: An instance of the graph populated with the states, actions, and choices from the model.

        Examples:
            >>> import stormvogel.examples as examples
            >>> mdp = examples.create_lion_mdp()
            >>> G = ModelGraph.from_model(mdp, state_properties = lambda s: {"labels": s.labels})
            >>> G.nodes[mdp.get_initial_state().id]
            {'type': <NodeType.STATE: 0>, 'labels': ['init']}
        """
        G = cls()
        for _, state in model:
            props = dict()
            if state_properties is not None:
                props = state_properties(state)
            G.add_state(state, **props)

        for state_id, transition in model.choices.items():
            state = model.get_state_by_id(state_id)
            for action, branch in transition.transition.items():
                action_props = dict()
                if action_properties is not None:
                    action_props = action_properties(state, action)
                G.add_action(state_id, action, **action_props)
                for probability, target in branch:
                    transition_props = dict()
                    if transition_properties is not None:
                        transition_props = transition_properties(state, action, target)
                    G.add_transition(
                        state_id,
                        action,
                        target,
                        probability=probability,
                        **transition_props,
                    )
        return G
