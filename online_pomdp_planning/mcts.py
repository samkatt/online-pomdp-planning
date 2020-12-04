"""Implementation of Monte-Carlo tree search"""
from __future__ import annotations

from copy import deepcopy
from math import log, sqrt
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

from online_pomdp_planning.types import Action, Belief, Observation, Simulator, State


class ActionNode:
    """A decision node in the MCTS tree

    This node maps observations to children nodes. In addition it stores
    statistics, such as the expected Q-value associated with this node.
    """

    def __init__(
        self,
        initial_statistics: Any,
        parent: ObservationNode,
    ):
        """Initializes the action node with given statistics

        :param initial_statistics: anything you would like to store
        :type initial_statistics: Any
        :param parent: the parent node in the tree
        :type parent: ObservationNode
        """
        self.parent = parent
        self.observation_nodes: Dict[Observation, ObservationNode] = {}
        self.stats = initial_statistics

    def add_observation_node(
        self, observation: Observation, observation_node: ObservationNode
    ):
        """Adds a node to the children of `self`

        Raises `AssertionError` if:
            - the parent of the added node is not self
            - `observation` is not already associated with a child node

        :param observation: the observation associated with child
        :type observation: Observation
        :param observation_node: the new child node
        :type observation_node: ObservationNode
        """
        assert observation not in self.observation_nodes
        assert observation_node.parent == self
        self.observation_nodes[observation] = observation_node

    def observation_node(self, observation: Observation) -> ObservationNode:
        """The child-node associated with given observation `o`

        Raises `KeyError` if `action` is not associated with a child node

        :param observation:
        :type observation: Observation
        :return: child node
        :rtype: ObservationNode
        """
        return self.observation_nodes[observation]


class ObservationNode:
    """A chance/observation node in the MCTS tree

    This node representation action-observation history, ending up in a
    particular observation. It has no statistics. If it has no parent, then it
    is the root node. This node maps actions to children nodes.
    """

    def __init__(self, parent: Optional[ActionNode] = None):  # pylint: disable=E1136
        """Initiates an observation node with given parent

        :param parent: if no parent is given, this must be the root node
        :type parent: Optional[ActionNode]
        """
        self.parent = parent
        self.action_nodes: Dict[Action, ActionNode] = {}

    @property
    def child_stats(self) -> Dict[Action, Any]:
        """Returns a mapping from actions to statistics (shortcut)

        :return: action -> stats mapping
        :rtype: Dict[Action, Any]
        """
        return {a: n.stats for a, n in self.action_nodes.items()}

    def add_action_node(self, action: Action, node: ActionNode):
        """Adds a `action` -> `node` mapping to children

        Raises `AssertionError` if:
            - the parent of the added node is not self
            - `action` is not already associated with a child node


        :param action: action associated with new child
        :type action: Action
        :param node: child node
        :type node: ActionNode
        """
        assert action not in self.action_nodes
        assert node.parent == self
        self.action_nodes[action] = node

    def action_node(self, action: Action) -> ActionNode:
        """Get child node associated with `action`

        Raises `KeyError` if `action` is not associated with a child node

        :param action:
        :type action: Action
        :return: returns child node
        :rtype: ActionNode
        """
        return self.action_nodes[action]


class LeafSelection(Protocol):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    def __call__(
        self, s: State, node: ObservationNode
    ) -> Tuple[ActionNode, State, Observation, bool, Any]:
        """Traverses through the tree and picks a leaf

        :type s: State
        :param node: (root) node
        :type node: ObservationNode
        :return: leaf node, state and obs, terminal flag and input to :py:class:`BackPropagation`
        :rtype: Tuple[ActionNode, State, Observation, bool, Any]
        """


def ucb(q: float, n: int, n_total: int, ucb_constant: float) -> float:
    """
    Returns the upper confidence bound of Q given statistics

    :param q: the q-value
    :type q: float
    :param n: the number of times this action has been chosen
    :type n: int
    :param n_total: the total number of times any action has been chosen
    :type n_total: int
    :param ucb_constant: the exploration constant
    :type ucb_constant: float
    :return: q + ucb_constant * sqrt(log(n_total) / n)
    :rtype: float
    """
    assert n >= 0 and n_total >= 0 and ucb_constant >= 0
    if n == 0:
        return float("inf")

    return q + ucb_constant * sqrt(log(n_total) / n)


def select_with_ucb(stats: Dict[Action, Any], ucb_constant: float) -> Action:
    """
    Select an action using UCB with given exploration constant

    Assumes `stats` contains entries for "n" and "qval"

    :param stats: the statistics: Action -> Dict where Dict is {"qval": float, "n": int}
    :type stats: Dict[Action, Any]
    :param ucb_constant: the exploration constant used in UCB
    :type ucb_constant: float
    :return: the action with the highest upper confidence bound
    :rtype: Action
    """
    total_visits = sum(s["n"] for s in stats.values())  # type: ignore
    actions_to_ucb = {
        a: ucb(s["qval"], s["n"], total_visits, ucb_constant)  # type: ignore
        for a, s in stats.items()
    }

    return max(actions_to_ucb, key=actions_to_ucb.get)  # type: ignore


def ucb_select_leaf(
    state: State, node: ObservationNode, sim: Simulator, ucb_constant: float
) -> Tuple[ActionNode, State, Observation, bool, List[float]]:
    """Tree policy according to UCB: ucb to pick actions, simulator to generate observations

    Implements :py:class:`LeafSelection`, and returns a list of rewards as output

    Picks action nodes according to :py:class:`select_with_ucb`, uses `sim` to
    generate observations and pick the respective nodes

    Note: has the potential to be more general and accept any selection method
    (either action node or observation node) as input

    :param state: the root state
    :type state: State
    :param node: the root node
    :type node: ObservationNode
    :param sim: a POMDP simulator
    :type sim: Simulator
    :param ucb_constant: exploration constant of UCB
    :type ucb_constant: float
    :return: leaf node, state, observation, terminal flag and list of rewards
    :rtype: Tuple[ActionNode, State, Observation, List[float]]
    """
    list_of_rewards: List[float] = []

    while True:

        action = select_with_ucb(node.child_stats, ucb_constant)
        state, obs, reward, terminal_flag = sim(state, action)

        list_of_rewards.append(reward)

        if terminal_flag:
            break

        try:
            node = node.action_node(action).observation_node(obs)
        except KeyError:
            # reached a leaf
            break

    return node.action_node(action), state, obs, terminal_flag, list_of_rewards


class Expansion(Protocol):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    def __call__(self, o: Observation, a: ActionNode) -> ObservationNode:
        """Expands a leaf node

        :param o: observation that resulted in leaf
        :type o: Observation
        :param a: action that resulted in leaf
        :type a: ActionNode
        :return: expanded node
        :rtype: ObservationNode
        """


def expand_node_with_all_actions(
    o: Observation,
    action_node: ActionNode,
    actions: Iterator[Action],
    init_stats: Any,
) -> ObservationNode:
    """
    Expands action new :py:class:`ObservationNode` with action child for each
    :py:class:`~online_pomdp_planning.types.Action`

    When provided with the available actions, this implements
    :py:class:`Expansion`

    :param o: the new observation
    :type o: Observation
    :param action_node: the current leaf node
    :type action:_node: ActionNode
    :param actions: the available actions
    :type actions: Action
    :param init_stats: the initial statistics for each node
    :type init_stats: Any
    :return: action new subtree as child of `action`
    :rtype: ObservationNode
    """
    expansion: ObservationNode = ObservationNode(parent=action_node)

    for a in actions:
        expansion.add_action_node(a, ActionNode(deepcopy(init_stats), expansion))

    action_node.add_observation_node(o, expansion)

    return expansion


class Evaluation(Protocol):
    """The signature of leaf node evaluation

    .. automethod:: __call__
    """

    def __call__(self, s: State, o: Observation, t: bool) -> Any:
        """Evaluates a leaf node

        :param s: state to evaluate
        :type s: State
        :param o: observation to evaluate
        :type o: Observation
        :param t: whether the episode terminated
        :type t: bool
        :return: evaluation, can be whatever, given to :py:class:`BackPropagation`
        :rtype: Any
        """


class BackPropagation(Protocol):
    """The signature for back propagation through nodes

    .. automethod:: __call__
    """

    def __call__(self, n: ObservationNode, out: Any, val: Any) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :type n: ObservationNode
        :param out: The output of the selection method
        :type out: Any
        :param val: The output of the evaluation method
        :type val: Any
        :return: has only side effects
        :rtype: None
        """


class ActionSelection(Protocol):
    """The signature for selection actions given the root node

    .. automethod:: __call__
    """

    def __call__(self, stats: Dict[Action, Any]) -> Action:
        """Selects the preferred action given statistics


        :param stats: statistics of the (root) node
        :type stats: Dict[Action, Any]
        :return: preferred action
        :rtype: Action
        """


def pick_max_q(stats: Dict[Action, Any]):
    """Picks the action with the highest 'q-value' in their statistics

    Assumes stats has a ['qval'] attribute

    Implements :py:class:`ActionSelection`

    :param stats: assumes a "q" property in the statistic
    :type stats: Dict[Action, Any]
    """
    return max(stats, key=lambda k: stats[k]["qval"])  # type: ignore


def mcts(
    belief: Belief,
    n_sims: int,
    leaf_select: LeafSelection,
    expand: Expansion,
    evaluate: Evaluation,
    backprop: BackPropagation,
    action_select: ActionSelection,
):
    """The general MCTS method, defined by its components

    MCTS will run `n_sims` simulations, where each simulation:

    #. Selects a leaf (action) node through `leaf_select`
    #. Expands the leaf node through `expand`
    #. Evaluates the leaf node through `evaluate`
    #. Back propagates and updates node values through `backprop`

    After spending the simulation budget, it picks an given the statistics
    stored in the root node through `action_select`.

    :param belief: the current belief (over the state) at the root node
    :type belief: Belief
    :param n_sims: number of simulations to run
    :type n_sims: int
    :param leaf_select: the method for selecting leaf nodes
    :type leaf_select: LeafSelection
    :param expand: the leaf expansion method
    :type expand: Expansion
    :param evaluate: the leaf evaluation method
    :type evaluate: Evaluation
    :param backprop: the method for updating the statistics in the visited nodes
    :type backprop: BackPropagation
    :param action_select: the method for picking an action given root node
    :type action_select: ActionSelection
    :return: the preferred action
    :rtype: Action
    """
    assert n_sims >= 0, "MCTS requires a positive number of simulations"

    root_node: ObservationNode = ObservationNode(parent=None)

    for _ in range(0, n_sims):
        state = belief()
        leaf: ActionNode

        leaf, state, obs, terminal_flag, selection_output = leaf_select(
            state, root_node
        )

        expanded_node = (
            expand(obs, leaf) if not terminal_flag else leaf.observation_node(obs)
        )

        evaluation = evaluate(state, obs, terminal_flag)
        backprop(expanded_node, selection_output, evaluation)

    return action_select(root_node.child_stats)
