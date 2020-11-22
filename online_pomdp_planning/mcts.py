"""Implementation of Monte-Carlo tree search"""
import abc
from typing import Dict, Optional, Tuple

from online_pomdp_planning.types import Action, Belief, Observation, State


class Statistics(abc.ABC):
    """The statistics stored in action nodes"""


class SelectionOutput(abc.ABC):
    """The output of the selection method (may have additional statistics)"""


class EvaluationMetric(abc.ABC):
    """The output of an leaf evaluation method"""


class ActionNode:
    """A decision node in the MCTS tree

    This node maps observations to children nodes. In addition it stores
    statistics, such as the expected Q-value associated with this node.
    """

    def __init__(
        self,
        initial_statistics: Statistics,
        parent: 'ObservationNode',
    ):
        """Initializes the action node with given statistics

        :param initial_statistics: anything you would like to store
        :type initial_statistics: Statistics
        :param parent: the parent node in the tree
        :type parent: 'ObservationNode'
        """
        self.parent = parent
        self.observation_nodes: Dict[Observation, 'ObservationNode'] = {}
        self.stats = initial_statistics

    def observation_node(self, observation: Observation) -> 'ObservationNode':
        """The child-node associated with given observation `o`

        :param observation:
        :type observation: Observation
        :return: child node
        :rtype: 'ObservationNode'
        """
        return self.observation_nodes[observation]


class ObservationNode:
    """A chance/observation node in the MCTS tree

    This node representation action-observation history, ending up in a
    particular observation. It has no statistics. If it has no parent, then it
    is the root node. This node maps actions to children nodes.
    """

    def __init__(self, parent: Optional[ActionNode]):
        """Initiates an observation node with given parent

        :param parent: if no parent is given, this must be the root node
        :type parent: Optional[ActionNode]
        """
        self.parent = parent
        self.action_nodes: Dict[Action, ActionNode] = {}

    @property
    def child_stats(self) -> Dict[Action, Statistics]:
        """Returns a mapping from actions to statistics (shortcut)

        :return: action -> stats mapping
        :rtype: Dict[Action, Statistics]
        """
        return {a: n.stats for a, n in self.action_nodes.items()}

    def action_node(self, action: Action) -> ActionNode:
        """Get child node associated with `action`

        :param action:
        :type action: Action
        :return: returns child node
        :rtype: ActionNode
        """
        # TODO: ensure one is created if it does not exist yet
        return self.action_nodes[action]


class LeafSelection(abc.ABC):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(
        self, s: State, node: ObservationNode
    ) -> Tuple[ActionNode, State, Observation, SelectionOutput]:
        """Traverses through the tree and picks a leaf

        :type s: State
        :param node: (root) node
        :type node: ObservationNode
        :return: leaf node selected
        :rtype: Tuple[ActionNode, State, Observation, SelectionOutput]
        """


class Expansion(abc.ABC):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, o: Observation, a: ActionNode) -> ObservationNode:
        """Expands a leaf node

        :param o: observation that resulted in leaf
        :type o: Observation
        :param a: action that resulted in leaf
        :type a: ActionNode
        :return: expanded node
        :rtype: ObservationNode
        """


class Evaluation(abc.ABC):
    """The signature of leaf node evaluation

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, s: State, o: Observation) -> EvaluationMetric:
        """Evaluates a leaf node

        :param s: state to evaluate
        :type s: State
        :param o: observation to evaluate
        :type o: Observation
        :return: evaluation (can be whatever)
        :rtype: EvaluationMetric
        """


class BackPropagation(abc.ABC):
    """The signature for back propagation through nodes

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(
        self, n: ObservationNode, out: SelectionOutput, val: EvaluationMetric
    ) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :type n: ObservationNode
        :param out: The output of the selection method
        :type out: SelectionOutput
        :param val: The output of the evaluation method
        :type val: EvaluationMetric
        :return: has only side effects
        :rtype: None
        """


class ActionSelection(abc.ABC):
    """The signature for selection actions given the root node

    .. automethod:: __call__
    """

    @abc.abstractmethod
    def __call__(self, n: ObservationNode) -> Action:


def general_MCTS(
    belief: Belief,
    n_sims: int,
    leaf_select: LeafSelection,
    expand: Expansion,
    evaluate: Evaluation,
    backprop: BackPropagation,
    action_select: ActionSelection,
) -> Action:
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
        leaf_node, leaf_state, leaf_observation, selection_output = leaf_select(  # type: ignore
            state, root_node
        )
        expanded_node = expand(leaf_observation, leaf_node)
        evaluation = evaluate(leaf_state, leaf_observation)  # type: ignore
        backprop(expanded_node, selection_output, evaluation)

    return action_select(root_node)
