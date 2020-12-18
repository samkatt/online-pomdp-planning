"""Implementation of Monte-Carlo tree search"""
from __future__ import annotations

import random
from copy import deepcopy
from functools import partial
from math import log, sqrt
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from tqdm import tqdm  # type: ignore
from typing_extensions import Protocol

from online_pomdp_planning.types import (
    Action,
    Belief,
    Info,
    Observation,
    Planner,
    Simulator,
    State,
)


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


class StopCondition(Protocol):
    """The protocol for a stop condition during MCTS

    Determines, given :py:class:`online_pomdp_planning.types.Info`, whether to
    continue simulating or not.

    .. automethod:: __call__
    """

    def __call__(self, info: Info) -> bool:
        """Signature for the stop condition

        Determines, given info, whether to stop or not

        :param info: run time information
        :type info: Info
        :return: ``True`` if determined stop condition is met
        :rtype: bool
        """


def has_simulated_n_times(n: int, info: Info) -> bool:
    """Returns true if number of iterations in ``info`` exceeds ``n``

    Given ``n``, implements :py:class:`StopCondition`

    :param n: number to have iterated
    :type n: int
    :param info: run time info (assumed to have entry "iteration" -> int)
    :type info: Info
    :return: true if number of iterations exceed ``n``
    :rtype: bool
    """
    assert n >= 0 and info["iteration"] >= 0

    return n <= info["iteration"]


class ProgressBar(StopCondition):
    """A :py:class:`StopCondition` call that prints out a progress bar

    Note: Always returns ``False``, and meant to be used in combination with
    other stop condition

    The progress bar is printed by ``tqdm``, and will magnificently fail if
    something else is printed or logged during.

    XXX: not tested because UI is hard to test, please modify with care

    .. automethod:: __call__
    """

    def __init__(self, max_sims: int):
        """Sets up a progress bar for up to ``max_sims`` simulations

        We assume that ``max_sims`` will be _exactly_ the number
        of samples to be accepted (i.e. calls to ``__call__``). Any less will
        not close the progress bar, any more and the progress bar will reset.

        :param max_sims: 'length' of progress bar
        :type max_sims: int
        """
        assert max_sims >= 0

        super().__init__()
        self._max_sims = max_sims

        # ``tqdm`` starts the progress bar upon initiation. At this point the
        # belief update is not happening yet, so we do not want to print it
        self.pbar: Optional[tqdm] = None  # pylint: disable=unsubscriptable-object

    def __call__(self, info: Info) -> bool:
        """Updates the progression bar

        Initiates the bar when ``info["iteration"]`` is 0, closes when
        ``self._max_sims`` is reached

        :param info: run time information (assumed to map "iteration" -> int)
        :type info: Info
        :return: _always_ ``False``
        :rtype: bool
        """
        current_sim = info["iteration"]
        assert current_sim >= 0

        if current_sim == 0:
            # the first sample is accepted, LGTM
            self.pbar = tqdm(total=self._max_sims)

        assert self.pbar
        self.pbar.update()

        if current_sim >= self._max_sims - 1:
            # last sample accepted!
            self.pbar.close()

        return False


class LeafSelection(Protocol):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    def __call__(
        self, s: State, node: ObservationNode, info: Info
    ) -> Tuple[ActionNode, State, Observation, bool, Any]:
        """Traverses through the tree and picks a leaf

        :type s: State
        :param node: (root) node
        :type node: ObservationNode
        :param info: run time information
        :type info: Info
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
    """Select an action using UCB with given exploration constant

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
    sim: Simulator, ucb_constant: float, state: State, node: ObservationNode, info: Info
) -> Tuple[ActionNode, State, Observation, bool, List[float]]:
    """Tree policy according to UCB: ucb to pick actions, simulator to generate observations

    When provided with `sim` and `ucb_constant`, it implements
    :py:class:`LeafSelection`, and returns a list of rewards as output

    Picks action nodes according to :py:class:`select_with_ucb`, uses `sim` to
    generate observations and pick the respective nodes

    Note: has the potential to be more general and accept any selection method
    (either action node or observation node) as input

    :param sim: a POMDP simulator
    :type sim: Simulator
    :param ucb_constant: exploration constant of UCB
    :type ucb_constant: float
    :param state: the root state
    :type state: State
    :param node: the root node
    :type node: ObservationNode
    :param info: run time information (ignored)
    :type info: Info
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
            # action node is a leaf
            break

    return node.action_node(action), state, obs, terminal_flag, list_of_rewards


class Expansion(Protocol):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    def __call__(self, o: Observation, a: ActionNode, info: Info):
        """Expands a leaf node

        :param o: observation that resulted in leaf
        :type o: Observation
        :param a: action that resulted in leaf
        :type a: ActionNode
        :param info: run time information
        :type info: Info
        :return: nothing, modifies the tree
        :rtype: None
        """


def expand_node_with_all_actions(
    actions: Iterator[Action],
    init_stats: Any,
    o: Observation,
    action_node: ActionNode,
    info: Info,
):
    """Adds an observation node to the tree with a child for each action

    Expands `action_node` with new :py:class:`ObservationNode` with action
    child for each :py:class:`~online_pomdp_planning.types.Action`

    When provided with the available actions and initial stats, this implements
    :py:class:`Expansion`

    :param actions: the available actions
    :type actions: Action
    :param init_stats: the initial statistics for each node
    :type init_stats: Any
    :param o: the new observation
    :type o: Observation
    :param action_node: the current leaf node
    :type action_node: ActionNode
    :param info: run time information (ignored)
    :type info: Info
    :return: modifies tree
    :rtype: None
    """
    expansion = ObservationNode(parent=action_node)

    for a in actions:
        expansion.add_action_node(a, ActionNode(deepcopy(init_stats), expansion))

    action_node.add_observation_node(o, expansion)


class Evaluation(Protocol):
    """The signature of leaf node evaluation

    .. automethod:: __call__
    """

    def __call__(self, s: State, o: Observation, t: bool, info: Info) -> Any:
        """Evaluates a leaf node

        :param s: state to evaluate
        :type s: State
        :param o: observation to evaluate
        :type o: Observation
        :param t: whether the episode terminated
        :type t: bool
        :param info: run time information
        :type info: Info
        :return: evaluation, can be whatever, given to :py:class:`BackPropagation`
        :rtype: Any
        """


class Policy(Protocol):
    """The signature for a policy"""

    def __call__(self, s: State, o: Observation) -> Action:
        """A (stochastic) mapping from state and/or observation to action

        :param s: the current state
        :type s: State
        :param o: the current observation
        :type o: Observation
        :return: an action
        :rtype: Action
        """


def random_policy(actions: Sequence[Action], _: State, __: Observation) -> Action:
    """A random policy just picks a random action

    Implements :py:class:`Policy` given `actions`

    :param actions: list of actions to pick randomly from
    :type actions: Sequence[Action]
    :param _: ignored (state)
    :type _: State
    :param __: ignored (observation)
    :type __: Observation
    :return: a random action
    :rtype: Action
    """
    return random.choice(actions)


def rollout(
    policy: Policy,
    sim: Simulator,
    depth: int,
    discount_factor: float,
    s: State,
    o: Observation,
    t: bool,
    info: Info,
) -> float:
    """Performs a rollout in `sim` according to `policy`

    If `policy`, `sim`, `depth`, and `discount_factor` are given, this
    implements :py:class:`Evaluation` where it returns a float as metric

    When the terminal flag `t` is set, this function will return 0.

    Given ``policy``, ``sim``, ``depth``, and ``discount_factor``, this
    implements :py:class`Evaluation`

    :param policy:
    :type policy: Policy
    :param sim: a POMDP simulator
    :type sim: Simulator
    :param depth: the longest number of actions to take
    :type depth: int
    :param discount_factor: discount factor of the problem
    :type discount_factor: float
    :param s: starting state
    :type s: State
    :param o: starting observation
    :type o: Observation
    :param t: whether the episode has terminated
    :type t: bool
    :param info: run time information (ignored)
    :type info: Info
    :return: the discounted return of following `policy` in `sim`
    :rtype: float
    """
    assert 0 <= discount_factor <= 1
    assert depth >= 0, "prevent never ending loop"

    ret = 0.0

    if t or depth == 0:
        return ret

    discount = 1.0
    for _ in range(depth):
        a = policy(s, o)
        s, o, r, t = sim(s, a)

        ret += r * discount
        discount *= discount_factor

        if t:
            break

    return ret


class BackPropagation(Protocol):
    """The signature for back propagation through nodes

    .. automethod:: __call__
    """

    def __call__(
        self,
        n: ActionNode,
        leaf_selection_output: Any,
        leaf_eval_output: Any,
        info: Info,
    ) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :type n: ActionNode
        :param leaf_selection_output: The output of the selection method
        :type leaf_selection_output: Any
        :param leaf_eval_output: The output of the evaluation method
        :type leaf_eval_output: Any
        :param info: run time information
        :type info: Info
        :return: has only side effects
        :rtype: None
        """


def backprop_running_q(
    discount_factor: float,
    leaf: ActionNode,
    leaf_selection_output: List[float],
    leaf_evaluation: float,
    info: Info,
) -> None:
    """Updates running Q average of visited nodes

    Implements :py:class:`BackPropagation`

    Updates the visited nodes (through parents of `leaf`) by updating the running
    Q average. Assumes the statistics in nodes have mappings "qval" -> float
    and "leaf" -> int.

    Given a `discount_factor`, implements :py:class:`BackPropagation` with a
    list of rewards as input from :py:class:`LeafSelection` and a return
    estimate (float) from :py:class:`Evaluation`.

    :param discount_factor: 'gamma' of the POMDP environment [0, 1]
    :type discount_factor: float
    :param leaf: leaf node
    :type leaf: ActionNode
    :param leaf_selection_output: list of rewards from tree policy
    :type leaf_selection_output: List[float]
    :param leaf_evaluation: return estimate
    :type leaf_evaluation: float
    :param info: run time information (ignored)
    :type info: Info
    :return: has only side effects
    :rtype: None
    """
    assert 0 <= discount_factor <= 1

    reverse_return = leaf_evaluation

    # loop through all rewards in reverse order
    # simultaneously traverse back up the tree through `leaf`
    n: Optional[ActionNode] = leaf  # pylint: disable=E1136
    for reward in reversed(leaf_selection_output):
        assert n, "somehow got to root without processing all rewards"

        reverse_return = reward + discount_factor * reverse_return

        # grab current stats
        stats = n.stats
        q, num = stats["qval"], stats["n"]

        # store next stats
        stats["qval"] = (q * num + reverse_return) / (num + 1)
        stats["n"] = num + 1

        # go up in tree
        n = n.parent.parent

    # make sure we reached the 'root action node'
    assert n is None


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


class TreeConstructor(Protocol):
    """The signature for creating the root node

    .. automethod:: __call__
    """

    def __call__(self) -> ObservationNode:
        """Creates a root node out of nothing

        :return: The root node
        :rtype: ObservationNode
        """


def create_root_node_with_child_for_all_actions(
    actions: Iterator[Action],
    init_stats: Any,
) -> ObservationNode:
    """Creates a tree by initiating the first action nodes

    :param actions: the actions to initiate nodes for
    :type actions: Iterator[Action]
    :param init_stats: the initial statistics of those nodes
    :type init_stats: Any
    :return: the root of the tree
    :rtype: ObservationNode
    """
    root = ObservationNode()

    for a in actions:
        root.add_action_node(a, ActionNode(deepcopy(init_stats), root))

    return root


def mcts(
    stop_cond: StopCondition,
    tree_constructor: TreeConstructor,
    leaf_select: LeafSelection,
    expand: Expansion,
    evaluate: Evaluation,
    backprop: BackPropagation,
    action_select: ActionSelection,
    belief: Belief,
) -> Tuple[Action, Info]:
    """The general MCTS method, defined by its components

    MCTS will simulate until ``stop_cond`` returns False, where each
    simulation:

    #. Selects a leaf (action) node through `leaf_select`
    #. Expands the leaf node through `expand`
    #. Evaluates the leaf node through `evaluate`
    #. Back propagates and updates node values through `backprop`

    After spending the simulation budget, it picks an given the statistics
    stored in the root node through `action_select`.

    The root node constructor allows for custom ways of initiating the tree

    During run time will maintain information
    :py:class`~online-pomdp-planning.types.Info`, with "iteration" -> #
    simulations run. This is passed to all the major components of MCTS, which
    in turn can populate them however they would like. Finally this is
    returned, and thus can be used for reporting and debugging like.

    :param stop_cond: the function that returns whether simulating should stop
    :type stop_cond: StopCondition
    :param tree_constructor: constructor the tree
    :type tree_constructor: TreeConstructor
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
    :param belief: the current belief (over the state) at the root node
    :type belief: Belief
    :return: the preferred action and run time information (e.g. # simulations)
    :rtype: Tuple[Action, Info]
    """

    info: Info = {"iteration": 0}

    root_node = tree_constructor()

    while not stop_cond(info):

        state = belief()
        leaf: ActionNode

        leaf, state, obs, terminal_flag, selection_output = leaf_select(
            state, root_node, info
        )

        if not terminal_flag:
            expand(obs, leaf, info)

        evaluation = evaluate(state, obs, terminal_flag, info)
        backprop(leaf, selection_output, evaluation, info)

        info["iteration"] += 1

    return action_select(root_node.child_stats), info


def create_POUCT(
    actions: Sequence[Action],
    sim: Simulator,
    num_sims: int,
    init_stats: Any = None,
    policy: Optional[Policy] = None,  # pylint: disable=E1136
    ucb_constant: float = 1,
    rollout_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :py:func:`mcts` where the components have been
    filled in.

    :param actions: all the actions available to the agent
    :type actions: Sequence[Action]
    :param sim: a simulator of the environment
    :type sim: Simulator
    :param num_sims: number of simulations to run
    :type num_sims: int
    :param init_stats: how to initialize node statistics, defaults to None which sets Q and n to 0
    :type init_stats: Any
    :param policy: the rollout policy, defaults to None, which sets a random policy
    :type policy: Optional[Policy]
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :type ucb_constant: Optional[float]
    :param rollout_depth: the depth a rollout will go up to, defaults to 100
    :type rollout_depth: Optional[int]
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :type discount_factor: Optional[float]
    :param progress_bar: flag to output a progress bar, defaults to False
    :type progress_bar: bool
    :return: MCTS with planner signature (given num sims)
    :rtype: Planner
    """

    assert num_sims > 0

    # defaults
    if not policy:
        policy = partial(random_policy, actions)
    if not init_stats:
        init_stats = {"qval": 0, "n": 0}

    # stop condition
    pbar = ProgressBar(num_sims) if progress_bar else lambda _: False
    real_stop_cond = partial(has_simulated_n_times, num_sims)

    def stop_condition(info: Info) -> bool:
        return real_stop_cond(info) or pbar(info)  # type: ignore

    tree_constructor = partial(
        create_root_node_with_child_for_all_actions, actions, init_stats
    )
    leaf_select = partial(ucb_select_leaf, sim, ucb_constant)
    expansion = partial(expand_node_with_all_actions, actions, init_stats)
    evaluation = partial(rollout, policy, sim, rollout_depth, discount_factor)
    backprop = partial(backprop_running_q, discount_factor)
    action_select = pick_max_q

    return partial(
        mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        expansion,
        evaluation,
        backprop,
        action_select,
    )
