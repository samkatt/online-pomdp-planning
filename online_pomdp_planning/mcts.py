"""Implementation of Monte-Carlo tree search"""
from __future__ import annotations

import random
from copy import deepcopy
from functools import partial
from math import isclose, log, sqrt
from timeit import default_timer as timer
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from tqdm import tqdm
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
from online_pomdp_planning.utils import MovingStatistic, normalize_float

Stats = Dict[str, Any]
"""Alias type for statistics: a mapping from some description to anything"""
ActionStats = Dict[Action, Stats]
"""Alias type for action statistics: a mapping from actions to :class:`Stats`"""


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
        :param parent: the parent node in the tree
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
        :param observation_node: the new child node
        """
        assert observation not in self.observation_nodes
        assert observation_node.parent == self
        self.observation_nodes[observation] = observation_node

    def observation_node(self, observation: Observation) -> ObservationNode:
        """The child-node associated with given observation `o`

        Raises `KeyError` if `action` is not associated with a child node

        :param observation:
        :return: child node
        """
        return self.observation_nodes[observation]


class ObservationNode:
    """A chance/observation node in the MCTS tree

    This node representation action-observation history, ending up in a
    particular observation. It has no statistics. If it has no parent, then it
    is the root node. This node maps actions to children nodes.
    """

    def __init__(self, parent: Optional[ActionNode] = None):
        """Initiates an observation node with given parent

        :param parent: if no parent is given, this must be the root node
        """
        self.parent = parent
        self.action_nodes: Dict[Action, ActionNode] = {}

    @property
    def child_stats(self) -> ActionStats:
        """Returns a mapping from actions to statistics (shortcut)

        :return: action -> stats mapping
        """
        return {a: n.stats for a, n in self.action_nodes.items()}

    def add_action_node(self, action: Action, node: ActionNode):
        """Adds a `action` -> `node` mapping to children

        Raises `AssertionError` if:
            - the parent of the added node is not self
            - `action` is not already associated with a child node


        :param action: action associated with new child
        :param node: child node
        """
        assert action not in self.action_nodes
        assert node.parent == self
        self.action_nodes[action] = node

    def action_node(self, action: Action) -> ActionNode:
        """Get child node associated with `action`

        Raises `KeyError` if `action` is not associated with a child node

        :param action:
        :return: returns child node
        """
        return self.action_nodes[action]


class DeterministicNode:
    """A node in the tree of a deterministic process

    While most MCTS methods assume stochastic dynamics, if the environment is
    deterministic, then there is one branching less to care about. This node
    allows for constructing such trees.

    Specifically, the children of `this` are themselves `DeterministicNode`.
    Each action has one 'outcome' and thus one node associated with it.
    """

    def __init__(self, stats: Any, parent: Optional[DeterministicNode]):
        """Initiates the node and sets its `parent` and `stats`

        :param stats: stored, can be anything that is used by algorithms traversing the tree
        :param parent: the parent of this node, if null then must be root
        """
        self.stats = stats
        self.parent = parent
        self.children: Dict[Action, DeterministicNode] = {}

    @property
    def expanded(self) -> bool:
        """Whether `self` has been expanded yet"""
        return len(self.children) > 0

    @property
    def child_stats(self) -> ActionStats:
        """Returns an action => stats dictionary of the children"""
        return {a: n.stats for a, n in self.children.items()}

    def add_child(self, a: Action, n: DeterministicNode):
        """Adds `n` as child associated with `a`"""
        assert a not in self.children
        assert n.parent == self
        self.children[a] = n

    def child(self, a: Action) -> DeterministicNode:
        """Get child node associated with `a`"""
        return self.children[a]


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
        :return: ``True`` if determined stop condition is met
        """


def no_stop(info: Info) -> bool:
    """:class:`StopCondition` implementatio that always returns `False`"""
    return False


def has_simulated_n_times(n: int, info: Info) -> bool:
    """Returns true if number of iterations in ``info`` exceeds ``n``

    Given ``n``, implements :py:class:`StopCondition`

    :param n: number to have iterated
    :param info: run time info (assumed to have entry "iteration" -> int)
    :return: true if number of iterations exceed ``n``
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
        """
        assert max_sims >= 0

        super().__init__()
        self._max_sims = max_sims

        # ``tqdm`` starts the progress bar upon initiation. At this point the
        # belief update is not happening yet, so we do not want to print it
        self.pbar: Optional[tqdm] = None

    def __call__(self, info: Info) -> bool:
        """Updates the progression bar

        Initiates the bar when ``info["iteration"]`` is 0, closes when
        ``self._max_sims`` is reached

        :param info: run time information (assumed to map "iteration" -> int)
        :return: _always_ ``False``
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

        :param node: (root) node
        :param info: run time information
        :return: leaf node, state and obs, terminal flag and input to :py:class:`BackPropagation`
        """


class DeterministicLeafSelection(Protocol):
    """The signature for leaf selection

    .. automethod:: __call__
    """

    def __call__(
        self, node: DeterministicNode, info: Info
    ) -> Tuple[DeterministicNode, Any]:
        """Traverses through the tree and picks a leaf

        :param node: (root) node
        :param info: run time information
        :return: leaf node and input to :py:class:`DeterministicBackPropagation`
        """


ActionScoringMethod = Callable[[ActionStats, Info], Dict[Action, float]]
"""Type used to evaluate actions during tree traversal"""


def ucb(
    q: float,
    n: int,
    n_total: int,
    ucb_constant: float,
) -> float:
    """Returns the upper confidence bound of Q

    UCB is `q + ucb_constant * sqrt(log(n_total) / n)`.

    :param q: the q-value
    :param n: the number of times this action has been chosen
    :param n_total: the total number of times any action has been chosen
    :param ucb_constant: the exploration constant
    :return: UCB of `q`
    """
    assert n >= 0 and n_total >= 0 and ucb_constant >= 0
    if n == 0:
        return float("inf")

    return q + ucb_constant * sqrt(log(n_total) / n)


def ucb_scores(
    stats: ActionStats,
    info: Info,
    ucb_constant: float,
) -> Dict[Action, float]:
    """The upper-confidence bound scoring method (used in :func:`select_action`)

    Assumes that ``stats`` contains an entry for "qval" and "n".

    See :func:`ucb`

    Given ``ucb_constant``, implements the :class:`ActionScoringMethod`.

    :param stats: an action => stats mapping
    :param info: ignored
    :param ucb_constant: the upper-confidence bound constant
    :return: an action => score mapping, here the upper confidence bound on the q values
    """
    total_visits = sum(s["n"] for s in stats.values())

    return {
        a: ucb(s["qval"], s["n"], total_visits, ucb_constant) for a, s in stats.items()
    }


def muzero_prior_score(p: float, n: int, n_total: int) -> float:
    """Computes the node's prior term in muzero's UCB scoring

    The contribution of the node specific prior in :func:`muzero_ucb_scores`::

        p * (sqrt(n_total) / (n + 1))

    XXX: *not* tested since I really have no idea, other than copying the
    formula from the paper or pseudocode, what outputs to expect for certain
    inputs.

    :param p: the prior probability given by some policy
    :param n: number of times this action has been chosen
    :param n_total: number of times *an action* has been chosen
    :return: a prior score [0,1]
    """
    return p * (sqrt(n_total) / (1 + n))


def muzero_ucb_scores(
    stats: ActionStats, info: Info, c1: float, c2: float
) -> Dict[Action, float]:
    """The UCB scoring method used my muzero

    Returns an action => score mapping, where the scores are::

        norm(q) + prior * (sqrt(N) / 1 + n) * (c1 + log((N + c2 + 1) / c2))
        q       + prior & exploration       *  base term

    with `N` being total number of visits and `n` being the number of visits of
    the child node, and `norm(q)` is the normalized q value of the node.

    The second (prior) term is implemented in :func:`muzero_prior_score`.

    Assumes ``stats`` contains `qval` and `n` for every action and that
    ``info`` contains `q_statistic`.

    XXX: *not* tested, no idea how to sensibly do that honestly

    See paper Schrittwieser, Julian, et al. Mastering atari, go, chess and
    shogi by planning with a learned model." Nature 588.7839 (2020): 604-609.".

    :param stats: action => stats mapping
    :param info: contains `q_staticstic`
    :param c1: first exploration constant
    :param c2: second exploration constant
    :return: action => float scores
    """
    q_stat = info["q_statistic"]

    # pre-computed for all actions
    total_visits = sum(s["n"] for s in stats.values())

    # base term
    base_term = c1 + log((total_visits + c2 + 1) / c2)

    # q: assigning ``0`` to unvisited actions (not "inf")
    q_values = {a: stat["qval"] if stat["n"] != 0 else 0 for a, stat in stats.items()}
    if q_stat.min < q_stat.max:
        q_values = {
            a: normalize_float(q, q_stat.min, q_stat.max) for a, q in q_values.items()
        }

    # p
    priors = {
        a: muzero_prior_score(stat["prior"], stat["n"], total_visits)
        for a, stat in stats.items()
    }

    # q     +    prior & expl   *   base term
    return {a: q_values[a] + priors[a] * base_term for a in stats}


def select_action(
    stats: ActionStats,
    info: Info,
    scoring_method: ActionScoringMethod,
) -> Action:
    """Select an action using `scoring_method`

    Exactly how each action is scored (given `stats`) is up to the scoring
    method. This function simply picks a randomly between the actions that are
    given the maximum score.

    :param stats: the statistics: Action -> Dict where Dict is {"qval": float, "n": int}
    :param info: current MCTS running info, given to ``scoring_method``
    :param scoring_method: the method that transforms `stats` into scores
    :return: the action with the highest upper confidence bound
    """
    action_scores = scoring_method(stats, info)

    max_score = max(action_scores.values())
    return random.choice(
        [a for a, score in action_scores.items() if score >= max_score]
    )


def select_leaf_by_max_scores(
    sim: Simulator,
    scoring_method: ActionScoringMethod,
    max_depth: int,
    state: State,
    node: ObservationNode,
    info: Info,
) -> Tuple[ActionNode, State, Observation, bool, List[float]]:
    """Tree policy: ``scoring_method`` to pick actions, simulator to generate observations

    When provided with ``sim`` and ``scoring_method``, it implements
    :class:`LeafSelection`, and returns a list of rewards as output

    Picks action nodes according to :func:`select_action` --- with
    ``scoring_method`` --- uses ``sim`` to generate observations and pick the
    respective nodes

    Tracks the tree depth, maintains a running statistic on it in ``info``, and
    stops going down the tree when ``max_depth`` is reached.

    :param sim: a POMDP simulator
    :param scoring_method: function that, given action stats, returns their scores
    :param max_depth: max length of the tree to go down
    :param state: the root state
    :param node: the root node
    :param info: run time information (ignored)
    :return: leaf node, state, observation, terminal flag and list of rewards
    """
    assert max_depth > 0
    list_of_rewards: List[float] = []

    depth = 0
    while True:

        action = select_action(node.child_stats, info, scoring_method)
        state, obs, reward, terminal_flag = sim(state, action)

        list_of_rewards.append(reward)
        depth += 1

        if terminal_flag or depth >= max_depth:
            break

        try:
            node = node.action_node(action).observation_node(obs)
        except KeyError:
            # action node is a leaf
            break

    # info tracking number of terminal simulations
    info.setdefault("ucb_num_terminal_sims", 0)
    if terminal_flag:
        info["ucb_num_terminal_sims"] += 1

    # info tracking tree depth
    info.setdefault("ucb_tree_depth", MovingStatistic())
    info["ucb_tree_depth"].add(depth)

    return node.action_node(action), state, obs, terminal_flag, list_of_rewards


def select_deterministc_leaf_by_max_scores(
    scoring_method: ActionScoringMethod,
    node: DeterministicNode,
    info: Info,
) -> Tuple[DeterministicNode, Any]:
    """Deterministic tree policy according to ``scoring_method``

    When provided with ``scoring_method``, it implements
    :class:`DeterministicLeafSelection`.

    Returns "None" as second argument

    Picks action nodes according to :py:class:`select_action` (with
    `scoring_method` :func:`muzero_ucb_scores`).

    Assumes "q_statistic" is in ``info``

    Tracks the tree depth, maintains a running statistic on it in
    ``info["ucb_tree_depth"]``.

    :param scoring_method: function that, given action stats, returns their scores
    :param node: the root node
    :param info: run time information
    :return: leaf node
    """

    depth = 0

    while node.expanded:
        action = select_action(node.child_stats, info, scoring_method)
        node = node.child(action)
        depth += 1

    # info tracking tree depth
    info.setdefault("ucb_tree_depth", MovingStatistic())
    info["ucb_tree_depth"].add(depth)

    return node, None


class Expansion(Protocol):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    def __call__(self, o: Observation, a: ActionNode, info: Info):
        """Expands a leaf node

        :param o: observation that resulted in leaf
        :param a: action that resulted in leaf
        :param info: run time information
        :return: nothing, modifies the tree
        """


class DeterministicNodeExpansion(Protocol):
    """The signature for leaf node expansion

    .. automethod:: __call__
    """

    def __call__(self, n: DeterministicNode, info: Info) -> Any:
        """Expands a leaf node

        :param n: current (not-expanded) leaf node
        :param info: run time information
        :return: evaluation, can be whatever, given to :py:class:`DeterministicBackPropagation`
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

    NOTE: ``action_node`` must not have a child node associated with ``o`` or
    this will result in an ``AssertionError``

    :param actions: the available actions
    :param init_stats: the initial statistics for each node
    :param o: the new observation
    :param action_node: the current leaf node
    :param info: run time information (ignored)
    :return: modifies tree
    """
    assert o not in action_node.observation_nodes

    if len(action_node.observation_nodes) == 0:
        # first time this action node was expanded,
        # so now we count it as part of the tree
        info["mcts_num_action_nodes"] = info.get("mcts_num_action_nodes", 0) + 1

    expansion = ObservationNode(parent=action_node)

    for a in actions:
        expansion.add_action_node(a, ActionNode(deepcopy(init_stats), expansion))

    action_node.add_observation_node(o, expansion)


def muzero_expand_node(
    n: DeterministicNode,
    info: Info,
    inference: Callable[[State, Action], MuzeroInferenceOutput],
) -> float:
    """Muzero's way of expanding and evaluating a node

    `n` is assumed to contain `action`, and its parent `latent_state`.

    Will create a child for each action in the policy evaluated through
    `inference` and create a child for it.

    :param n: node to be expanded
    :param info: ignored
    :param inference: function used to get latent state and evaluations
    """
    assert n.parent
    assert not n.expanded

    network_output = inference(n.parent.stats["latent_state"], n.stats["action"])

    n.stats["latent_state"] = network_output.latent_state
    n.stats["reward"] = network_output.reward

    n.children = {
        a: DeterministicNode({"prior": p, "action": a, "n": 0, "qval": 0.0}, n)
        for a, p in network_output.policy.items()
    }

    return network_output.value


class Evaluation(Protocol):
    """The signature of leaf node evaluation

    .. automethod:: __call__
    """

    def __call__(self, s: State, o: Observation, t: bool, info: Info) -> Any:
        """Evaluates a leaf node

        :param s: state to evaluate
        :param o: observation to evaluate
        :param t: whether the episode terminated
        :param info: run time information
        :return: evaluation, can be whatever, given to :py:class:`BackPropagation`
        """


class Policy(Protocol):
    """The signature for a policy"""

    def __call__(self, s: State, o: Observation) -> Action:
        """A (stochastic) mapping from state and/or observation to action

        :param s: the current state
        :param o: the current observation
        :return: an action
        """


def random_policy(actions: Sequence[Action], _: State, __: Observation) -> Action:
    """A random policy just picks a random action

    Implements :py:class:`Policy` given `actions`

    :param actions: list of actions to pick randomly from
    :param _: ignored (state)
    :param __: ignored (observation)
    :return: a random action
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
    :param sim: a POMDP simulator
    :param depth: the longest number of actions to take
    :param discount_factor: discount factor of the problem
    :param s: starting state
    :param o: starting observation
    :param t: whether the episode has terminated
    :param info: run time information (ignored)
    :return: the discounted return of following `policy` in `sim`
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
        :param leaf_selection_output: The output of the selection method
        :param leaf_eval_output: The output of the evaluation method
        :param info: run time information
        :return: has only side effects
        """


class DeterministicBackPropagation(Protocol):
    """The signature for back propagation through deterministic nodes

    .. automethod:: __call__
    """

    def __call__(
        self,
        n: DeterministicNode,
        leaf_selection_output: Any,
        leaf_eval_output: Any,
        info: Info,
    ) -> None:
        """Updates the nodes visited during selection

        :param n: The leaf node that was expanded
        :param leaf_selection_output: The output of the selection method
        :param leaf_eval_output: The output of the evaluation method
        :param info: run time information
        :return: has only side effects
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

    Updates the visited nodes (through parents of `leaf`) by updating the
    running Q average. Assumes the statistics in nodes have mappings "qval" ->
    float and "leaf" -> int.

    Given a `discount_factor`, implements :py:class:`BackPropagation` with a
    list of rewards as input from :class:`LeafSelection` and a return estimate
    (float) from :class:`Evaluation`.

    :param discount_factor: 'gamma' of the POMDP environment [0, 1]
    :param leaf: leaf node
    :param leaf_selection_output: list of rewards from tree policy
    :param leaf_evaluation: return estimate
    :param info: run time information (ignored)
    :return: has only side effects
    """
    assert 0 <= discount_factor <= 1

    reverse_return = leaf_evaluation

    # loop through all rewards in reverse order
    # simultaneously traverse back up the tree through `leaf`
    n: Optional[ActionNode] = leaf
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


def deterministic_qval_backpropagation(
    discount_factor: float,
    leaf: DeterministicNode,
    leaf_selection_output: Any,
    leaf_eval_output: float,
    info: Info,
) -> None:
    """Backpropagation for deterministic trees (used in muzero)

    Implements :py:class:`DeterministicBackPropagation` given

    Will close bounds `info["q_statistic"]`

    :param discount_factor: discount factor used in computing return
    :param n: the leaf node to start propagating back from
    :param leaf_selection_output: ignored
    :param leaf_eval_output: expected to be a float (evaluation)
    :param info: "q_statistic" updated
    :return: none, only side efects in tree
    """
    assert 0 <= discount_factor <= 1

    value = leaf_eval_output

    n: Optional[DeterministicNode] = leaf

    while n:

        value += n.stats["reward"]

        # update stats in `n`
        num, q = n.stats["n"], n.stats["qval"]

        n.stats["qval"] = (q * num + value) / (num + 1)
        n.stats["n"] = num + 1

        # adjust bounds in `info`
        info["q_statistic"].add(n.stats["qval"])

        value *= discount_factor
        n = n.parent


class ActionSelection(Protocol):
    """The signature for selection actions given the root node

    .. automethod:: __call__
    """

    def __call__(self, stats: ActionStats, info: Info) -> Action:
        """Selects the preferred action given statistics


        :param stats: statistics of the (root) node
        :param info: run time information
        :return: preferred action
        """


def max_q_action_selector(stats: ActionStats, info: Info) -> Action:
    """Picks the action with the highest 'q-value' in their statistics

    Assumes stats has a ['qval'] attribute

    Implements :py:class:`ActionSelection`

    Adds ranking to ``info["max_q_action_selector-values"]``, which is a sorted
    list (by q-value) of action-stats pairs

    :param stats: assumes a "q" property in the statistic
    :param info: run time information (adds "max_q_action_selector-values")
    :return: action with highest q value
    """
    qvals = [(k, v["qval"]) for k, v in stats.items()]
    info["max_q_action_selector-values"] = sorted(
        qvals, key=lambda action_qval_pair: action_qval_pair[1], reverse=True
    )
    return info["max_q_action_selector-values"][0][0]


def max_visits_action_selector(stats: ActionStats, info: Info) -> Action:
    """implements :py:class:`ActionSelection`. Samples action most picked by MCTS.

    Assumes `stats` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contains a "n" entry that reflects how often
    the action has been chosen.

    Populates `info['visit_action_selector-counts']` with visit counts

    """
    action_visits = [(k, v["n"]) for k, v in stats.items()]

    info["visit_action_selector-counts"] = sorted(
        action_visits,
        key=lambda pair: pair[1],
        reverse=True,
    )

    return info["visit_action_selector-counts"][0][0]


def visit_prob_action_selector(stats: ActionStats, info: Info) -> Action:
    """implements :py:class:`ActionSelection`. Samples action according to visitation counts

    Assumes `stats` is a action => dict statistic dictionary. Each of those
    dictionaries is expected to contains a "n" entry that reflects how often
    the action has been chosen.

    Populates `info['visit_action_selector-probabilities']` with probability
    distribution and `info['visit_action_selector-counts']` with the actual
    visits.
    """
    # extract (and sort by) visits statistic
    action_visits = [(k, v["n"]) for k, v in stats.items()]
    info["visit_action_selector-counts"] = sorted(
        action_visits,
        key=lambda pair: pair[1],
        reverse=True,
    )

    # compute (sorted) probabilities
    tot = sum(visits for (_, visits) in info["visit_action_selector-counts"])
    assert tot > 0

    info["visit_action_selector-probabilities"] = [
        (a, visits / tot) for (a, visits) in info["visit_action_selector-counts"]
    ]

    return random.choices(
        [x[0] for x in info["visit_action_selector-probabilities"]],
        [x[1] for x in info["visit_action_selector-probabilities"]],
    )[0]


class TreeConstructor(Protocol):
    """The signature for creating the root node

    .. automethod:: __call__
    """

    def __call__(self) -> ObservationNode:
        """Creates a root node out of nothing

        :return: The root node
        """


class DeterministicTreeConstructor(Protocol):
    """The signature for creating the root node

    .. automethod:: __call__
    """

    def __call__(self, history_representation: Any) -> DeterministicNode:
        """Creates a root node out of nothing

        :return: The root node
        """


def create_root_node_with_child_for_all_actions(
    actions: Iterator[Action],
    init_stats: Any,
) -> ObservationNode:
    """Creates a tree by initiating the first action nodes

    :param actions: the actions to initiate nodes for
    :param init_stats: the initial statistics of those nodes
    :return: the root of the tree
    """
    root = ObservationNode()

    for a in actions:
        root.add_action_node(a, ActionNode(deepcopy(init_stats), root))

    return root


def create_muzero_root(
    latent_state: Any,
    reward: float,
    prior: Dict[Action, float],
    noise_dirichlet_alpha: float,
    noise_exploration_fraction: float,
) -> DeterministicNode:
    """Creates a root node for mu-zero

    Given all input, implements :py:class:`DeterministicTreeConstructor`

    The prior value given to each child of the returned root is a weighted
    combination of `prior` and some noise. The variance of the noise is given
    by `noise_dirichlet_alpha`, which is the parameter of a Dirichlet. The
    _larger_ this value, the higher the noise. The `noise_exploration_fraction`
    is the weight given to the noise.

    :param latent_state: the current history/observation/state representation for muzero dynamics
    :param reward: reward associated with current history/observation/state
    :param prior: action -> probability mapping of current history/observation/state
    :param noise_dirichlet_alpha: level of noise added to `prior` of root children
    :param noise_exploration_fraction: trade off between `prior` and noise
    """

    assert 0 <= noise_exploration_fraction <= 1
    assert isclose(1, sum(p for p in prior.values())), f"Prior is weird: {prior}"

    root = DeterministicNode(
        {"latent_state": latent_state, "reward": reward, "n": 0, "qval": 0.0}, None
    )

    # (noisy) prior in root for extra exploration
    noise_gen = np.random.default_rng().dirichlet([noise_dirichlet_alpha] * len(prior))

    # magic stolen from mu-zero:
    # each prior is given some noise.
    # Amount of variance in noise is set through `noise_dirichlet_alpha`
    # basically weighted average (according to `noise_exploration_fraction`
    noisy_prior = {
        a: p * (1 - noise_exploration_fraction)
        + noise_gen[i] * noise_exploration_fraction
        for i, (a, p) in enumerate(prior.items())
    }

    root.children = {
        a: DeterministicNode({"prior": p, "action": a, "n": 0, "qval": 0.0}, root)
        for a, p in noisy_prior.items()
    }

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

    Lastly ``info`` returned will contain "plan_runtime" measurement.

    :param stop_cond: the function that returns whether simulating should stop
    :param tree_constructor: constructor the tree
    :param leaf_select: the method for selecting leaf nodes
    :param expand: the leaf expansion method
    :param evaluate: the leaf evaluation method
    :param backprop: the method for updating the statistics in the visited nodes
    :param action_select: the method for picking an action given root node
    :param belief: the current belief (over the state) at the root node
    :return: the preferred action and run time information (e.g. # simulations)
    """

    info: Info = {"iteration": 0}

    root_node = tree_constructor()

    t = timer()

    while not stop_cond(info):

        state = belief()

        leaf, state, obs, terminal_flag, selection_output = leaf_select(
            state, root_node, info
        )

        # So there are two scenarios in which ``leaf`` should not be expanded.
        # Either (1) the last transition was terminal, or (2) ``leaf`` is
        # actually not a leaf. (2) happens, for example, when ``leaf_select``
        # has a max depth it will go. We capture this (hopefully) by checking
        # if there are any children in ``leaf``.
        if not terminal_flag and len(leaf.observation_nodes) == 0:
            expand(obs, leaf, info)

        evaluation = evaluate(state, obs, terminal_flag, info)
        backprop(leaf, selection_output, evaluation, info)

        info["iteration"] += 1

    info["plan_runtime"] = timer() - t

    return action_select(root_node.child_stats, info), info


def deterministic_tree_search(
    stop_cond: StopCondition,
    tree_constructor: DeterministicTreeConstructor,
    leaf_select: DeterministicLeafSelection,
    expand_and_evaluate: DeterministicNodeExpansion,
    backprop: DeterministicBackPropagation,
    action_select: ActionSelection,
    history_representation: Any,
) -> Tuple[Action, Info]:
    """General deterministic tree search, defined by its components

    This search will simulate until ``stop_cond`` returns False, where each
    simulation:

    #. Selects a leaf (action) node through `leaf_select`
    #. Expands and evaluates the leaf node through `expand_and_evaluate`
    #. Back propagates and updates node values through `backprop`

    After spending the simulation budget, according to `stop_cond` it picks an
    given the statistics stored in the root node through `action_select`.

    The root node constructor allows for custom ways of initiating the tree

    During run time will maintain information
    :py:class`~online-pomdp-planning.types.Info`, with "iteration" -> #
    simulations run. This is passed to all the major components of MCTS, which
    in turn can populate them however they would like. Finally this is
    returned, and thus can be used for reporting and debugging like.

    Lastly ``info`` returned will contain 'plan_runtime' measurement.

    :param stop_cond: the function that returns whether simulating should stop
    :param tree_constructor: constructor the tree
    :param leaf_select: the method for selecting leaf nodes
    :param expand_and_evaluate: the leaf expansion method
    :param backprop: the method for updating the statistics in the visited nodes
    :param action_select: the method for picking an action given root node
    :param history_representation: whatever the input to `tree_constructor`
    :return: the preferred action and run time information (e.g. # simulations)
    """

    info: Info = {"iteration": 0, "q_statistic": MovingStatistic()}

    root_node = tree_constructor(history_representation)

    t = timer()

    while not stop_cond(info):

        leaf, selection_output = leaf_select(root_node, info)
        evaluation = expand_and_evaluate(leaf, info)

        backprop(leaf, selection_output, evaluation, info)

        info["iteration"] += 1

    info["plan_runtime"] = timer() - t

    return action_select(root_node.child_stats, info), info


def create_POUCT(
    actions: Sequence[Action],
    sim: Simulator,
    num_sims: int,
    init_stats: Any = None,
    leaf_eval: Optional[Evaluation] = None,
    ucb_constant: float = 1,
    rollout_depth: int = 100,
    max_tree_depth: int = 100,
    discount_factor: float = 0.95,
    progress_bar: bool = False,
) -> Planner:
    """Creates PO-UCT given the available actions and a simulator

    Returns an instance of :py:func:`mcts` where the components have been
    filled in.

    :param actions: all the actions available to the agent
    :param sim: a simulator of the environment
    :param num_sims: number of simulations to run
    :param init_stats: how to initialize node statistics, defaults to None which sets Q and n to 0
    :param leaf_eval: the evaluation of leaves, defaults to `None`, which assumes a random rollout
    :param ucb_constant: exploration constant used in UCB, defaults to 1
    :param rollout_depth: the depth a rollout will go up to, defaults to 100
    :param max_tree_depth: the depth the tree is allowed to grow to, defaults to 100
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param progress_bar: flag to output a progress bar, defaults to False
    :return: MCTS with planner signature (given num sims)
    """
    assert num_sims > 0 and max_tree_depth > 0

    # defaults
    if not leaf_eval:
        leaf_eval = partial(
            rollout,
            partial(random_policy, actions),
            sim,
            rollout_depth,
            discount_factor,
        )

    if not init_stats:
        init_stats = {"qval": 0, "n": 0}

    # stop condition: keep track of `pbar` if ``progress_bar`` is set
    pbar = no_stop
    if progress_bar:
        pbar = ProgressBar(num_sims)
    real_stop_cond = partial(has_simulated_n_times, num_sims)

    def stop_condition(info: Info) -> bool:
        return real_stop_cond(info) or pbar(info)

    tree_constructor = partial(
        create_root_node_with_child_for_all_actions, actions, init_stats
    )
    node_scoring_method = partial(ucb_scores, ucb_constant=ucb_constant)
    leaf_select = partial(
        select_leaf_by_max_scores, sim, node_scoring_method, max_tree_depth
    )
    expansion = partial(expand_node_with_all_actions, actions, init_stats)
    backprop = partial(backprop_running_q, discount_factor)
    action_select = max_q_action_selector

    return partial(
        mcts,
        stop_condition,
        tree_constructor,
        leaf_select,
        expansion,
        leaf_eval,
        backprop,
        action_select,
    )


class MuzeroInferenceOutput(NamedTuple):
    """The output of calls to muzero's models"""

    value: float
    """The estimate value of the input (latent state/history)"""
    reward: float
    """The reward of a particular transition, seems to be ignored in initial inference"""
    policy: Dict[Action, float]
    """a => probability mapping"""
    latent_state: State
    """latent representation of the history/state"""


def create_muzero(
    initial_inference: Callable[[Any], MuzeroInferenceOutput],
    recurrent_inference: Callable[[State, Action], MuzeroInferenceOutput],
    num_sims: int,
    c1: float = 0.75,
    c2: float = 10,
    discount_factor: float = 0.95,
    root_noise_dirichlet_alpha: float = 10,
    root_noise_exploration_fraction: float = 0.2,
) -> Callable[[Any], Tuple[Action, Info]]:
    """Creates a muzero planner

    Runs :func:`deterministic_tree_search` like Muzero does. Initially maps
    some history representation to a latent state through `initial_inference`,
    then continues running the tree search by picking nodes through statistics
    in nodes and expanding nodes with `recurrent_inference`.

    Note that the 'type' of muzero is not well defined in this repository. The
    input is not a :class:`online_pomdp_planning.types.Belief`, but unclear
    what it is.

    :param initial_inference: the initial mapping and inference over history
    :param recurrent_inference: the recurrent/dynamics/evaluation of tree expansion
    :param num_sims: number of tree expansions/simulations done
    :param c1: first exploration parameter used in mu-zero's UCB scoring method, defaults to 0.75
    :param c2: second exploration parameter used in mu-zero's UCB scoring method, defaults to 10
    :param discount_factor: the discount factor of the environment, defaults to 0.95
    :param root_noise_dirichlet_alpha: variance of (Dirichlet) root prior noise, default to 10
    :param root_noise_exploration_fraction: fraction of noise added to root prior, default to .2
    """
    stop_cond = partial(has_simulated_n_times, num_sims)

    def tree_constructor(history_representation: Any):
        inference = initial_inference(history_representation)
        return create_muzero_root(
            inference.latent_state,
            inference.reward,
            inference.policy,
            root_noise_dirichlet_alpha,
            root_noise_exploration_fraction,
        )

    # must use UCB with bounds, and deterministic nodes and everything
    # basically `muzero_pseudocode.py:468-471`
    node_scoring_method = partial(muzero_ucb_scores, c1=c1, c2=c2)
    leaf_select = partial(select_deterministc_leaf_by_max_scores, node_scoring_method)

    expand_and_evaluate = partial(muzero_expand_node, inference=recurrent_inference)
    action_select = max_visits_action_selector
    backprop = partial(deterministic_qval_backpropagation, discount_factor)

    return partial(
        deterministic_tree_search,
        stop_cond,
        tree_constructor,
        leaf_select,
        expand_and_evaluate,
        backprop,
        action_select,
    )
